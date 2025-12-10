"""DPGD LogitsProcessor for applying difficulty-based penalties during generation."""

import warnings
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, PreTrainedTokenizer, PreTrainedTokenizerFast

from ..embeddings.difficulty_embeddings import DifficultyEmbeddingSet
from ..embeddings.embedding_cache import extract_embedding_matrix, get_token_embedding
from ..profiles.profile_schema import HybridDifficultyProfile


class DPGDLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor that applies Dynamic Profile-Guided Decoding penalties.
    
    Applies three types of penalties:
    1. Semantic Distance Penalty (SDP): Based on cosine similarity to difficulty embeddings
       - Penalizes tokens that are semantically CLOSE to difficult word prototypes
    2. Frequency Penalty: Based on word frequency scores (for tokens completing difficult words)
    3. Morphology Penalty: Based on morphological complexity (for tokens completing difficult words)
    4. Masking: Completely mask unknown words (K_w = 1 means unknown)
    """
    
    def __init__(
        self,
        profile: HybridDifficultyProfile,
        embedding_set: DifficultyEmbeddingSet,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        model: torch.nn.Module,
        alpha_SDP: float = 1.0,
        alpha_Freq: float = 1.0,
        alpha_Morph: float = 1.0,
        delta_SDP: float = 0.5,
        lambda_strength: float = 1.0,
        max_decode_tokens: int = 20,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the DPGD LogitsProcessor.
        
        Args:
            profile: HybridDifficultyProfile with word difficulty scores
            embedding_set: DifficultyEmbeddingSet with normalized embeddings
            tokenizer: HuggingFace tokenizer
            model: Model to extract embeddings from
            alpha_SDP: Weight for Semantic Distance Penalty
            alpha_Freq: Weight for frequency penalty
            alpha_Morph: Weight for morphology penalty
            delta_SDP: Threshold/margin for SDP calculation. Must be > 0 for SDP
                to have any effect. Paper specifies δ ∈ (0, 2]. Default: 0.5
            lambda_strength: Global personalization strength scaling all penalties
            max_decode_tokens: Maximum number of tokens to decode for word boundary detection
            device: Device to run computations on
        """
        # Validate delta_SDP
        if delta_SDP <= 0 and alpha_SDP > 0:
            warnings.warn(
                f"delta_SDP={delta_SDP} <= 0 will disable Semantic Distance Penalty entirely. "
                f"Per paper, δ should be in (0, 2]. Consider using delta_SDP > 0.",
                UserWarning,
            )
        self.profile = profile
        self.embedding_set = embedding_set
        self.tokenizer = tokenizer
        self.model = model
        self.alpha_SDP = alpha_SDP
        self.alpha_Freq = alpha_Freq
        self.alpha_Morph = alpha_Morph
        self.delta_SDP = delta_SDP
        self.lambda_strength = lambda_strength
        self.max_decode_tokens = max_decode_tokens
        
        # Get device
        if device is None:
            device = next(model.parameters()).device
        self.device = device
        
        # Extract embedding matrix for candidate token embeddings
        self.embedding_matrix = extract_embedding_matrix(model, normalize=True)
        self.embedding_matrix = self.embedding_matrix.to(device)
        
        # Move embedding set to device
        self.embedding_set = embedding_set.to(device)
        
        # Cache for decoded sequences (optimized: only cache recent tokens)
        self._decoded_cache: Dict[Tuple[int, ...], str] = {}
        # Track last processed sequence to enable incremental updates
        self._last_seq_ids: Dict[int, Tuple[int, ...]] = {}
        
        # Build prefix tree for efficient word matching
        self._build_prefix_tree()
        
        # Pre-compute token mappings for difficult words (performance optimization)
        self._precompute_token_mappings()
    
    def _build_prefix_tree(self) -> None:
        """Build a prefix tree (trie) for efficient word prefix matching."""
        self.prefix_tree: Dict[str, Set[str]] = {}
        
        for word in self.profile.V_difficult:
            word_lower = word.lower()
            # Add all prefixes of the word
            for i in range(1, len(word_lower) + 1):
                prefix = word_lower[:i]
                if prefix not in self.prefix_tree:
                    self.prefix_tree[prefix] = set()
                self.prefix_tree[prefix].add(word_lower)
    
    def _precompute_token_mappings(self) -> None:
        """
        Pre-compute token mappings for all difficult words to avoid on-the-fly tokenization.
        
        Creates two caches:
        1. _word_start_tokens: Maps first token of each difficult word to its penalties.
           Used at word boundaries to prevent STARTING difficult words.
        2. _prefix_to_tokens: Maps character prefixes to continuation tokens.
           Used mid-word to prevent COMPLETING difficult words.
        
        This ensures deterministic penalties apply both when:
        - Starting a new word that is difficult (word boundary case)
        - Continuing a partial word toward a difficult completion
        """
        # Cache for word-START tokens: token_id -> (word, F_norm, M_w, is_unknown)
        # These are the FIRST tokens of difficult words, used at word boundaries
        self._word_start_tokens: Dict[int, Tuple[str, float, float, bool]] = {}
        
        # Cache for word-CONTINUATION: prefix -> {token_id: (word, F_norm, M_w, is_unknown)}
        self._prefix_to_tokens: Dict[str, Dict[int, Tuple[str, float, float, bool]]] = {}
        
        for word in self.profile.V_difficult:
            word_lower = word.lower()
            word_profile = self.profile.get_word_profile(word)
            if not word_profile:
                continue
            
            is_unknown = word_profile.K_w == 1
            
            # === Word-START case: tokenize full word, get first token ===
            # This handles the word boundary case where we want to prevent
            # STARTING a difficult word. Only actual word tokens will be here,
            # never pure whitespace or punctuation.
            full_encoding = self.tokenizer(
                word_lower,
                add_special_tokens=False,
                return_tensors="pt",
            )
            full_tokens = full_encoding["input_ids"][0].tolist()
            
            if full_tokens:
                first_token_id = full_tokens[0]
                
                # Store with max penalties if multiple difficult words share first token
                if first_token_id in self._word_start_tokens:
                    _, existing_f, existing_m, existing_unknown = self._word_start_tokens[first_token_id]
                    self._word_start_tokens[first_token_id] = (
                        word_lower,
                        max(existing_f, word_profile.F_norm),
                        max(existing_m, word_profile.M_w),
                        existing_unknown or is_unknown,
                    )
                else:
                    self._word_start_tokens[first_token_id] = (
                        word_lower, word_profile.F_norm, word_profile.M_w, is_unknown
                    )
            
            # === Word-CONTINUATION case: tokenize suffixes for each prefix ===
            for prefix_len in range(1, len(word_lower)):
                prefix = word_lower[:prefix_len]
                suffix = word_lower[prefix_len:]
                
                if not suffix:
                    continue
                
                # Tokenize the suffix
                suffix_encoding = self.tokenizer(
                    suffix,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                suffix_tokens = suffix_encoding["input_ids"][0].tolist()
                
                if not suffix_tokens:
                    continue
                
                suffix_first_token = suffix_tokens[0]
                
                # Initialize prefix dict if needed
                if prefix not in self._prefix_to_tokens:
                    self._prefix_to_tokens[prefix] = {}
                
                # Store with max penalties if multiple words share same prefix+token
                if suffix_first_token in self._prefix_to_tokens[prefix]:
                    _, existing_f, existing_m, existing_unknown = self._prefix_to_tokens[prefix][suffix_first_token]
                    self._prefix_to_tokens[prefix][suffix_first_token] = (
                        word_lower,
                        max(existing_f, word_profile.F_norm),
                        max(existing_m, word_profile.M_w),
                        existing_unknown or is_unknown,
                    )
                else:
                    self._prefix_to_tokens[prefix][suffix_first_token] = (
                        word_lower, word_profile.F_norm, word_profile.M_w, is_unknown
                    )
    
    def _get_current_word_prefix(
        self,
        input_ids: torch.Tensor,
        batch_idx: int = 0,
    ) -> Tuple[str, bool]:
        """
        Determine the current word being generated from input_ids.
        
        Optimized: Only decodes the last N tokens instead of entire sequence.
        
        Args:
            input_ids: Current input token IDs [batch_size, seq_len]
            batch_idx: Batch index to process
            
        Returns:
            Tuple of (current_word_prefix, is_complete_word)
            - current_word_prefix: The partial word being generated (lowercased)
            - is_complete_word: True if the word is complete (ends with space or punctuation)
        """
        seq_ids = input_ids[batch_idx].cpu().tolist()
        seq_key = tuple(seq_ids)
        
        # Optimization: Only decode the last N tokens for word boundary detection
        # This avoids decoding the entire sequence on every step
        num_tokens_to_decode = min(self.max_decode_tokens, len(seq_ids))
        recent_token_ids = seq_ids[-num_tokens_to_decode:]
        recent_key = tuple(recent_token_ids)
        
        # Check cache for recent tokens
        if recent_key in self._decoded_cache:
            decoded_recent = self._decoded_cache[recent_key]
        else:
            # Only decode recent tokens
            decoded_recent = self.tokenizer.decode(recent_token_ids, skip_special_tokens=True)
            self._decoded_cache[recent_key] = decoded_recent
            # Limit cache size to prevent memory issues
            if len(self._decoded_cache) > 1000:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self._decoded_cache))
                del self._decoded_cache[oldest_key]
        
        # Get the last token to see what's being generated
        last_token_id = seq_ids[-1]
        last_token = self.tokenizer.decode([last_token_id], skip_special_tokens=True)
        
        # Clean the decoded text
        text_clean = decoded_recent.strip()
        
        if not text_clean:
            return "", False
        
        # Find the last word boundary
        # Look for the last space or punctuation
        import string
        punctuation = set(string.punctuation + ' ')
        
        # Find the start of the last word
        word_start = len(text_clean)
        for i in range(len(text_clean) - 1, -1, -1):
            if text_clean[i] in punctuation:
                word_start = i + 1
                break
        else:
            # No punctuation found, entire text is one word
            word_start = 0
        
        # Extract the last word (might be partial)
        last_word = text_clean[word_start:].lower()
        
        # Check if word is complete
        # Handle various tokenizer conventions for word boundaries:
        # - Standard: spaces or punctuation in the decoded token
        # - SentencePiece: token starts with '▁' (U+2581) indicating word start
        # - GPT-2/BPE: token starts with 'Ġ' (U+0120) indicating space prefix
        # - Llama/other: may use different markers
        
        # Check for tokenizer-specific word-start markers in the raw token
        raw_last_token = self.tokenizer.convert_ids_to_tokens([last_token_id])[0] if last_token_id < len(self.tokenizer) else ""
        has_word_boundary_marker = (
            raw_last_token.startswith('▁') or      # SentencePiece
            raw_last_token.startswith('Ġ') or      # GPT-2 style
            raw_last_token.startswith('<0x20>') or  # Some tokenizers use hex space
            raw_last_token.startswith(' ')          # Direct space prefix
        )
        
        is_complete = (
            has_word_boundary_marker or
            last_token.strip() != last_token or
            any(c in punctuation for c in last_token)
        )
        
        # Remove trailing punctuation from word for matching
        last_word_clean = last_word.rstrip(string.punctuation)
        
        return last_word_clean, is_complete
    
    def _compute_semantic_distance_penalty(
        self,
        candidate_token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Semantic Distance Penalty (SDP) for candidate tokens.
        
        Per paper:
        - D_min(e_i) = 1 - max_similarity (distance to nearest difficult prototype)
        - P_SDP(t_i) = gamma * S_{w*} * max(0, delta - D_min)
                     = gamma * S_{w*} * max(0, max_sim - (1 - delta))
        
        Tokens CLOSE to difficult prototypes (high similarity) get penalized.
        
        Args:
            candidate_token_ids: Token IDs to compute penalties for [vocab_size]
            
        Returns:
            Tuple of:
            - P_SDP: Penalty scores [vocab_size]
            - max_similarities: Raw similarity scores [vocab_size]
            - nearest_indices: Index of nearest difficult word for each token [vocab_size]
        """
        vocab_size = candidate_token_ids.shape[0]
        
        # Handle empty embedding set edge case
        if self.embedding_set.num_words == 0:
            return (
                torch.zeros(vocab_size, device=self.device),
                torch.zeros(vocab_size, device=self.device),
                torch.zeros(vocab_size, dtype=torch.long, device=self.device),
            )
        
        # Get embeddings for all candidate tokens
        # Shape: [vocab_size, hidden_dim]
        candidate_embeddings = self.embedding_matrix[candidate_token_ids]
        
        # Compute cosine similarity between candidates and difficulty embeddings
        # E_diff shape: [num_difficult_words, hidden_dim]
        # candidate_embeddings shape: [vocab_size, hidden_dim]
        # Result: [vocab_size, num_difficult_words]
        similarities = torch.matmul(
            candidate_embeddings,
            self.embedding_set.E_diff.t()
        )  # Cosine similarity (embeddings are normalized)
        
        # Get max similarity and the index of nearest difficult word
        # Shape: [vocab_size]
        max_similarities, nearest_indices = torch.max(similarities, dim=1)

        # Per paper: P_SDP = gamma * S_w* * max(0, delta - D_min)
        # Where D_min = 1 - max_sim.
        #
        # We absorb gamma into alpha_SDP in the logits processor, so here we only
        # compute S_w* * max(0, delta - D_min). When delta_SDP <= 0, this
        # correctly yields zero SDP for all tokens (no semantic penalty).

        # Weight by S_w of nearest difficult word
        S_scores_tensor = torch.tensor(self.embedding_set.S_scores, device=self.device)
        nearest_S_w = S_scores_tensor[nearest_indices]

        # Margin-based formulation from the paper (always used).
        # When delta_SDP <= 0, margin_violation becomes zero everywhere,
        # so SDP has no effect, which matches the theoretical definition.
        D_min = 1.0 - max_similarities
        margin_violation = torch.clamp(self.delta_SDP - D_min, min=0.0)
        P_SDP = nearest_S_w * margin_violation
        
        return P_SDP, max_similarities, nearest_indices
    
    def _get_completing_tokens(
        self,
        current_word_prefix: str,
        at_word_boundary: bool = False,
    ) -> Dict[int, Tuple[str, float, float, bool]]:
        """
        Find tokens that would complete or start a difficult word.
        
        Uses pre-computed token mappings for O(1) lookup instead of on-the-fly tokenization.
        
        Per the paper's idealized φ^(k) definition (Section 3.3):
        - A^(k)(t_i) contains difficult words w where Tok(w) has the current suffix
          followed by t_i as a prefix.
        - At word boundaries (empty suffix), this means t_i is a prefix of Tok(w),
          i.e., t_i could be the FIRST token of a difficult word.
        
        This implementation handles both cases:
        1. Word boundary (at_word_boundary=True): Return first tokens of difficult words
        2. Mid-word (prefix provided): Return continuation tokens for that prefix
        
        Args:
            current_word_prefix: Current partial word being generated (may be empty)
            at_word_boundary: If True, we're at a word boundary and should return
                              word-start tokens instead of continuation tokens
            
        Returns:
            Dict mapping token_id -> (completed_word, F_norm, M_w, is_unknown)
        """
        if at_word_boundary or not current_word_prefix:
            # At word boundary: return tokens that would START a difficult word.
            # These are the first tokens of each difficult word.
            # Only genuine word tokens are in this set (no whitespace/punctuation).
            return self._word_start_tokens.copy()
        
        # Mid-word: use pre-computed prefix->token mappings for O(1) lookup
        prefix_tokens = self._prefix_to_tokens.get(current_word_prefix, {})
        
        completing_tokens: Dict[int, Tuple[str, float, float, bool]] = {}
        
        # Copy tokens, using max penalties for any conflicts
        for token_id, (word, f_norm, m_w, is_unknown) in prefix_tokens.items():
            if token_id in completing_tokens:
                _, existing_f, existing_m, existing_unknown = completing_tokens[token_id]
                completing_tokens[token_id] = (
                    word,
                    max(existing_f, f_norm),
                    max(existing_m, m_w),
                    existing_unknown or is_unknown,
                )
            else:
                completing_tokens[token_id] = (word, f_norm, m_w, is_unknown)
        
        return completing_tokens
    
    def _compute_deterministic_penalties(
        self,
        vocab_size: int,
        current_word_prefix: str,
        is_complete_word: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute deterministic penalties (frequency, morphology, masking).
        
        Targets SPECIFIC tokens that would either:
        1. START a difficult word (at word boundaries)
        2. COMPLETE a difficult word (when a prefix is being formed)
        
        Args:
            vocab_size: Size of vocabulary
            current_word_prefix: Current partial word being generated
            is_complete_word: Whether the current word is complete (at word boundary)
            
        Returns:
            Tuple of (frequency_penalties, morphology_penalties, mask_tokens)
            All have shape [vocab_size]
        """
        freq_penalties = torch.zeros(vocab_size, device=self.device)
        morph_penalties = torch.zeros(vocab_size, device=self.device)
        mask_tokens = torch.zeros(vocab_size, dtype=torch.bool, device=self.device)
        
        # Determine which tokens to penalize based on context:
        # - At word boundary (is_complete_word=True): penalize word-START tokens
        # - Mid-word (is_complete_word=False): penalize word-CONTINUATION tokens
        completing_tokens = self._get_completing_tokens(
            current_word_prefix,
            at_word_boundary=is_complete_word,
        )
        
        for token_id, (word, F_norm, M_w, is_unknown) in completing_tokens.items():
            if token_id < vocab_size:
                freq_penalties[token_id] = F_norm
                morph_penalties[token_id] = M_w
                if is_unknown:
                    mask_tokens[token_id] = True
        
        return freq_penalties, morph_penalties, mask_tokens
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply DPGD penalties to logits.
        
        Per paper, modified logit:
            z'_i = z_i - lambda * P(t_i)
        
        Where P(t_i) combines:
            - alpha_SDP * P_SDP(t_i): semantic similarity to difficult prototypes
            - alpha_Freq * F_norm: frequency-based difficulty (for completing tokens)
            - alpha_Morph * M_w: morphological complexity (for completing tokens)
            - C_mask * 1[K_w=1]: hard mask for unknown words
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            scores: Logits scores [batch_size, vocab_size]
            
        Returns:
            Modified scores with penalties applied
        """
        batch_size, vocab_size = scores.shape
        
        # Process each batch item
        for batch_idx in range(batch_size):
            # Step A: Determine current word being generated
            current_word_prefix, is_complete_word = self._get_current_word_prefix(
                input_ids,
                batch_idx=batch_idx,
            )
            
            # Get candidate token IDs (all vocabulary)
            candidate_token_ids = torch.arange(vocab_size, device=self.device)
            
            # Step B: Compute Semantic Distance Penalty (applies to ALL tokens)
            # Tokens semantically close to difficult words get penalized
            P_SDP, _, _ = self._compute_semantic_distance_penalty(candidate_token_ids)
            sdp_penalty = self.alpha_SDP * P_SDP
            
            # Step C: Compute Deterministic Penalties (applies to SPECIFIC tokens)
            # Only tokens that would complete a difficult word get F_norm/M_w penalties
            freq_penalties, morph_penalties, mask_tokens = self._compute_deterministic_penalties(
                vocab_size,
                current_word_prefix,
                is_complete_word,
            )
            
            freq_penalty = self.alpha_Freq * freq_penalties
            morph_penalty = self.alpha_Morph * morph_penalties
            
            # Step D: Apply penalties
            # Total penalty = SDP + Frequency + Morphology
            total_penalty = sdp_penalty + freq_penalty + morph_penalty

            # Apply global personalization strength (lambda)
            scaled_penalty = self.lambda_strength * total_penalty
            
            # Subtract penalties from scores (higher penalty = lower score = less likely)
            scores[batch_idx] = scores[batch_idx] - scaled_penalty
            
            # Apply masking for unknown words (set to -inf)
            # Only mask SPECIFIC tokens that would complete unknown words
            if mask_tokens.any():
                scores[batch_idx][mask_tokens] = float('-inf')
        
        return scores

