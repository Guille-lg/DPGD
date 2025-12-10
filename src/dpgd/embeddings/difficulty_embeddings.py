"""Construction of Hybrid Difficulty Embedding Set."""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from ..profiles.profile_schema import HybridDifficultyProfile
from ..data.tokenization import TokenizerWrapper
from .embedding_cache import extract_embedding_matrix, get_token_embeddings, normalize_embeddings


class DifficultyEmbeddingSet:
    """
    Container for difficulty word embeddings and their scores.
    
    Stores:
    - E_diff: Tensor of shape [num_difficult_words, hidden_dim] with normalized embeddings
    - S_scores: List of composite difficulty scores S_w for each word
    - word_to_idx: Mapping from word to index in E_diff
    """
    
    def __init__(
        self,
        E_diff: torch.Tensor,
        S_scores: List[float],
        words: List[str],
        word_to_idx: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize the difficulty embedding set.
        
        Args:
            E_diff: Embedding tensor [num_difficult_words, hidden_dim]
            S_scores: List of composite scores S_w
            words: List of difficult words (in same order as embeddings)
            word_to_idx: Optional mapping from word to index
        """
        self.E_diff = E_diff
        self.S_scores = S_scores
        self.words = words
        
        if word_to_idx is None:
            self.word_to_idx = {word: idx for idx, word in enumerate(words)}
        else:
            self.word_to_idx = word_to_idx
    
    @property
    def num_words(self) -> int:
        """Get the number of difficult words."""
        return self.E_diff.shape[0]
    
    @property
    def hidden_dim(self) -> int:
        """Get the hidden dimension of embeddings."""
        return self.E_diff.shape[1]
    
    def get_embedding(self, word: str) -> Optional[torch.Tensor]:
        """
        Get embedding for a specific word.
        
        Args:
            word: Word to look up
            
        Returns:
            Embedding tensor [hidden_dim] if word exists, None otherwise
        """
        idx = self.word_to_idx.get(word.lower())
        if idx is None:
            return None
        return self.E_diff[idx]
    
    def get_score(self, word: str) -> Optional[float]:
        """
        Get difficulty score for a specific word.
        
        Args:
            word: Word to look up
            
        Returns:
            Score S_w if word exists, None otherwise
        """
        idx = self.word_to_idx.get(word.lower())
        if idx is None:
            return None
        return self.S_scores[idx]
    
    def to(self, device: Union[str, torch.device]) -> "DifficultyEmbeddingSet":
        """
        Move embeddings to a device.
        
        Args:
            device: Target device (e.g., "cuda", "cpu")
            
        Returns:
            New DifficultyEmbeddingSet on the target device
        """
        return DifficultyEmbeddingSet(
            E_diff=self.E_diff.to(device),
            S_scores=self.S_scores.copy(),
            words=self.words.copy(),
            word_to_idx=self.word_to_idx.copy(),
        )


def build_difficulty_embeddings(
    profile: HybridDifficultyProfile,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, TokenizerWrapper],
    model: PreTrainedModel,
    device: Optional[Union[str, torch.device]] = None,
) -> DifficultyEmbeddingSet:
    """
    Build the Hybrid Difficulty Embedding Set from a profile.
    
    For each difficult word w in the profile:
    1. Tokenize w into subwords [t1, t2, ...]
    2. Retrieve embeddings for each subword token
    3. Average them to get e_w
    4. Normalize e_w to get tilde_e_w (normalized embedding)
    
    Args:
        profile: HybridDifficultyProfile with difficult words identified
        tokenizer: HuggingFace tokenizer or TokenizerWrapper
        model: HuggingFace model to extract embeddings from
        device: Optional device to move tensors to (default: model's device)
        
    Returns:
        DifficultyEmbeddingSet containing:
        - E_diff: Tensor [num_difficult_words, hidden_dim] with normalized embeddings
        - S_scores: List of composite scores S_w
        - words: List of difficult words
    """
    # Extract actual tokenizer if wrapped
    if isinstance(tokenizer, TokenizerWrapper):
        tokenizer = tokenizer.tokenizer
    
    # Get device
    if device is None:
        device = next(model.parameters()).device
    
    # Extract and normalize embedding matrix
    embedding_matrix = extract_embedding_matrix(model, normalize=False)
    embedding_matrix = embedding_matrix.to(device)
    
    # Get difficult words from profile
    difficult_words = list(profile.V_difficult)
    
    if not difficult_words:
        # Return empty set
        return DifficultyEmbeddingSet(
            E_diff=torch.empty(0, embedding_matrix.shape[1], device=device),
            S_scores=[],
            words=[],
        )
    
    # Process each difficult word
    word_embeddings = []
    S_scores = []
    valid_words = []
    
    if len(difficult_words) > 10:
        print(f"      Processing {len(difficult_words)} difficult words...")
    
    for i, word in enumerate(difficult_words):
        if len(difficult_words) > 10 and (i + 1) % max(1, len(difficult_words) // 5) == 0:
            print(f"        Progress: {i + 1}/{len(difficult_words)} words processed")
        # Get word profile and score
        word_profile = profile.get_word_profile(word)
        if word_profile is None:
            continue
        
        S_w = word_profile.S_w
        S_scores.append(S_w)
        valid_words.append(word)
        
        # Tokenize word into subwords
        # We tokenize without special tokens to get only the word's tokens
        encoding = tokenizer(
            word,
            add_special_tokens=False,
            return_tensors="pt",
        )
        
        token_ids = encoding["input_ids"][0].tolist()  # Remove batch dimension
        
        if not token_ids:
            # If tokenization fails, skip this word
            # Use a zero embedding as fallback
            word_emb = torch.zeros(embedding_matrix.shape[1], device=device)
        else:
            # Get embeddings for each subword token
            # Shape: [num_subwords, hidden_dim]
            subword_embeddings = get_token_embeddings(
                token_ids,
                embedding_matrix,
                normalize=False,
            )
            
            # Average subword embeddings to get word embedding e_w
            # Shape: [hidden_dim]
            word_emb = subword_embeddings.mean(dim=0)
        
        # Normalize to get tilde_e_w
        # Shape: [hidden_dim]
        word_emb_normalized = normalize_embeddings(word_emb.unsqueeze(0)).squeeze(0)
        
        word_embeddings.append(word_emb_normalized)
    
    # Stack all word embeddings into tensor
    # Shape: [num_difficult_words, hidden_dim]
    E_diff = torch.stack(word_embeddings, dim=0)
    
    # Create embedding set
    embedding_set = DifficultyEmbeddingSet(
        E_diff=E_diff,
        S_scores=S_scores,
        words=valid_words,
    )
    
    return embedding_set


def compute_word_embedding(
    word: str,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, TokenizerWrapper],
    embedding_matrix: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute embedding for a single word by averaging subword embeddings.
    
    Args:
        word: Word to embed
        tokenizer: HuggingFace tokenizer or TokenizerWrapper
        embedding_matrix: Embedding matrix [vocab_size, hidden_dim]
        normalize: If True, normalize the final embedding
        
    Returns:
        Word embedding tensor of shape [hidden_dim]
    """
    # Extract actual tokenizer if wrapped
    if isinstance(tokenizer, TokenizerWrapper):
        tokenizer = tokenizer.tokenizer
    
    # Tokenize word
    encoding = tokenizer(
        word,
        add_special_tokens=False,
        return_tensors="pt",
    )
    
    token_ids = encoding["input_ids"][0].tolist()
    
    if not token_ids:
        # Return zero embedding if tokenization fails
        word_emb = torch.zeros(embedding_matrix.shape[1], device=embedding_matrix.device)
    else:
        # Get subword embeddings
        subword_embeddings = get_token_embeddings(
            token_ids,
            embedding_matrix,
            normalize=False,
        )
        
        # Average to get word embedding
        word_emb = subword_embeddings.mean(dim=0)
    
    # Normalize if requested
    if normalize:
        word_emb = normalize_embeddings(word_emb.unsqueeze(0)).squeeze(0)
    
    return word_emb

