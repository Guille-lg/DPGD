"""Tokenization utilities for DPGD."""

from typing import Dict, List, Optional, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class TokenizerWrapper:
    """
    Wrapper around a HuggingFace tokenizer for DPGD.
    
    Provides a consistent interface for tokenization operations.
    """
    
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ):
        """
        Initialize the tokenizer wrapper.
        
        Args:
            tokenizer: HuggingFace tokenizer instance
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
    
    def encode(
        self,
        text: Union[str, List[str]],
        return_tensors: Optional[str] = None,
        add_special_tokens: bool = True,
    ) -> Dict[str, Union[List[int], List[List[int]]]]:
        """
        Encode text(s) into token IDs.
        
        Args:
            text: Input text or list of texts
            return_tensors: If "pt", return PyTorch tensors
            add_special_tokens: Whether to add special tokens (CLS, SEP, etc.)
            
        Returns:
            Dictionary with 'input_ids' and optionally 'attention_mask'
        """
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
        )
    
    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True,
    ) -> Union[str, List[str]]:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs or list of token ID lists
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text or list of texts
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens (strings).
        
        Args:
            text: Input text
            
        Returns:
            List of token strings
        """
        return self.tokenizer.tokenize(text)
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return len(self.tokenizer)
    
    @property
    def pad_token_id(self) -> Optional[int]:
        """Get the pad token ID."""
        return self.tokenizer.pad_token_id
    
    @property
    def cls_token_id(self) -> Optional[int]:
        """Get the CLS token ID."""
        return self.tokenizer.cls_token_id
    
    @property
    def sep_token_id(self) -> Optional[int]:
        """Get the SEP token ID."""
        return self.tokenizer.sep_token_id


def get_word_to_token_alignment(
    sentence: str,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, TokenizerWrapper],
) -> Dict[int, List[int]]:
    """
    Map each word in a sentence to its corresponding subword token indices.
    
    This function is critical for DPGD as it allows applying penalties to specific
    word parts (subword tokens) based on word-level complexity analysis.
    
    Args:
        sentence: Input sentence (space-separated words)
        tokenizer: HuggingFace tokenizer or TokenizerWrapper instance
        
    Returns:
        Dictionary mapping word index to list of token indices.
        Example: {0: [1, 2], 1: [3], 2: [4, 5, 6]} means:
        - Word 0 maps to tokens 1 and 2
        - Word 1 maps to token 3
        - Word 2 maps to tokens 4, 5, and 6
    
    Note:
        Token indices include special tokens (CLS at position 0, SEP at the end).
        The mapping accounts for subword tokenization (e.g., BPE, WordPiece).
    """
    # Extract actual tokenizer if wrapped
    if isinstance(tokenizer, TokenizerWrapper):
        tokenizer = tokenizer.tokenizer
    
    # Split sentence into words (preserving word boundaries)
    words = sentence.split()
    
    if not words:
        return {}
    
    # Tokenize the full sentence to get all tokens
    # Use return_offsets_mapping to get character-level alignment
    encoding = tokenizer(
        sentence,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    
    tokens = encoding.tokens()
    offsets = encoding.offset_mapping
    
    # Build word-to-token mapping
    word_to_tokens: Dict[int, List[int]] = {i: [] for i in range(len(words))}
    
    # Calculate word boundaries in the original sentence
    word_boundaries: List[tuple[int, int]] = []
    char_pos = 0
    for word in words:
        word_start = char_pos
        word_end = char_pos + len(word)
        word_boundaries.append((word_start, word_end))
        char_pos = word_end + 1  # +1 for the space after the word
    
    # Map tokens to words using character offsets
    for token_idx, (token, (char_start, char_end)) in enumerate(zip(tokens, offsets)):
        # Skip special tokens (they have offset (0, 0))
        if char_start == char_end == 0:
            continue
        
        # Find which word(s) this token overlaps with
        # A token can span multiple words in edge cases, but typically belongs to one word
        token_center = (char_start + char_end) // 2
        
        # Find the word that contains the token's center point
        for word_idx, (word_start, word_end) in enumerate(word_boundaries):
            if word_start <= token_center < word_end:
                word_to_tokens[word_idx].append(token_idx)
                break
            # Also check if token fully contains a word (edge case)
            elif char_start <= word_start and char_end >= word_end:
                word_to_tokens[word_idx].append(token_idx)
                break
    
    return word_to_tokens


def get_token_to_word_alignment(
    sentence: str,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, TokenizerWrapper],
) -> Dict[int, int]:
    """
    Map each token index to its corresponding word index (reverse of get_word_to_token_alignment).
    
    Args:
        sentence: Input sentence (space-separated words)
        tokenizer: HuggingFace tokenizer or TokenizerWrapper instance
        
    Returns:
        Dictionary mapping token index to word index.
        Example: {1: 0, 2: 0, 3: 1, 4: 2, 5: 2, 6: 2}
    """
    word_to_tokens = get_word_to_token_alignment(sentence, tokenizer)
    
    token_to_word: Dict[int, int] = {}
    for word_idx, token_indices in word_to_tokens.items():
        for token_idx in token_indices:
            token_to_word[token_idx] = word_idx
    
    return token_to_word

