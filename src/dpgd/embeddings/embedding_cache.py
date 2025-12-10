"""Utilities for extracting and caching model embeddings."""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel


def extract_embedding_matrix(
    model: PreTrainedModel,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Extract the static input embedding matrix from a HuggingFace model.
    
    Args:
        model: HuggingFace PreTrainedModel instance
        normalize: If True, normalize embeddings using L2 norm
        
    Returns:
        Embedding matrix tensor of shape [vocab_size, hidden_dim]
    """
    # Get the input embeddings layer
    try:
        embedding_layer = model.get_input_embeddings()
        if embedding_layer is None:
            raise ValueError("Model does not have input embeddings")
        
        # Extract the weight matrix
        # Shape: [vocab_size, hidden_dim]
        embedding_matrix = embedding_layer.weight.data.clone()
        
    except AttributeError:
        # Fallback: try to access embeddings directly
        if hasattr(model, "embeddings") and hasattr(model.embeddings, "word_embeddings"):
            embedding_matrix = model.embeddings.word_embeddings.weight.data.clone()
        elif hasattr(model, "shared"):
            # For some models (e.g., T5), embeddings are in shared layer
            embedding_matrix = model.shared.weight.data.clone()
        else:
            raise ValueError(
                "Could not extract embedding matrix from model. "
                "Model must have get_input_embeddings() method or embeddings attribute."
            )
    
    # Normalize if requested
    if normalize:
        embedding_matrix = normalize_embeddings(embedding_matrix)
    
    return embedding_matrix


def normalize_embeddings(embeddings: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Normalize embeddings using L2 norm.
    
    Args:
        embeddings: Embedding tensor of shape [..., hidden_dim]
        dim: Dimension along which to normalize (default: -1, last dimension)
        
    Returns:
        Normalized embedding tensor with same shape
    """
    # Compute L2 norm along specified dimension
    norm = torch.norm(embeddings, p=2, dim=dim, keepdim=True)
    
    # Avoid division by zero
    norm = torch.clamp(norm, min=1e-8)
    
    # Normalize
    normalized = embeddings / norm
    
    return normalized


def get_token_embedding(
    token_id: int,
    embedding_matrix: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Get embedding for a specific token ID.
    
    Args:
        token_id: Token ID (vocab index)
        embedding_matrix: Embedding matrix [vocab_size, hidden_dim]
        normalize: If True, normalize the returned embedding
        
    Returns:
        Embedding vector of shape [hidden_dim]
    """
    if token_id < 0 or token_id >= embedding_matrix.shape[0]:
        raise ValueError(
            f"Token ID {token_id} out of range [0, {embedding_matrix.shape[0]})"
        )
    
    embedding = embedding_matrix[token_id].clone()
    
    if normalize:
        embedding = normalize_embeddings(embedding.unsqueeze(0)).squeeze(0)
    
    return embedding


def get_token_embeddings(
    token_ids: Union[List[int], torch.Tensor],
    embedding_matrix: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Get embeddings for multiple token IDs.
    
    Args:
        token_ids: List or tensor of token IDs
        embedding_matrix: Embedding matrix [vocab_size, hidden_dim]
        normalize: If True, normalize the returned embeddings
        
    Returns:
        Embedding tensor of shape [num_tokens, hidden_dim]
    """
    if isinstance(token_ids, list):
        token_ids = torch.tensor(token_ids, dtype=torch.long)
    
    # Ensure token_ids are within valid range
    max_id = embedding_matrix.shape[0] - 1
    token_ids = torch.clamp(token_ids, min=0, max=max_id)
    
    embeddings = embedding_matrix[token_ids]
    
    if normalize:
        embeddings = normalize_embeddings(embeddings)
    
    return embeddings

