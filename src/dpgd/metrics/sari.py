"""SARI (System output Against References and Input) metric implementation."""

from typing import List, Optional


def compute_sari(
    sources: List[str],
    predictions: List[str],
    references: List[List[str]],
) -> float:
    """
    Compute SARI score.
    
    SARI measures how well the system output balances:
    - Keeping appropriate words from the source
    - Adding appropriate words from references
    - Deleting inappropriate words from the source
    
    Args:
        sources: List of source sentences
        predictions: List of predicted/simplified sentences
        references: List of reference sentences (each item is a list of reference strings)
        
    Returns:
        SARI score (0-1, higher is better)
    """
    try:
        from easse.sari import corpus_sari
    except ImportError:
        # Fallback: simple implementation
        return _simple_sari(sources, predictions, references)
    
    # EASSE expects references as list of lists
    # Ensure references are in correct format
    formatted_refs = []
    for ref_list in references:
        if isinstance(ref_list, str):
            formatted_refs.append([ref_list])
        else:
            formatted_refs.append(ref_list)
    
    return corpus_sari(
        orig_sents=sources,
        sys_sents=predictions,
        refs_sents=formatted_refs,
    )


def _simple_sari(
    sources: List[str],
    predictions: List[str],
    references: List[List[str]],
) -> float:
    """
    Simple SARI implementation as fallback.
    
    This is a simplified version. For accurate results, install easse:
    pip install easse
    """
    import warnings
    
    warnings.warn(
        "easse not installed. Using simplified SARI implementation. "
        "Install with: pip install easse",
        UserWarning,
    )
    
    # Very basic implementation
    # In practice, you should use the easse library
    total_score = 0.0
    count = 0
    
    for source, pred, ref_list in zip(sources, predictions, references):
        # Simple word overlap metric
        source_words = set(source.lower().split())
        pred_words = set(pred.lower().split())
        
        # Average over references
        ref_scores = []
        for ref in ref_list:
            ref_words = set(ref.lower().split())
            
            # Keep score: words in both source and prediction
            keep = len(source_words & pred_words)
            
            # Add score: words in prediction and reference but not in source
            add = len((pred_words & ref_words) - source_words)
            
            # Delete score: words in source but not in prediction or reference
            delete = len(source_words - pred_words - ref_words)
            
            # Simple average
            score = (keep + add + delete) / max(len(source_words) + len(pred_words), 1)
            ref_scores.append(score)
        
        total_score += sum(ref_scores) / len(ref_scores) if ref_scores else 0.0
        count += 1
    
    return total_score / count if count > 0 else 0.0

