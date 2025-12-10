"""Profile Hit Rate (PHR) metric implementation."""

import re
from typing import List, Set

from ..profiles.profile_schema import HybridDifficultyProfile


def extract_words(text: str) -> Set[str]:
    """
    Extract words from text (lowercased, punctuation removed).
    
    Args:
        text: Input text
        
    Returns:
        Set of words
    """
    # Remove punctuation and split
    words = re.findall(r'\b\w+\b', text.lower())
    return set(words)


def compute_phr(
    sources: List[str],
    predictions: List[str],
    profile: HybridDifficultyProfile,
) -> float:
    """
    Compute Profile Hit Rate (PHR).
    
    Formula: PHR = 1 - (|V_violated| / |V_active|)
    
    Where:
    - V_violated: Difficult words that appeared in the output (should be avoided)
    - V_active: Difficult words present in Source or Output
    
    Higher scores indicate better performance (fewer difficult words in output).
    
    Args:
        sources: List of source sentences
        predictions: List of predicted/simplified sentences
        profile: HybridDifficultyProfile with difficult words
        
    Returns:
        PHR score (0-1, higher is better)
    """
    if not sources or not predictions:
        return 0.0
    
    if len(sources) != len(predictions):
        raise ValueError("Sources and predictions must have the same length")
    
    # Get set of difficult words from profile
    V_difficult = profile.V_difficult
    
    if not V_difficult:
        # No difficult words defined, return perfect score
        return 1.0
    
    # Convert to lowercase for matching
    V_difficult_lower = {w.lower() for w in V_difficult}
    
    # Collect sentence-level PHR scores for examples where difficult words are active
    sentence_phrs = []
    
    for source, prediction in zip(sources, predictions):
        # Extract words from source and prediction
        source_words = extract_words(source)
        pred_words = extract_words(prediction)
        
        # V_active: Difficult words present in Source or Output
        V_active = V_difficult_lower & (source_words | pred_words)
        
        # V_violated: Difficult words that appeared in the output (should be avoided)
        V_violated = V_difficult_lower & pred_words
        
        if len(V_active) == 0:
            # Skip examples where the profile is not active
            continue
        
        sent_phr = 1.0 - (len(V_violated) / len(V_active))
        sentence_phrs.append(sent_phr)
    
    if not sentence_phrs:
        # Profile never active: define PHR as perfect
        return 1.0
    
    # Corpus-level PHR: average over sentence-level scores
    phr = sum(sentence_phrs) / len(sentence_phrs)
    
    # Ensure PHR is in [0, 1]
    return max(0.0, min(1.0, phr))


def compute_phr_details(
    sources: List[str],
    predictions: List[str],
    profile: HybridDifficultyProfile,
) -> dict:
    """
    Compute PHR with detailed statistics.
    
    Args:
        sources: List of source sentences
        predictions: List of predicted/simplified sentences
        profile: HybridDifficultyProfile with difficult words
        
    Returns:
        Dictionary with PHR score and detailed statistics
    """
    if not sources or not predictions:
        return {
            "phr": 0.0,
            "total_violated": 0,
            "total_active": 0,
            "violated_words": [],
        }
    
    V_difficult = profile.V_difficult
    V_difficult_lower = {w.lower() for w in V_difficult}
    
    total_violated = 0
    total_active = 0
    all_violated_words = set()
    
    # Collect sentence-level PHR scores for detailed reporting
    sentence_phrs = []
    
    for source, prediction in zip(sources, predictions):
        source_words = extract_words(source)
        pred_words = extract_words(prediction)
        
        V_active = V_difficult_lower & (source_words | pred_words)
        V_violated = V_difficult_lower & pred_words
        
        total_active += len(V_active)
        total_violated += len(V_violated)
        all_violated_words.update(V_violated)
        
        if len(V_active) > 0:
            sent_phr = 1.0 - (len(V_violated) / len(V_active))
            sentence_phrs.append(sent_phr)
    
    phr = sum(sentence_phrs) / len(sentence_phrs) if sentence_phrs else 1.0
    
    return {
        "phr": max(0.0, min(1.0, phr)),
        "total_violated": total_violated,
        "total_active": total_active,
        "violated_words": sorted(list(all_violated_words)),
    }

