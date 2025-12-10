"""Readability metrics for Spanish text."""

import re
from typing import List, Optional


def count_syllables(word: str) -> int:
    """
    Count syllables in a Spanish word using heuristics.
    
    Args:
        word: Word to count syllables for
        
    Returns:
        Number of syllables
    """
    word = word.lower().strip()
    
    if not word:
        return 0
    
    # Remove punctuation
    word = re.sub(r'[^\wáéíóúüñ]', '', word)
    
    if not word:
        return 0
    
    # Count vowels (including accented vowels)
    vowels = 'aeiouáéíóúü'
    vowel_count = sum(1 for char in word if char.lower() in vowels)
    
    # Handle diphthongs and triphthongs
    # Common Spanish diphthongs: ai, ei, oi, ui, au, eu, ou, ia, ie, io, ua, ue, uo
    diphthongs = [
        'ai', 'ei', 'oi', 'ui', 'au', 'eu', 'ou',
        'ia', 'ie', 'io', 'ua', 'ue', 'uo',
        'ay', 'ey', 'oy', 'uy',
        'ya', 'ye', 'yi', 'yo', 'yu',
    ]
    
    # Count diphthongs
    diphthong_count = 0
    word_lower = word.lower()
    for diphthong in diphthongs:
        diphthong_count += word_lower.count(diphthong)
    
    # Adjust: each diphthong reduces syllable count by 1
    # (since two vowels together count as one syllable)
    syllables = max(1, vowel_count - diphthong_count)
    
    # Handle silent 'h' (doesn't affect syllable count)
    # Handle word-final 'e' that might be silent in some contexts
    # These are already handled by vowel counting
    
    return syllables


def compute_szigriszt_pazos(text: str) -> float:
    """
    Compute Szigriszt-Pazos readability index for Spanish text.
    
    Formula: SP = 206.84 - (62.3 * (syllables / words)) - (words / sentences)
    
    Higher scores indicate easier text.
    
    Args:
        text: Input text
        
    Returns:
        Szigriszt-Pazos readability score
    """
    if not text or not text.strip():
        return 0.0
    
    # Split into sentences (simple heuristic: split on . ! ?)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 0.0
    
    # Count words and syllables
    total_words = 0
    total_syllables = 0
    
    for sentence in sentences:
        words = sentence.split()
        total_words += len(words)
        
        for word in words:
            syllables = count_syllables(word)
            total_syllables += syllables
    
    if total_words == 0:
        return 0.0
    
    # Compute index
    avg_syllables_per_word = total_syllables / total_words
    avg_words_per_sentence = total_words / len(sentences)
    
    sp_index = 206.84 - (62.3 * avg_syllables_per_word) - avg_words_per_sentence
    
    return sp_index


def compute_szigriszt_pazos_batch(texts: List[str]) -> List[float]:
    """
    Compute Szigriszt-Pazos index for multiple texts.
    
    Args:
        texts: List of input texts
        
    Returns:
        List of readability scores
    """
    return [compute_szigriszt_pazos(text) for text in texts]

