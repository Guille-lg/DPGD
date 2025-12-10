"""Schema for Hybrid Difficulty Profile."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Set


@dataclass
class WordProfile:
    """
    Profile for a single word type.
    
    Attributes:
        word: The word type (surface form)
        F_norm: Normalized frequency difficulty score (0-1, higher = rarer = more difficult)
        M_w: Morphological complexity score (0-1, higher = more complex)
        K_w: Unknown flag (1 = unknown/should be hard-masked, 0 = known)
        S_w: Composite difficulty score (higher = more difficult)
    """
    word: str
    F_norm: float = 0.0
    M_w: float = 0.0
    K_w: int = 0
    S_w: float = 0.0
    
    def compute_composite_score(
        self,
        beta_Freq: float,
        beta_Morph: float,
        beta_Mask: float,
    ) -> None:
        """
        Compute the composite score S_w using the formula:
        S_w = beta_Freq * F_norm + beta_Morph * M_w + beta_Mask * K_w
        
        Args:
            beta_Freq: Weight for frequency component
            beta_Morph: Weight for morphological complexity component
            beta_Mask: Weight for known/unknown component
        """
        self.S_w = (
            beta_Freq * self.F_norm
            + beta_Morph * self.M_w
            + beta_Mask * self.K_w
        )


@dataclass
class HybridDifficultyProfile:
    """
    Hybrid Difficulty Profile storing word-level difficulty scores.
    
    This profile maps word types to their difficulty components and composite scores.
    It maintains a set of "Difficult Words" (V_difficult) based on a threshold.
    """
    
    word_profiles: Dict[str, WordProfile] = field(default_factory=dict)
    V_difficult: Set[str] = field(default_factory=set)
    
    # Configuration parameters
    beta_Freq: float = 0.0
    beta_Morph: float = 0.0
    beta_Mask: float = 0.0
    threshold: float = 0.5
    
    def add_word_profile(self, profile: WordProfile) -> None:
        """
        Add a word profile to the profile.
        
        Args:
            profile: WordProfile instance to add
        """
        # Store profiles keyed by lowercased word to ensure consistent lookups
        self.word_profiles[profile.word.lower()] = profile
    
    def compute_all_scores(self) -> None:
        """Compute composite scores for all word profiles."""
        for profile in self.word_profiles.values():
            profile.compute_composite_score(
                self.beta_Freq,
                self.beta_Morph,
                self.beta_Mask,
            )
    
    def update_difficult_words(self, threshold: Optional[float] = None) -> None:
        """
        Update the set of difficult words based on composite scores.
        
        Words with S_w >= threshold are considered difficult.
        
        Args:
            threshold: Optional threshold override (uses self.threshold if None)
        """
        if threshold is None:
            threshold = self.threshold
        
        # Maintain difficult words as lowercased surface forms
        self.V_difficult = {
            word.lower()
            for word, profile in self.word_profiles.items()
            if profile.S_w >= threshold
        }
    
    def get_word_profile(self, word: str) -> Optional[WordProfile]:
        """
        Get the profile for a specific word.
        
        Args:
            word: Word to look up
            
        Returns:
            WordProfile if found, None otherwise
        """
        # Look up using lowercased key to match storage convention
        return self.word_profiles.get(word.lower())
    
    def is_difficult(self, word: str) -> bool:
        """
        Check if a word is in the difficult set.
        
        Args:
            word: Word to check
            
        Returns:
            True if word is difficult, False otherwise
        """
        # Compare using lowercased word
        return word.lower() in self.V_difficult
    
    def get_difficulty_score(self, word: str) -> Optional[float]:
        """
        Get the composite difficulty score for a word.
        
        Args:
            word: Word to look up
            
        Returns:
            Composite score S_w if word exists, None otherwise
        """
        profile = self.word_profiles.get(word.lower())
        return profile.S_w if profile else None

