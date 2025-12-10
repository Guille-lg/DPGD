"""Profile builder factory for creating Hybrid Difficulty Profiles."""

from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from .frequency_loader import FrequencyLoader
from .morphology_analyzer import MorphologyAnalyzer
from .profile_schema import HybridDifficultyProfile, WordProfile


class ProfileBuilder:
    """
    Factory class for building Hybrid Difficulty Profiles.
    
    Takes configuration (betas, thresholds) and vocabulary, then constructs
    a profile with difficulty scores for each word.
    """
    
    def __init__(
        self,
        beta_Freq: float = 0.4,
        beta_Morph: float = 0.4,
        beta_Mask: float = 0.2,
        threshold: float = 0.5,
        frequency_file: Optional[Union[str, Path]] = None,
        long_word_threshold: int = 12,
    ):
        """
        Initialize the profile builder.
        
        Args:
            beta_Freq: Weight for frequency component (default: 0.4)
            beta_Morph: Weight for morphological complexity component (default: 0.4)
            beta_Mask: Weight for known/unknown component (default: 0.2)
            threshold: Threshold for determining difficult words (default: 0.5)
            frequency_file: Path to frequency CSV file (optional)
            long_word_threshold: Threshold for long word detection (default: 12)
        """
        self.beta_Freq = beta_Freq
        self.beta_Morph = beta_Morph
        self.beta_Mask = beta_Mask
        self.threshold = threshold
        
        # Initialize components
        self.frequency_loader = FrequencyLoader(
            frequency_file=frequency_file,
            use_mock_if_missing=False,
        )
        self.morphology_analyzer = MorphologyAnalyzer(
            long_word_threshold=long_word_threshold,
        )
    
    def build_profile(
        self,
        vocabulary: Union[List[str], Set[str]],
        known_words: Optional[Set[str]] = None,
    ) -> HybridDifficultyProfile:
        """
        Build a Hybrid Difficulty Profile for a given vocabulary.
        
        Args:
            vocabulary: Set or list of words to profile
            known_words: Optional set of known words (for K_w calculation).
                        If None, words with frequency > 0 are considered known.
        
        Returns:
            HybridDifficultyProfile with computed scores and V_difficult set
        """
        # Convert to set and normalize
        vocab_set = {word.lower().strip() for word in vocabulary if word.strip()}
        
        # Create profile
        profile = HybridDifficultyProfile(
            beta_Freq=self.beta_Freq,
            beta_Morph=self.beta_Morph,
            beta_Mask=self.beta_Mask,
            threshold=self.threshold,
        )
        
        # Build word profiles
        for word in vocab_set:
            word_profile = self._create_word_profile(word, known_words)
            profile.add_word_profile(word_profile)
        
        # Compute all composite scores
        profile.compute_all_scores()
        
        # Update difficult words set
        profile.update_difficult_words()
        
        # Log statistics
        if len(vocab_set) > 0:
            avg_score = sum(p.S_w for p in profile.word_profiles.values()) / len(profile.word_profiles)
            max_score = max(p.S_w for p in profile.word_profiles.values()) if profile.word_profiles else 0
            print(f"      Profile statistics: avg_S_w={avg_score:.3f}, max_S_w={max_score:.3f}")
        
        return profile
    
    def _create_word_profile(
        self,
        word: str,
        known_words: Optional[Set[str]] = None,
    ) -> WordProfile:
        """
        Create a WordProfile for a single word.
        
        Args:
            word: Word to profile
            known_words: Optional set of known words
            
        Returns:
            WordProfile with F_norm, M_w, K_w computed
        """
        # F_norm: Normalized frequency score (0-1)
        F_norm = self.frequency_loader.get_normalized_frequency(word)
        
        # M_w: Morphological complexity score (0-1)
        M_w = self.morphology_analyzer.analyze(word)
        
        # K_w: Unknown flag (1 = unknown/should be masked, 0 = known)
        # Per paper: K_w = 1 means word is explicitly unknown and should be hard-masked
        if known_words is not None:
            # Use provided known words set: K_w = 1 if NOT in known words (unknown)
            K_w = 0 if word.lower() in {w.lower() for w in known_words} else 1
        else:
            # Consider word unknown if it has NO frequency data (not in corpus)
            K_w = 0 if self.frequency_loader.has_word(word) else 1
        
        # Create profile (S_w will be computed later)
        return WordProfile(
            word=word,
            F_norm=F_norm,
            M_w=M_w,
            K_w=K_w,
            S_w=0.0,  # Will be computed by compute_all_scores()
        )
    
    def build_profile_from_text(
        self,
        text: str,
        known_words: Optional[Set[str]] = None,
    ) -> HybridDifficultyProfile:
        """
        Build a profile from a text by extracting unique words.
        
        Args:
            text: Input text (will be split into words)
            known_words: Optional set of known words
            
        Returns:
            HybridDifficultyProfile
        """
        # Extract vocabulary from text
        words = text.lower().split()
        # Remove punctuation and get unique words
        import string
        vocab = {
            word.strip(string.punctuation)
            for word in words
            if word.strip(string.punctuation)
        }
        
        return self.build_profile(vocab, known_words=known_words)
    
    def update_profile(
        self,
        profile: HybridDifficultyProfile,
        new_vocabulary: Union[List[str], Set[str]],
        known_words: Optional[Set[str]] = None,
    ) -> HybridDifficultyProfile:
        """
        Update an existing profile with new vocabulary.
        
        Args:
            profile: Existing profile to update
            new_vocabulary: New words to add to the profile
            known_words: Optional set of known words
            
        Returns:
            Updated profile
        """
        vocab_set = {word.lower().strip() for word in new_vocabulary if word.strip()}
        
        # Add new word profiles
        for word in vocab_set:
            if word not in profile.word_profiles:
                word_profile = self._create_word_profile(word, known_words)
                profile.add_word_profile(word_profile)
        
        # Recompute all scores
        profile.compute_all_scores()
        profile.update_difficult_words()
        
        return profile

