"""Frequency data loader for word frequency scores."""

import csv
import random
from pathlib import Path
from typing import Dict, Optional, Union


class FrequencyLoader:
    """
    Load word frequency data from CSV files.
    
    Expected CSV format:
        word,count
        el,12345
        la,9876
        ...
    """
    
    def __init__(
        self,
        frequency_file: Optional[Union[str, Path]] = None,
        use_mock_if_missing: bool = True,
    ):
        """
        Initialize the frequency loader.
        
        Args:
            frequency_file: Path to CSV file with word frequencies
            use_mock_if_missing: If True, use mock data when file is missing
        """
        self.frequency_file = Path(frequency_file) if frequency_file else None
        self.use_mock_if_missing = use_mock_if_missing
        self._frequencies: Dict[str, int] = {}
        self._max_frequency: int = 0
        
        if self.frequency_file and self.frequency_file.exists():
            self._load_frequencies()
        elif use_mock_if_missing:
            self._generate_mock_frequencies()
    
    def _load_frequencies(self) -> None:
        """Load frequencies from CSV file."""
        self._frequencies = {}
        
        with open(self.frequency_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Expect columns: word, count (case-insensitive)
            for row in reader:
                # Handle different column name variations
                word_key = None
                count_key = None
                
                for key in row.keys():
                    key_lower = key.lower()
                    if key_lower in ['word', 'token', 'term']:
                        word_key = key
                    elif key_lower in ['count', 'frequency', 'freq', 'f']:
                        count_key = key
                
                if word_key and count_key:
                    word = row[word_key].strip().lower()
                    try:
                        count = int(row[count_key])
                        self._frequencies[word] = count
                    except (ValueError, KeyError):
                        continue
        
        if self._frequencies:
            self._max_frequency = max(self._frequencies.values())
        else:
            self._max_frequency = 1
    
    def _generate_mock_frequencies(self) -> None:
        """
        Generate mock frequency data for testing.
        
        Creates random frequencies for common Spanish words and some complex words.
        """
        # Common Spanish words with high frequencies
        common_words = [
            "el", "la", "de", "que", "y", "a", "en", "un", "es", "se",
            "no", "te", "lo", "le", "da", "su", "por", "son", "con", "para",
            "del", "una", "está", "más", "como", "muy", "sin", "sobre", "también",
            "después", "hasta", "donde", "quien", "están", "pero", "cual", "cuando",
            "todo", "esta", "ser", "haber", "hacer", "poder", "decir", "ir", "ver",
            "dar", "saber", "querer", "llegar", "pasar", "deber", "poner", "parecer",
            "quedar", "hablar", "llevar", "dejar", "seguir", "encontrar", "llamar",
        ]
        
        # Complex words with lower frequencies
        complex_words = [
            "inconstitucionalmente", "desinstitucionalización", "reestructuración",
            "conceptualización", "interdependencia", "globalización", "meticulosa",
            "multifacético", "transformaciones", "descentralización", "organizacional",
            "austeridad", "desregulación", "estructurales", "exhaustivo",
        ]
        
        # Generate frequencies: common words get high frequencies, complex words get low
        random.seed(42)  # For reproducibility
        
        for word in common_words:
            # Common words: 1000-100000
            self._frequencies[word] = random.randint(1000, 100000)
        
        for word in complex_words:
            # Complex words: 1-100
            self._frequencies[word] = random.randint(1, 100)
        
        self._max_frequency = max(self._frequencies.values())
        print(f"Generated mock frequency data with {len(self._frequencies)} words.")
    
    def get_frequency(self, word: str) -> int:
        """
        Get the raw frequency count for a word.
        
        Args:
            word: Word to look up (will be lowercased)
            
        Returns:
            Frequency count, or 0 if word not found
        """
        word_lower = word.lower().strip()
        return self._frequencies.get(word_lower, 0)
    
    def get_normalized_frequency(self, word: str) -> float:
        """
        Get normalized frequency score (0-1) for a word.
        
        Args:
            word: Word to look up
            
        Returns:
            Normalized frequency score between 0 and 1
        """
        if self._max_frequency == 0:
            return 0.0
        
        raw_freq = self.get_frequency(word)
        # Normalize: F_norm = 1 - (freq / max_freq)
        # Higher frequency = lower difficulty = lower F_norm
        # Lower frequency = higher difficulty = higher F_norm
        F_norm = 1.0 - (raw_freq / self._max_frequency)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, F_norm))
    
    def has_word(self, word: str) -> bool:
        """
        Check if a word exists in the frequency data.
        
        Args:
            word: Word to check
            
        Returns:
            True if word exists, False otherwise
        """
        return word.lower().strip() in self._frequencies
    
    @property
    def num_words(self) -> int:
        """Get the number of words in the frequency data."""
        return len(self._frequencies)
    
    @property
    def max_frequency(self) -> int:
        """Get the maximum frequency in the dataset."""
        return self._max_frequency

