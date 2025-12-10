"""Morphological complexity analyzer for Spanish."""

import re
from typing import Optional, Set


class MorphologyAnalyzer:
    """
    Analyzer for morphological complexity in Spanish.
    
    Uses heuristics to identify complex morphological patterns:
    - Long words (> 12 characters)
    - Specific complex suffixes (e.g., "-mente", "-ción", "-ción")
    - Compound words
    - Derivational affixes
    """
    
    def __init__(
        self,
        long_word_threshold: int = 12,
        complex_suffixes: Optional[Set[str]] = None,
        complex_prefixes: Optional[Set[str]] = None,
    ):
        """
        Initialize the morphology analyzer.
        
        Args:
            long_word_threshold: Minimum length for a word to be considered long
            complex_suffixes: Set of complex suffixes (default: Spanish adverbial and nominal suffixes)
            complex_prefixes: Set of complex prefixes (default: Spanish derivational prefixes)
        """
        self.long_word_threshold = long_word_threshold
        
        # Default complex suffixes for Spanish
        if complex_suffixes is None:
            self.complex_suffixes = {
                # Adverbial suffixes
                "mente",  # e.g., "inconstitucionalmente"
                # Nominal suffixes
                "ción", "sión",  # e.g., "conceptualización", "decisión"
                "idad", "edad",  # e.g., "complejidad", "velocidad"
                "ismo", "ista",  # e.g., "capitalismo", "optimista"
                "anza", "encia",  # e.g., "esperanza", "diferencia"
                "ura", "tura",  # e.g., "estructura", "naturaleza"
                # Adjectival suffixes
                "oso", "osa",  # e.g., "peligroso", "hermosa"
                "able", "ible",  # e.g., "posible", "probable"
                # Verbal suffixes
                "izar", "ificar",  # e.g., "organizar", "simplificar"
            }
        else:
            self.complex_suffixes = complex_suffixes
        
        # Default complex prefixes for Spanish
        if complex_prefixes is None:
            self.complex_prefixes = {
                "des", "dis",  # e.g., "desinstitucionalización"
                "re",  # e.g., "reestructuración"
                "in", "im", "ir",  # e.g., "inconstitucionalmente"
                "pre", "pro",  # e.g., "predecir", "proponer"
                "anti", "contra",  # e.g., "anticonstitucional"
                "inter", "intra",  # e.g., "interdependencia"
                "multi", "semi",  # e.g., "multifacético", "semicírculo"
                "super", "sub",  # e.g., "superestructura", "subestimar"
            }
        else:
            self.complex_prefixes = complex_prefixes
    
    def analyze(self, word: str) -> float:
        """
        Analyze morphological complexity of a word.
        
        Returns a score between 0 (simple) and 1 (complex).
        
        Args:
            word: Word to analyze (will be lowercased)
            
        Returns:
            Morphological complexity score M_w (0-1)
        """
        word_lower = word.lower().strip()
        
        if not word_lower:
            return 0.0
        
        complexity_factors = []
        
        # Factor 1: Length
        if len(word_lower) > self.long_word_threshold:
            # Long words are more complex
            length_score = min(1.0, (len(word_lower) - self.long_word_threshold) / 10.0)
            complexity_factors.append(length_score)
        
        # Factor 2: Complex suffixes
        for suffix in self.complex_suffixes:
            if word_lower.endswith(suffix):
                # Suffix presence indicates complexity
                complexity_factors.append(0.7)
                break
        
        # Factor 3: Complex prefixes
        for prefix in self.complex_prefixes:
            if word_lower.startswith(prefix):
                # Prefix presence indicates complexity
                complexity_factors.append(0.5)
                break
        
        # Factor 4: Multiple morphemes (hyphens or multiple affixes)
        if '-' in word_lower:
            complexity_factors.append(0.6)
        
        # Factor 5: Repeated complex patterns
        suffix_count = sum(1 for suffix in self.complex_suffixes if word_lower.endswith(suffix))
        prefix_count = sum(1 for prefix in self.complex_prefixes if word_lower.startswith(prefix))
        if suffix_count + prefix_count > 1:
            complexity_factors.append(0.8)
        
        # Factor 6: Very long words with multiple affixes
        if len(word_lower) > 15 and (suffix_count > 0 or prefix_count > 0):
            complexity_factors.append(0.9)
        
        # Compute final score: take maximum or average of factors
        if complexity_factors:
            # Use maximum to capture highest complexity indicator
            M_w = max(complexity_factors)
        else:
            # Simple word
            M_w = 0.0
        
        # Normalize to [0, 1]
        return min(1.0, max(0.0, M_w))
    
    def is_complex(self, word: str, threshold: float = 0.5) -> bool:
        """
        Check if a word is morphologically complex.
        
        Args:
            word: Word to check
            threshold: Complexity threshold (default: 0.5)
            
        Returns:
            True if word is complex, False otherwise
        """
        return self.analyze(word) >= threshold
    
    def get_complexity_category(self, word: str) -> str:
        """
        Get a human-readable complexity category.
        
        Args:
            word: Word to categorize
            
        Returns:
            Category string: "simple", "moderate", or "complex"
        """
        score = self.analyze(word)
        
        if score < 0.3:
            return "simple"
        elif score < 0.7:
            return "moderate"
        else:
            return "complex"

