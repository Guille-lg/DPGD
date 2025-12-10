"""ALEXSIS dataset loader."""

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union


class ALEXSISDataset:
    """
    Dataset class for loading the ALEXSIS dataset.
    
    The dataset is expected to be in JSONL format with each line containing:
    {
        "source": "original complex text",
        "reference": "simplified reference text"
    }
    """
    
    def __init__(self, data_path: Union[str, Path], generate_dummy_if_missing: bool = True):
        """
        Initialize the ALEXSIS dataset.
        
        Args:
            data_path: Path to the JSONL dataset file
            generate_dummy_if_missing: If True, generate dummy data when file is missing
        """
        self.data_path = Path(data_path)
        self.generate_dummy_if_missing = generate_dummy_if_missing
        self._examples: List[Dict[str, str]] = []
        
        if not self.data_path.exists():
            if generate_dummy_if_missing:
                self._generate_dummy_data()
            else:
                raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        else:
            self._load_data()
    
    def _generate_dummy_data(self) -> None:
        """Generate dummy ALEXSIS data for testing purposes."""
        dummy_examples = [
            {
                "source": "El tribunal declaró inconstitucionalmente la ley.",
                "reference": "El tribunal declaró que la ley no era válida.",
            },
            {
                "source": "La implementación de políticas públicas requiere una meticulosa planificación.",
                "reference": "Hacer políticas públicas necesita planear con cuidado.",
            },
            {
                "source": "La desinstitucionalización del sistema educativo generó controversias.",
                "reference": "Quitar las instituciones del sistema educativo causó discusiones.",
            },
            {
                "source": "El fenómeno de la globalización económica es multifacético.",
                "reference": "El hecho de que el mundo esté conectado tiene muchas partes.",
            },
            {
                "source": "La interdependencia entre los mercados financieros es innegable.",
                "reference": "Los mercados de dinero dependen unos de otros sin duda.",
            },
            {
                "source": "La conceptualización teórica del problema requiere un análisis exhaustivo.",
                "reference": "Pensar sobre el problema necesita estudiarlo mucho.",
            },
            {
                "source": "La desregulación del sector energético provocó transformaciones estructurales.",
                "reference": "Quitar las reglas del sector de energía cambió cómo funciona.",
            },
            {
                "source": "La implementación de medidas de austeridad generó malestar social.",
                "reference": "Aplicar medidas de ahorro causó que la gente estuviera molesta.",
            },
            {
                "source": "El proceso de descentralización administrativa es complejo.",
                "reference": "El proceso de dar poder a lugares locales es difícil.",
            },
            {
                "source": "La reestructuración organizacional implica cambios significativos.",
                "reference": "Cambiar cómo está organizada una empresa trae cambios grandes.",
            },
        ]
        
        # Create parent directory if it doesn't exist
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write dummy data to file
        with open(self.data_path, 'w', encoding='utf-8') as f:
            for example in dummy_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        self._examples = dummy_examples
        print(f"Generated dummy ALEXSIS dataset at {self.data_path} with {len(dummy_examples)} examples.")
    
    def _load_data(self) -> None:
        """Load data from the JSONL file."""
        self._examples = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    example = json.loads(line)
                    if "source" not in example or "reference" not in example:
                        raise ValueError(
                            f"Example at line {line_num} missing 'source' or 'reference' field"
                        )
                    self._examples.append({
                        "source": example["source"],
                        "reference": example["reference"],
                    })
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num}: {e}")
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self._examples)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        """
        Get a single example by index.
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary with 'source' and 'reference' keys
        """
        if idx < 0 or idx >= len(self._examples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self._examples)}")
        return self._examples[idx]
    
    def __iter__(self) -> Iterator[Dict[str, str]]:
        """Iterate over all examples in the dataset."""
        for example in self._examples:
            yield example
    
    def get_sources(self) -> List[str]:
        """Get all source texts."""
        return [ex["source"] for ex in self._examples]
    
    def get_references(self) -> List[str]:
        """Get all reference texts."""
        return [ex["reference"] for ex in self._examples]

