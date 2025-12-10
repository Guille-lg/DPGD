"""Text preprocessing utilities."""

import re
import unicodedata
from typing import Optional


def standardize_text(
    text: str,
    normalize_quotes: bool = True,
    normalize_whitespace: bool = True,
    strip_whitespace: bool = True,
) -> str:
    """
    Standardize raw text by normalizing quotes, whitespace, and stripping.
    
    Args:
        text: Input text to standardize
        normalize_quotes: If True, normalize various quote types to standard quotes
        normalize_whitespace: If True, normalize multiple whitespace characters to single space
        strip_whitespace: If True, strip leading and trailing whitespace
        
    Returns:
        Standardized text
    """
    if not text:
        return ""
    
    # Normalize Unicode characters (e.g., decompose composed characters)
    text = unicodedata.normalize("NFKC", text)
    
    # Normalize quotes
    if normalize_quotes:
        # Replace various quote types with standard quotes
        text = text.replace('"', '"').replace('"', '"')  # Smart double quotes
        text = text.replace(''', "'").replace(''', "'")  # Smart single quotes
        text = text.replace('«', '"').replace('»', '"')  # Guillemets
        text = text.replace('„', '"').replace('"', '"')  # German quotes
        text = text.replace('‚', "'").replace(chr(0x2019), "'")  # German single quotes
    
    # Normalize whitespace
    if normalize_whitespace:
        # Replace various whitespace characters with regular space
        text = re.sub(r'[\t\n\r\f\v]+', ' ', text)  # Replace tabs, newlines, etc. with space
        text = re.sub(r'[ \u00A0\u1680\u2000-\u200B\u202F\u205F\u3000]+', ' ', text)  # Various space chars
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
    
    # Strip leading and trailing whitespace
    if strip_whitespace:
        text = text.strip()
    
    return text


def preprocess_dataset(
    input_path: str,
    output_path: str,
    source_field: str = "source",
    reference_field: str = "reference",
) -> None:
    """
    Preprocess a JSONL dataset file by standardizing text fields.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        source_field: Name of the source text field
        reference_field: Name of the reference text field
    """
    import json
    from pathlib import Path
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            try:
                example = json.loads(line)
                
                # Standardize text fields
                if source_field in example:
                    example[source_field] = standardize_text(example[source_field])
                if reference_field in example:
                    example[reference_field] = standardize_text(example[reference_field])
                
                # Write processed example
                f_out.write(json.dumps(example, ensure_ascii=False) + '\n')
                processed_count += 1
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON line: {e}")
                continue
    
    print(f"Preprocessed {processed_count} examples from {input_path} to {output_path}")

