"""IO utility functions for file operations."""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, Union


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Dictionary containing the YAML contents
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If the YAML is malformed
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """
    Save a dictionary to a YAML file.
    
    Args:
        data: Dictionary to save
        file_path: Path where to save the YAML file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a JSON file and return its contents as a dictionary.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the JSON contents
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the JSON is malformed
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Save a dictionary to a JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path where to save the JSON file
        indent: Number of spaces for indentation (default: 2)
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

