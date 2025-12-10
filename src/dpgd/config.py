"""Configuration loading and management for DPGD."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from .utils.io_utils import load_yaml


class ConfigLoader:
    """Load and merge multiple YAML configuration files."""
    
    def __init__(self, base_config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the ConfigLoader.
        
        Args:
            base_config_path: Optional base configuration file path
        """
        self.base_config_path = Path(base_config_path) if base_config_path else None
        self._config: Dict[str, Any] = {}
    
    def load(self, *config_paths: Union[str, Path]) -> Dict[str, Any]:
        """
        Load and merge multiple configuration files.
        
        Files are loaded in order, with later files overriding earlier ones.
        
        Args:
            *config_paths: Variable number of configuration file paths
            
        Returns:
            Merged configuration dictionary
        """
        configs = []
        
        # Load base config if provided
        if self.base_config_path and self.base_config_path.exists():
            configs.append(load_yaml(self.base_config_path))
        
        # Load all provided config paths
        for config_path in config_paths:
            path = Path(config_path)
            if path.exists():
                configs.append(load_yaml(path))
            else:
                raise FileNotFoundError(f"Configuration file not found: {path}")
        
        # Merge all configs
        merged_config = {}
        for config in configs:
            merged_config = self._deep_merge(merged_config, config)
        
        self._config = merged_config
        return merged_config
    
    def load_from_dict(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        profile_config: Optional[Dict[str, Any]] = None,
        experiment_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Load configuration from dictionaries.
        
        Args:
            model_config: Model configuration dictionary
            profile_config: Profile configuration dictionary
            experiment_config: Experiment configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        configs = []
        
        if model_config:
            configs.append(model_config)
        if profile_config:
            configs.append(profile_config)
        if experiment_config:
            configs.append(experiment_config)
        
        merged_config = {}
        for config in configs:
            merged_config = self._deep_merge(merged_config, config)
        
        self._config = merged_config
        return merged_config
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            update: Dictionary to merge into base
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key (supports dot notation).
        
        Args:
            key: Configuration key (e.g., "model.name" or "model")
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the current configuration dictionary."""
        return self._config.copy()
    
    def to_pydantic(self, model_class: Type[BaseModel]) -> BaseModel:
        """
        Convert the configuration dictionary to a Pydantic model.
        
        Args:
            model_class: Pydantic model class to instantiate
            
        Returns:
            Pydantic model instance
        """
        return model_class(**self._config)

