"""
Configuration utility module for loading and managing YAML configuration files.

This module provides functions to load, parse, and access configuration files used
throughout the project, following a standardized format for model, training, data, 
and evaluation settings.
"""

import os
import yaml
import logging
from typing import Dict, Any
import json
import copy

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        Dict[str, Any]: Dictionary containing the configuration.
        
    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If the YAML file is malformed.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise

def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract model configuration from the full configuration.
    
    Args:
        config (Dict[str, Any]): Full configuration dictionary.
        
    Returns:
        Dict[str, Any]: Model configuration section.
    """
    if 'model' not in config:
        logger.warning("Model configuration section not found in config")
        return {}
    
    return config['model']

def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract training configuration from the full configuration.
    
    Args:
        config (Dict[str, Any]): Full configuration dictionary.
        
    Returns:
        Dict[str, Any]: Training configuration section.
    """
    if 'training' not in config:
        logger.warning("Training configuration section not found in config")
        return {}
    
    return config['training']

def get_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract data configuration from the full configuration.
    
    Args:
        config (Dict[str, Any]): Full configuration dictionary.
        
    Returns:
        Dict[str, Any]: Data configuration section.
    """
    if 'data' not in config:
        logger.warning("Data configuration section not found in config")
        return {}
    
    return config['data']

def get_inference_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract inference configuration from the full configuration.
    
    Args:
        config (Dict[str, Any]): Full configuration dictionary.
        
    Returns:
        Dict[str, Any]: Inference configuration section.
    """
    if 'inference' not in config:
        logger.warning("Inference configuration section not found in config")
        return {}
    
    return config['inference']

def get_evaluation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract evaluation configuration from the full configuration.
    
    Args:
        config (Dict[str, Any]): Full configuration dictionary.
        
    Returns:
        Dict[str, Any]: Evaluation configuration section.
    """
    if 'evaluation' not in config:
        logger.warning("Evaluation configuration section not found in config")
        return {}
    
    return config['evaluation']

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary to save.
        config_path (str): Path where the configuration will be saved.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved configuration to {config_path}")
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {e}")
        raise

def update_config(base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a configuration dictionary with override values.
    
    Args:
        base_config (Dict[str, Any]): Base configuration dictionary.
        overrides (Dict[str, Any]): Dictionary with override values.
        
    Returns:
        Dict[str, Any]: Updated configuration dictionary.
    """
    # Create a deep copy to avoid modifying the original
    config = copy.deepcopy(base_config)
    
    # Apply overrides
    for section, values in overrides.items():
        if isinstance(values, dict) and section in config and isinstance(config[section], dict):
            # Recursively update nested dictionaries
            config[section].update(values)
        else:
            # Replace or add the value directly
            config[section] = values
    
    return config

def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a value from a nested configuration dictionary using a key path.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary.
        key_path (str): Path to the key, using dots for nested dictionaries (e.g., 'model.architecture').
        default (Any, optional): Default value to return if the key is not found. Defaults to None.
        
    Returns:
        Any: The value at the specified key path, or the default value if not found.
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default

def config_to_json(config: Dict[str, Any]) -> str:
    """
    Convert a configuration dictionary to a JSON string.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary.
        
    Returns:
        str: JSON string representation of the configuration.
    """
    return json.dumps(config, indent=2)

def config_to_args(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a configuration dictionary to a flat dictionary of command-line arguments.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary.
        
    Returns:
        Dict[str, Any]: Flattened dictionary with keys like 'model.architecture'.
    """
    args = {}
    
    def _flatten(d, prefix=''):
        for key, value in d.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                _flatten(value, new_key)
            else:
                args[new_key] = value
    
    _flatten(config)
    return args

def load_or_create_config(config_path: str = "configs/model_config.yaml", default_config: Dict[str, Any] = {}) -> Dict[str, Any]:
    """
    Load a configuration file if it exists, or create a new one with default values.
    
    Args:
        config_path (str): Path to the configuration file.
        default_config (Dict[str, Any]): Default configuration to use if the file doesn't exist.
        
    Returns:
        Dict[str, Any]: Loaded or default configuration.
    """
    if os.path.exists(config_path):
        return load_config(config_path)
    else:
        logger.warning(f"Configuration file not found at {config_path}, creating with defaults")
        save_config(default_config, config_path)
        return default_config

if __name__ == "__main__":
    # Simple test/demo code
    logging.basicConfig(level=logging.INFO)
    
    # Default configuration for testing
    default_config = {
        "model": {
            "name": "OCRDeobfuscatorCNN",
            "architecture": "unet",
            "input_shape": [64, 64, 1],
            "filters": [64, 128, 256, 512]
        },
        "training": {
            "batch_size": 64,
            "epochs": 100,
            "learning_rate": 0.001
        }
    }
    
    # Test config loading or creation
    test_path = "test_config.yaml"
    config = load_or_create_config(test_path, default_config)
    
    # Test config getters
    model_config = get_model_config(config)
    logger.info(f"Model architecture: {model_config.get('architecture')}")
    
    # Test config value getter
    lr = get_config_value(config, "training.learning_rate", default=0.01)
    logger.info(f"Learning rate: {lr}")
    
    # Clean up test file
    if os.path.exists(test_path):
        os.remove(test_path)
        logger.info(f"Removed test config file: {test_path}") 