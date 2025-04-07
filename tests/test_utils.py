

import yaml
from pathlib import Path
import torch
import random
import numpy as np

def load_test_config(config_path=None):
    """Load test configuration from YAML file.

    Args:
        config_path (str, optional): Path to config file. If None, uses default.

    Returns:
        dict: Test configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent / "test_config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def setup_test_environment(config):
    """Set up test environment with configuration settings.

    Args:
        config (dict): Test configuration dictionary
    """
    # Set random seeds for reproducibility
    if 'environment' in config:
        env_config = config['environment']
        if 'seed' in env_config:
            seed = env_config['seed']
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

def get_test_model(config):
    """Create a test model instance based on configuration.

    Args:
        config (dict): Test configuration dictionary

    Returns:
        torch.nn.Module: Test model instance
    """
    from src.models.letter_classifier import LetterClassifierCNN

    model_config = config['model']
    model = LetterClassifierCNN(
        input_channels=model_config['input_channels'],
        num_classes=model_config['num_classes'],
        dropout_rates=model_config['dropout_rates']
    )
    return model

def create_test_image(config):
    """Create a test image based on configuration.

    Args:
        config (dict): Test configuration dictionary

    Returns:
        torch.Tensor: Test image tensor
    """
    test_data_config = config['test_data']
    image_size = test_data_config['image_size']

    # Create a simple test image
    image = torch.zeros((1, *image_size))
    # Add some random noise for testing
    image += torch.randn_like(image) * 0.1
    return image

def get_test_batch(config):
    """Create a batch of test images based on configuration.

    Args:
        config (dict): Test configuration dictionary

    Returns:
        torch.Tensor: Batch of test images
    """
    batch_size = config['inference']['batch_size']
    images = torch.stack([create_test_image(config) for _ in range(batch_size)])
    return images