"""
Pytest configuration and shared fixtures.
"""
# Set environment variables before any imports to prevent TensorFlow crashes
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
os.environ['TRANSFORMERS_OFFLINE'] = '1'   # Prevent downloads
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Prevent tokenizer warnings

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator
import json

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def sample_config_file(temp_dir: Path) -> Path:
    """Create a sample configuration file."""
    config_data = {
        "default_platforms": ["twitter"],
        "default_time_window": 14,
        "rate_limits": {
            "twitter": 100,
            "reddit": 120
        },
        "batch_size": 50,
        "visualization_style": "light"
    }
    
    config_path = temp_dir / "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    
    return config_path

@pytest.fixture
def mock_context():
    """Create a mock analysis context."""
    return {
        'queries': ['test', 'sample'],
        'time_window_days': 7,
        'platforms': ['twitter', 'reddit'],
        'output_prefix': 'test_analysis',
        'output_directory': './output',
        'formats': ['json', 'csv'],
        'visualize': False,
        'version': '1.0.0'
    }

@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests."""
    import logging
    logging.basicConfig(level=logging.DEBUG)