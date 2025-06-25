"""
Tests for configuration management.
"""
import pytest
import os
import json
import tempfile
from pathlib import Path
from src.config import Config

class TestConfig:
    """Test configuration functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        
        assert config.default_time_window == 30
        assert 'twitter' in config.default_platforms
        assert 'reddit' in config.default_platforms
        assert config.sentiment_model == 'cardiffnlp/twitter-roberta-base-sentiment-latest'
        assert config.output_directory == './output'
        assert config.cache_directory == './cache'
        assert config.batch_size == 100
        assert config.max_retries == 3
    
    def test_config_directories_created(self, temp_dir):
        """Test that directories are created."""
        config = Config()
        config.output_directory = str(temp_dir / 'output')
        config.cache_directory = str(temp_dir / 'cache')
        
        # Manually create directories (since __init__ was already called)
        config._create_directories()
        
        assert Path(config.output_directory).exists()
        assert Path(config.cache_directory).exists()
    
    def test_config_from_file(self, sample_config_file):
        """Test loading configuration from file."""
        config = Config(config_file=str(sample_config_file))
        
        assert config.default_time_window == 14  # From file
        assert config.rate_limits['twitter'] == 100
        assert config.rate_limits['reddit'] == 120
        assert config.visualization_style == 'light'
        assert config.batch_size == 50
    
    def test_config_invalid_file(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            Config(config_file='nonexistent.json')
    
    def test_config_invalid_json(self, temp_dir):
        """Test loading from invalid JSON file."""
        bad_json = temp_dir / 'bad.json'
        with open(bad_json, 'w') as f:
            f.write('{ invalid json')
        
        with pytest.raises(json.JSONDecodeError):
            Config(config_file=str(bad_json))
    
    def test_get_rate_limit(self):
        """Test rate limit retrieval."""
        config = Config()
        
        assert config.get_rate_limit('twitter') == 50
        assert config.get_rate_limit('reddit') == 60
        assert config.get_rate_limit('unknown') == 60  # Default
        assert config.get_rate_limit('TWITTER') == 50  # Case insensitive
    
    def test_get_platform_config(self):
        """Test platform configuration retrieval."""
        config = Config()
        platform_config = config.get_platform_config('twitter')
        
        assert platform_config['rate_limit'] == 50
        assert platform_config['user_agent'] == 'MakeSenseOfIt/1.0'
        assert platform_config['timeout'] == 30
        assert platform_config['max_retries'] == 3
        assert platform_config['retry_delay'] == 1.0
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = Config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'default_platforms' in config_dict
        assert 'rate_limits' in config_dict
        assert config_dict['default_time_window'] == 30
        assert config_dict['sentiment_model'] == config.sentiment_model
    
    def test_config_save(self, temp_dir):
        """Test saving configuration."""
        config = Config()
        save_path = temp_dir / 'saved_config.json'
        
        # Modify a value
        config.batch_size = 200
        
        # Save
        config.save(str(save_path))
        
        assert save_path.exists()
        
        # Load and verify
        with open(save_path) as f:
            saved_data = json.load(f)
        
        assert saved_data['batch_size'] == 200
        assert saved_data['default_time_window'] == 30
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = Config()
        
        # Invalid rate limits
        config.rate_limits = "not a dict"
        with pytest.raises(ValueError, match="rate_limits must be a dictionary"):
            config._validate()
        
        # Invalid batch size
        config = Config()
        config.batch_size = -1
        with pytest.raises(ValueError, match="Invalid batch size"):
            config._validate()
        
        # Invalid directory
        config = Config()
        config.output_directory = ""
        with pytest.raises(ValueError, match="Invalid output_directory"):
            config._validate()
    
    def test_config_str_representation(self):
        """Test string representation of config."""
        config = Config()
        str_repr = str(config)
        
        assert isinstance(str_repr, str)
        assert 'default_platforms' in str_repr
        assert 'twitter' in str_repr
        
        # Should be valid JSON
        parsed = json.loads(str_repr)
        assert parsed['default_time_window'] == 30