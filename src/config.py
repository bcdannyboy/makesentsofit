"""
Configuration management for MakeSenseOfIt.
Handles loading, validation, and access to configuration values.
"""
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class Config:
    """
    Application configuration with defaults and file loading support.
    """
    # Default configuration values
    default_platforms: List[str] = field(default_factory=lambda: ['twitter', 'reddit'])
    default_time_window: int = 30
    rate_limits: Dict[str, int] = field(default_factory=lambda: {
        'twitter': 50,
        'reddit': 60
    })
    sentiment_model: str = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    output_directory: str = './output'
    cache_directory: str = './cache'
    visualization_style: str = 'dark'
    batch_size: int = 100
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Additional settings
    user_agent: str = 'MakeSenseOfIt/1.0'
    timeout: int = 30
    max_posts_per_query: Optional[int] = None

    # Analysis defaults loaded from config
    queries: List[str] = field(default_factory=list)
    output_formats: List[str] = field(default_factory=lambda: ['json'])
    output_prefix: Optional[str] = None
    visualize: bool = False
    verbose: bool = False
    limit: Optional[int] = None
    
    # Platform-specific configurations
    reddit: Optional[Dict[str, Any]] = None
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration from defaults and optional file.
        
        Args:
            config_file: Path to JSON configuration file
        """
        # Set defaults first
        self._set_defaults()
        
        # Load from file if provided
        if config_file:
            self._load_from_file(config_file)
        
        # Validate configuration
        self._validate()
        
        # Ensure directories exist
        self._create_directories()
        
        logger.debug(f"Configuration initialized: {self._summary()}")
    
    def _set_defaults(self):
        """Set default configuration values."""
        self.default_platforms = ['twitter', 'reddit']
        self.default_time_window = 30
        self.rate_limits = {'twitter': 50, 'reddit': 60}
        self.sentiment_model = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
        self.output_directory = './output'
        self.cache_directory = './cache'
        self.visualization_style = 'dark'
        self.batch_size = 100
        self.max_retries = 3
        self.retry_delay = 1.0
        self.user_agent = 'MakeSenseOfIt/1.0'
        self.timeout = 30
        self.max_posts_per_query = None
        self.reddit = None
        self.queries = []
        self.output_formats = ['json']
        self.output_prefix = None
        self.visualize = False
        self.verbose = False
        self.limit = None
    
    def _load_from_file(self, config_file: str):
        """
        Load configuration from JSON file.
        
        Args:
            config_file: Path to configuration file
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                
            # Update configuration with loaded values
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    logger.debug(f"Loaded config: {key} = {value}")
                else:
                    logger.warning(f"Unknown configuration key: {key}")
                    
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _validate(self):
        """Validate configuration values."""
        # Validate rate limits
        if not isinstance(self.rate_limits, dict):
            raise ValueError("rate_limits must be a dictionary")
        
        for platform, limit in self.rate_limits.items():
            if not isinstance(limit, int) or limit <= 0:
                raise ValueError(f"Invalid rate limit for {platform}: {limit}")
        
        # Validate batch size
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(f"Invalid batch size: {self.batch_size}")
        
        # Validate directories
        for dir_attr in ['output_directory', 'cache_directory']:
            dir_path = getattr(self, dir_attr)
            if not isinstance(dir_path, str) or not dir_path:
                raise ValueError(f"Invalid {dir_attr}: {dir_path}")
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for dir_attr in ['output_directory', 'cache_directory']:
            dir_path = Path(getattr(self, dir_attr))
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to create directory {dir_path}: {e}")
                raise
    
    def _summary(self) -> str:
        """Get configuration summary for logging."""
        return f"platforms={self.default_platforms}, output={self.output_directory}"
    
    def get_rate_limit(self, platform: str) -> int:
        """
        Get rate limit for a specific platform.
        
        Args:
            platform: Platform name
            
        Returns:
            Rate limit (requests per minute)
        """
        return self.rate_limits.get(platform.lower(), 60)
    
    def get_platform_config(self, platform: str) -> Dict[str, Any]:
        """
        Get all configuration for a specific platform.
        
        Args:
            platform: Platform name
            
        Returns:
            Platform-specific configuration
        """
        return {
            'rate_limit': self.get_rate_limit(platform),
            'user_agent': self.user_agent,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save(self, filepath: str):
        """
        Save current configuration to file.
        
        Args:
            filepath: Path to save configuration
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Configuration saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return json.dumps(self.to_dict(), indent=2)