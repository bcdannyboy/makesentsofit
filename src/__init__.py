"""
MakeSenseOfIt source package.
Social Media Sentiment Analysis tool.
"""

__version__ = '1.0.0'
__author__ = 'MakeSenseOfIt Team'
__email__ = 'contact@makesentsofit.com'

# Package metadata
__all__ = ['Config', 'setup_logging', 'get_logger']

# Import main components for easier access
from .config import Config
from .logger import setup_logging, get_logger