"""
Utility functions for MakeSenseOfIt.
Common helper functions used throughout the application.
"""
import os
import json
import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(path)
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {dir_path}")
        return dir_path
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise

def generate_timestamp(fmt: str = '%Y%m%d_%H%M%S') -> str:
    """
    Generate a timestamp string.
    
    Args:
        fmt: Timestamp format (default: YYYYMMDD_HHMMSS)
        
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime(fmt)

def save_json(
    data: Dict[str, Any],
    filepath: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False
) -> Path:
    """
    Save data to JSON file with error handling.
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
        ensure_ascii: Whether to escape non-ASCII characters
        
    Returns:
        Path to saved file
    """
    filepath = Path(filepath)
    
    try:
        # Ensure directory exists
        ensure_directory(filepath.parent)
        
        # Save JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)
        
        logger.debug(f"Saved JSON to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")
        raise

def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load data from JSON file with error handling.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    filepath = Path(filepath)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.debug(f"Loaded JSON from: {filepath}")
        return data
        
    except FileNotFoundError:
        logger.error(f"JSON file not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load JSON from {filepath}: {e}")
        raise

def format_number(num: Union[int, float], decimals: int = 0) -> str:
    """
    Format number with thousands separator.
    
    Args:
        num: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted number string
    """
    if decimals > 0:
        return f"{num:,.{decimals}f}"
    return f"{int(num):,}"

def truncate_text(
    text: str,
    max_length: int = 100,
    suffix: str = "..."
) -> str:
    """
    Truncate text to maximum length with suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Original filename
        replacement: Character to replace invalid chars with
        
    Returns:
        Sanitized filename
    """
    # Invalid filename characters
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    
    # Replace invalid characters
    sanitized = re.sub(invalid_chars, replacement, filename)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Limit length
    max_length = 255
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized or "unnamed"

def calculate_date_range(days_back: int) -> tuple[datetime, datetime]:
    """
    Calculate date range from days back to now.
    
    Args:
        days_back: Number of days to go back
        
    Returns:
        Tuple of (start_date, end_date)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    return start_date, end_date

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def get_file_size(filepath: Union[str, Path]) -> str:
    """
    Get human-readable file size.
    
    Args:
        filepath: Path to file
        
    Returns:
        Formatted file size string
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return "0 B"
    
    size = filepath.stat().st_size
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    
    return f"{size:.1f} TB"

def create_hash(text: str) -> str:
    """
    Create SHA256 hash of text.
    
    Args:
        text: Text to hash
        
    Returns:
        Hex digest of hash
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result

class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "Operation"):
        """Initialize timer."""
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timer."""
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and log duration."""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logger.debug(f"{self.name} took {format_duration(duration)}")
        return False
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()