"""
Tests for utility functions.
"""
import pytest
import json
from datetime import datetime, timedelta
from pathlib import Path
from src.utils import (
    ensure_directory, generate_timestamp, save_json, load_json,
    format_number, truncate_text, sanitize_filename,
    calculate_date_range, format_duration, get_file_size,
    create_hash, chunk_list, merge_dicts, Timer
)

class TestUtils:
    """Test utility functions."""
    
    def test_ensure_directory(self, temp_dir):
        """Test directory creation."""
        new_dir = temp_dir / 'new' / 'nested' / 'dir'
        result = ensure_directory(new_dir)
        
        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir
        
        # Should not fail if directory exists
        result2 = ensure_directory(new_dir)
        assert result2 == new_dir
    
    def test_generate_timestamp(self):
        """Test timestamp generation."""
        # Default format
        ts1 = generate_timestamp()
        assert len(ts1) == 15  # YYYYMMDD_HHMMSS
        assert '_' in ts1
        
        # Custom format
        ts2 = generate_timestamp('%Y-%m-%d')
        assert len(ts2) == 10
        assert '-' in ts2
    
    def test_save_and_load_json(self, temp_dir):
        """Test JSON save and load."""
        data = {
            'string': 'test',
            'number': 42,
            'list': [1, 2, 3],
            'nested': {'key': 'value'}
        }
        
        filepath = temp_dir / 'test.json'
        
        # Save
        saved_path = save_json(data, filepath)
        assert saved_path == filepath
        assert filepath.exists()
        
        # Load
        loaded = load_json(filepath)
        assert loaded == data
    
    def test_save_json_with_datetime(self, temp_dir):
        """Test saving JSON with datetime objects."""
        data = {
            'timestamp': datetime.now(),
            'date': datetime.now().date()
        }
        
        filepath = temp_dir / 'datetime_test.json'
        save_json(data, filepath)
        
        # Should convert to string
        with open(filepath) as f:
            content = json.load(f)
        
        assert isinstance(content['timestamp'], str)
        assert isinstance(content['date'], str)
    
    def test_load_json_errors(self, temp_dir):
        """Test JSON loading error cases."""
        # Non-existent file
        with pytest.raises(FileNotFoundError):
            load_json('nonexistent.json')
        
        # Invalid JSON
        bad_json = temp_dir / 'bad.json'
        with open(bad_json, 'w') as f:
            f.write('{ invalid json')
        
        with pytest.raises(json.JSONDecodeError):
            load_json(bad_json)
    
    def test_format_number(self):
        """Test number formatting."""
        assert format_number(1234) == '1,234'
        assert format_number(1234567) == '1,234,567'
        assert format_number(1234.5678, decimals=2) == '1,234.57'
        assert format_number(0) == '0'
        assert format_number(-1234) == '-1,234'
    
    def test_truncate_text(self):
        """Test text truncation."""
        # Short text not truncated
        assert truncate_text('short', 10) == 'short'
        
        # Long text truncated
        long_text = 'a' * 20
        assert truncate_text(long_text, 10) == 'aaaaaaa...'
        
        # Custom suffix
        assert truncate_text(long_text, 10, suffix='…') == 'aaaaaaaaa…'
        
        # Edge case
        assert truncate_text('exact', 5) == 'exact'
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        # Remove invalid characters
        assert sanitize_filename('file<>name.txt') == 'file__name.txt'
        assert sanitize_filename('path/to/file') == 'path_to_file'
        
        # Remove leading/trailing dots and spaces
        assert sanitize_filename(' .filename. ') == 'filename'
        
        # Empty filename
        assert sanitize_filename('') == 'unnamed'
        assert sanitize_filename('...') == 'unnamed'
        
        # Long filename
        long_name = 'a' * 300
        result = sanitize_filename(long_name)
        assert len(result) <= 255
    
    def test_calculate_date_range(self):
        """Test date range calculation."""
        start, end = calculate_date_range(7)
        
        # Check types
        assert isinstance(start, datetime)
        assert isinstance(end, datetime)
        
        # Check range
        difference = end - start
        assert difference.days == 7
        
        # End should be close to now
        assert (datetime.now() - end).total_seconds() < 1
    
    def test_format_duration(self):
        """Test duration formatting."""
        assert format_duration(0.5) == '0.5s'
        assert format_duration(59.9) == '59.9s'
        assert format_duration(60) == '1m 0s'
        assert format_duration(125) == '2m 5s'
        assert format_duration(3665) == '1h 1m'
        assert format_duration(7200) == '2h 0m'
    
    def test_get_file_size(self, temp_dir):
        """Test file size formatting."""
        # Non-existent file
        assert get_file_size('nonexistent.txt') == '0 B'
        
        # Create files of different sizes
        small_file = temp_dir / 'small.txt'
        small_file.write_text('a' * 100)
        assert 'B' in get_file_size(small_file)
        
        kb_file = temp_dir / 'kb.txt'
        kb_file.write_text('a' * 2000)
        size_str = get_file_size(kb_file)
        assert 'KB' in size_str or 'B' in size_str
    
    def test_create_hash(self):
        """Test hash creation."""
        # Same input produces same hash
        hash1 = create_hash('test string')
        hash2 = create_hash('test string')
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length
        
        # Different input produces different hash
        hash3 = create_hash('different string')
        assert hash3 != hash1
    
    def test_chunk_list(self):
        """Test list chunking."""
        # Even chunks
        lst = list(range(10))
        chunks = chunk_list(lst, 3)
        assert len(chunks) == 4
        assert chunks[0] == [0, 1, 2]
        assert chunks[-1] == [9]
        
        # Single chunk
        chunks = chunk_list(lst, 20)
        assert len(chunks) == 1
        assert chunks[0] == lst
        
        # Empty list
        chunks = chunk_list([], 5)
        assert chunks == []
    
    def test_merge_dicts(self):
        """Test dictionary merging."""
        dict1 = {
            'a': 1,
            'b': {'x': 1, 'y': 2},
            'c': [1, 2, 3]
        }
        
        dict2 = {
            'a': 2,
            'b': {'y': 3, 'z': 4},
            'd': 4
        }
        
        result = merge_dicts(dict1, dict2)
        
        assert result['a'] == 2  # Overwritten
        assert result['b']['x'] == 1  # Preserved
        assert result['b']['y'] == 3  # Overwritten
        assert result['b']['z'] == 4  # Added
        assert result['c'] == [1, 2, 3]  # Preserved
        assert result['d'] == 4  # Added
    
    def test_timer(self):
        """Test timer context manager."""
        with Timer('test_operation') as timer:
            assert timer.elapsed >= 0
            # Simulate some work
            import time
            time.sleep(0.01)
        
        assert timer.elapsed >= 0.01