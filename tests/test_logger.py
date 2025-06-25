"""
Tests for logging functionality.
"""
import pytest
import logging
from pathlib import Path
from src.logger import (
    setup_logging, get_logger, log_banner, log_success,
    log_error, log_warning, log_info, LogContext
)

class TestLogger:
    """Test logging functionality."""
    
    def test_setup_logging_default(self):
        """Test default logging setup."""
        logger = setup_logging(verbose=False)
        
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
    
    def test_setup_logging_verbose(self):
        """Test verbose logging setup."""
        logger = setup_logging(verbose=True)
        
        assert logger.level == logging.DEBUG
    
    def test_setup_logging_with_file(self, temp_dir):
        """Test logging to file."""
        log_file = temp_dir / 'test.log'
        logger = setup_logging(log_file=str(log_file))
        
        # Log something
        test_logger = get_logger('test')
        test_logger.info('Test message')
        
        # Check file was created and contains message
        assert log_file.exists()
        with open(log_file) as f:
            content = f.read()
        assert 'Test message' in content
    
    def test_get_logger(self):
        """Test getting logger instance."""
        logger1 = get_logger('test.module1')
        logger2 = get_logger('test.module2')
        
        assert logger1.name == 'test.module1'
        assert logger2.name == 'test.module2'
        assert logger1 != logger2
    
    def test_log_context(self, caplog):
        """Test logging context manager."""
        with LogContext('test_operation') as ctx:
            assert ctx.operation == 'test_operation'
            assert ctx.start_time is not None
        
        # Check logs
        assert 'Starting: test_operation' in caplog.text
        assert 'Completed: test_operation' in caplog.text
    
    def test_log_context_with_exception(self, caplog):
        """Test logging context with exception."""
        try:
            with LogContext('failing_operation'):
                raise ValueError('Test error')
        except ValueError:
            pass
        
        assert 'Failed: failing_operation' in caplog.text
        assert 'Test error' in caplog.text
    
    def test_log_functions(self, capsys):
        """Test convenience logging functions."""
        # These functions print to console via Rich
        log_banner('Test Banner', 'Subtitle')
        log_success('Success message')
        log_error('Error message')
        log_warning('Warning message')
        log_info('Info message')
        
        captured = capsys.readouterr()
        assert 'Test Banner' in captured.out
        assert 'Success message' in captured.out
        assert 'Error message' in captured.out
        assert 'Warning message' in captured.out
        assert 'Info message' in captured.out
    
    def test_third_party_logging_suppressed(self):
        """Test that third-party library logging is suppressed."""
        setup_logging()
        
        urllib_logger = logging.getLogger('urllib3')
        requests_logger = logging.getLogger('requests')
        
        assert urllib_logger.level == logging.WARNING
        assert requests_logger.level == logging.WARNING