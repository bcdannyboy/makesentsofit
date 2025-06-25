"""
Tests for CLI functionality.
"""
import pytest
from click.testing import CliRunner
from makesentsofit import main
from src.cli import (
    validate_cli_args, validate_queries, parse_platforms,
    format_output_summary, create_progress_bar,
    VALID_PLATFORMS, VALID_FORMATS
)

class TestCLI:
    """Test CLI functionality."""
    
    def test_cli_basic(self):
        """Test basic CLI functionality."""
        runner = CliRunner()
        result = runner.invoke(main, ['--queries', 'test', '--time', '1'])
        
        assert result.exit_code == 0
        assert 'MakeSenseOfIt' in result.output
        assert 'Queries: test' in result.output
        assert 'Configuration validated successfully' in result.output
    
    def test_cli_multiple_queries(self):
        """Test multiple queries parsing."""
        runner = CliRunner()
        result = runner.invoke(main, [
            '--queries', 'test1,test2,test3',
            '--time', '7'
        ])
        
        assert result.exit_code == 0
        assert 'test1' in result.output
        assert 'test2' in result.output
        assert 'test3' in result.output
    
    def test_cli_missing_queries(self):
        """Test that queries are required."""
        runner = CliRunner()
        result = runner.invoke(main, ['--time', '7'])
        
        assert result.exit_code != 0
        assert 'Error' in result.output or 'required' in result.output
    
    def test_cli_all_options(self):
        """Test CLI with all options."""
        runner = CliRunner()
        result = runner.invoke(main, [
            '--queries', 'bitcoin,btc',
            '--time', '30',
            '--platforms', 'twitter,reddit',
            '--output', 'btc_analysis',
            '--format', 'json,csv,html',
            '--visualize',
            '--verbose'
        ])
        
        assert result.exit_code == 0
        assert 'bitcoin' in result.output
        assert 'btc' in result.output
        assert 'Visualizations: Yes' in result.output
        assert 'Advanced Settings' in result.output  # Verbose mode
    
    def test_cli_invalid_platform(self):
        """Test CLI with invalid platform."""
        runner = CliRunner()
        result = runner.invoke(main, [
            '--queries', 'test',
            '--platforms', 'twitter,facebook,invalid'
        ])
        
        assert result.exit_code == 1
        assert 'Invalid platforms' in result.output
    
    def test_cli_invalid_time_window(self):
        """Test CLI with invalid time window."""
        runner = CliRunner()
        result = runner.invoke(main, [
            '--queries', 'test',
            '--time', '0'
        ])
        
        assert result.exit_code == 1
        assert 'Time window must be at least 1 day' in result.output
    
    def test_cli_with_config_file(self, sample_config_file):
        """Test CLI with configuration file."""
        runner = CliRunner()
        result = runner.invoke(main, [
            '--queries', 'test',
            '--config', str(sample_config_file)
        ])
        
        assert result.exit_code == 0
        assert 'Configuration validated successfully' in result.output
    
    def test_validate_cli_args(self):
        """Test CLI argument validation."""
        # Valid arguments
        errors = validate_cli_args(
            queries=['test'],
            time_window=7,
            platforms=['twitter'],
            formats=['json']
        )
        assert len(errors) == 0
        
        # Empty queries
        errors = validate_cli_args(
            queries=[],
            time_window=7,
            platforms=['twitter'],
            formats=['json']
        )
        assert 'No queries provided' in errors
        
        # Invalid time window
        errors = validate_cli_args(
            queries=['test'],
            time_window=0,
            platforms=['twitter'],
            formats=['json']
        )
        assert any('Time window' in e for e in errors)
        
        # Invalid platform
        errors = validate_cli_args(
            queries=['test'],
            time_window=7,
            platforms=['invalid'],
            formats=['json']
        )
        assert any('Invalid platforms' in e for e in errors)
        
        # Invalid format
        errors = validate_cli_args(
            queries=['test'],
            time_window=7,
            platforms=['twitter'],
            formats=['invalid']
        )
        assert any('Invalid formats' in e for e in errors)
    
    def test_validate_queries(self):
        """Test query validation."""
        # Valid queries
        queries, warnings = validate_queries(['test', ' spaces ', 'multiple words'])
        assert len(queries) == 3
        assert queries[0] == 'test'
        assert queries[1] == 'spaces'
        assert queries[2] == 'multiple words'
        assert len(warnings) == 0
        
        # Empty queries filtered out
        queries, warnings = validate_queries(['test', '', '  ', 'valid'])
        assert len(queries) == 2
        assert 'test' in queries
        assert 'valid' in queries
        
        # Short queries
        queries, warnings = validate_queries(['a', 'test'])
        assert len(queries) == 1
        assert queries[0] == 'test'
        assert any('too short' in w for w in warnings)
        
        # Long queries truncated
        long_query = 'a' * 150
        queries, warnings = validate_queries([long_query])
        assert len(queries) == 1
        assert len(queries[0]) == 100
        assert any('truncated' in w for w in warnings)
    
    def test_parse_platforms(self):
        """Test platform parsing."""
        # Valid platforms
        valid, invalid = parse_platforms('twitter,reddit')
        assert len(valid) == 2
        assert 'twitter' in valid
        assert 'reddit' in valid
        assert len(invalid) == 0
        
        # Case insensitive
        valid, invalid = parse_platforms('Twitter,REDDIT')
        assert 'twitter' in valid
        assert 'reddit' in valid
        
        # Mixed valid and invalid
        valid, invalid = parse_platforms('twitter,facebook,reddit,instagram')
        assert len(valid) == 2
        assert 'twitter' in valid
        assert 'reddit' in valid
        assert len(invalid) == 2
        assert 'facebook' in invalid
        assert 'instagram' in invalid
    
    def test_format_output_summary(self, mock_context):
        """Test output summary formatting."""
        summary = format_output_summary(mock_context)
        
        assert 'Analysis Configuration Summary' in summary
        assert 'Version: 1.0.0' in summary
        assert 'test' in summary
        assert 'sample' in summary
        assert 'Time Window: 7 days' in summary
        assert 'Platforms: twitter, reddit' in summary
    
    def test_create_progress_bar(self):
        """Test progress bar creation."""
        # Empty progress
        bar = create_progress_bar(0, 100)
        assert bar.startswith('[')
        assert bar.endswith(']')
        assert '=' not in bar
        
        # Half progress
        bar = create_progress_bar(50, 100, width=10)
        assert bar.count('=') == 5
        assert bar.count('-') == 5
        
        # Full progress
        bar = create_progress_bar(100, 100, width=10)
        assert bar.count('=') == 10
        assert '-' not in bar
        
        # Zero total (edge case)
        bar = create_progress_bar(0, 0)
        assert '[' in bar and ']' in bar