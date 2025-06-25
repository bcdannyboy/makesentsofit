"""
Tests for export functionality.
Comprehensive test coverage for Phase 5 components.
"""
import pytest
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import shutil

from src.export import DataFormatter, ExportWriter
from src.scrapers.base import Post


def create_test_post(id, content='Test content', author='testuser', 
                    platform='twitter', sentiment_label='NEUTRAL',
                    timestamp=None, engagement=None):
    """Helper to create test posts."""
    if timestamp is None:
        timestamp = datetime.now() - timedelta(hours=id)
    
    if engagement is None:
        engagement = {'likes': 10 * id, 'retweets': 5 * id}
    
    post = Post(
        id=str(id),
        platform=platform,
        author=author,
        author_id=str(100 + id),
        content=content,
        title=f"Title {id}" if platform == 'reddit' else None,
        timestamp=timestamp,
        engagement=engagement,
        url=f'https://{platform}.com/{id}',
        query='test query',
        metadata={
            'hashtags': ['test', 'export'] if id % 2 == 0 else [],
            'mentions': ['@user1'] if id % 3 == 0 else []
        }
    )
    
    # Add sentiment
    post.sentiment = {
        'label': sentiment_label,
        'score': 0.8,
        'method': 'transformer'
    }
    
    return post


def create_test_context(num_posts=10):
    """Create a complete test context."""
    posts = [create_test_post(i) for i in range(num_posts)]
    
    context = {
        'queries': ['test', 'sample'],
        'time_window_days': 7,
        'platforms': ['twitter', 'reddit'],
        'output_prefix': 'test_analysis',
        'formats': ['json', 'csv', 'html'],
        'visualize': True,
        'version': '1.0.0',
        'start_time': datetime.now() - timedelta(hours=1),
        'collection_time': 3600.0,
        'posts': posts,
        'statistics': {
            'total_posts': len(posts),
            'date_range': {
                'start': (datetime.now() - timedelta(days=7)).isoformat(),
                'end': datetime.now().isoformat(),
                'days': 7
            },
            'by_platform': {'twitter': 6, 'reddit': 4},
            'by_query': {'test': 5, 'sample': 5},
            'sentiment_distribution': {
                'counts': {'POSITIVE': 3, 'NEGATIVE': 3, 'NEUTRAL': 4},
                'percentages': {'POSITIVE': 30.0, 'NEGATIVE': 30.0, 'NEUTRAL': 40.0},
                'sentiment_ratio': 0.0
            },
            'engagement': {
                'total_likes': 450,
                'total_shares': 225,
                'avg_engagement': 67.5,
                'max_engagement': 135
            },
            'authors': {
                'unique_authors': 1,
                'most_active': {'testuser': 10},
                'posts_per_author': {'mean': 10.0}
            },
            'hashtags': {
                'top_hashtags': {'test': 5, 'export': 5},
                'unique_hashtags': 2
            },
            'negative_users': [
                {
                    'author': 'negative_user',
                    'platform': 'twitter',
                    'post_count': 5,
                    'negative_ratio': 0.8,
                    'negative_posts': 4,
                    'positive_posts': 0,
                    'neutral_posts': 1,
                    'avg_sentiment_score': 0.2,
                    'avg_engagement': 50
                }
            ],
            'viral_posts': [
                {
                    'id': '1',
                    'platform': 'twitter',
                    'author': 'viral_user',
                    'engagement': 10000,
                    'likes': 8000,
                    'shares': 2000,
                    'sentiment': 'POSITIVE',
                    'content_preview': 'This is a viral post...',
                    'timestamp': datetime.now().isoformat(),
                    'url': 'https://twitter.com/1'
                }
            ]
        },
        'time_series': {
            'daily_sentiment': {
                '2024-01-01': {
                    'positive': 2,
                    'negative': 1,
                    'neutral': 3,
                    'total': 6,
                    'sentiment_ratio': 0.167,
                    'positive_ratio': 0.333,
                    'total_engagement': 300,
                    'unique_authors': 1
                }
            },
            'trends': {
                'overall_trend': 'stable',
                'trend_strength': 0.1
            },
            'anomalies': [],
            'sentiment_volatility': {
                'overall': 0.5,
                'daily': 0.3,
                'hourly': 0.1
            }
        },
        'deduplication': {
            'total_posts': 12,
            'unique_posts': 10,
            'duplicates_removed': 2
        }
    }
    
    return context


class TestDataFormatter:
    """Test data formatting functionality."""
    
    def test_formatter_initialization(self):
        """Test formatter initialization."""
        formatter = DataFormatter()
        
        assert formatter.timestamp is not None
        assert formatter.timestamp_str is not None
        assert isinstance(formatter.timestamp, datetime)
    
    def test_format_for_json(self):
        """Test JSON formatting."""
        context = create_test_context()
        formatter = DataFormatter()
        
        json_data = formatter.format_for_json(context)
        
        # Check structure
        assert 'metadata' in json_data
        assert 'summary' in json_data
        assert 'statistics' in json_data
        assert 'time_series' in json_data
        assert 'posts' in json_data
        
        # Check metadata
        metadata = json_data['metadata']
        assert metadata['version'] == '1.0.0'
        assert metadata['analysis_parameters']['queries'] == ['test', 'sample']
        assert metadata['analysis_parameters']['time_window_days'] == 7
        
        # Check posts
        assert len(json_data['posts']) == 10
        assert all('sentiment' in post for post in json_data['posts'])
    
    def test_format_for_csv(self):
        """Test CSV formatting."""
        context = create_test_context()
        formatter = DataFormatter()
        
        dataframes = formatter.format_for_csv(context)
        
        # Check DataFrames created
        assert 'posts' in dataframes
        assert 'summary_statistics' in dataframes
        assert 'daily_statistics' in dataframes
        assert 'negative_users' in dataframes
        assert 'viral_posts' in dataframes
        assert 'hashtags' in dataframes
        
        # Check posts DataFrame
        posts_df = dataframes['posts']
        assert len(posts_df) == 10
        assert 'id' in posts_df.columns
        assert 'sentiment' in posts_df.columns
        assert 'engagement_total' in posts_df.columns
        
        # Check summary statistics
        summary_df = dataframes['summary_statistics']
        assert 'metric' in summary_df.columns
        assert 'value' in summary_df.columns
        assert 'category' in summary_df.columns
    
    def test_format_for_html(self):
        """Test HTML formatting."""
        context = create_test_context()
        formatter = DataFormatter()
        
        html_data = formatter.format_for_html(context)
        
        # Check required fields
        assert 'title' in html_data
        assert 'generated_at' in html_data
        assert 'queries' in html_data
        assert 'total_posts' in html_data
        assert 'sentiment_distribution' in html_data
        assert 'chart_data' in html_data
        
        # Check formatting
        assert isinstance(html_data['total_posts'], str)
        assert ',' not in html_data['total_posts'] or len(context['posts']) > 999
        
        # Check chart data
        chart_data = html_data['chart_data']
        if 'sentiment_pie' in chart_data:
            assert 'labels' in chart_data['sentiment_pie']
            assert 'data' in chart_data['sentiment_pie']
    
    def test_text_truncation(self):
        """Test text truncation functionality."""
        formatter = DataFormatter()
        
        # Short text
        assert formatter._truncate_text('short', 10) == 'short'
        
        # Long text
        long_text = 'a' * 200
        truncated = formatter._truncate_text(long_text, 100)
        assert len(truncated) == 100
        assert truncated.endswith('...')
        
        # None text
        assert formatter._truncate_text(None) == ''
    
    def test_date_range_formatting(self):
        """Test date range formatting."""
        formatter = DataFormatter()
        
        # Same day
        date_range = {
            'start': '2024-01-15T10:00:00',
            'end': '2024-01-15T18:00:00',
            'days': 1
        }
        formatted = formatter._format_date_range(date_range)
        assert 'January 15, 2024' in formatted
        
        # Different months
        date_range = {
            'start': '2024-01-01T00:00:00',
            'end': '2024-02-01T00:00:00',
            'days': 31
        }
        formatted = formatter._format_date_range(date_range)
        assert 'January' in formatted
        assert 'February' in formatted
    
    def test_empty_context_handling(self):
        """Test handling of empty or minimal context."""
        formatter = DataFormatter()
        
        # Empty context
        empty_context = {'posts': []}
        json_data = formatter.format_for_json(empty_context)
        assert json_data['posts'] == []
        
        csv_data = formatter.format_for_csv(empty_context)
        assert len(csv_data) >= 1  # At least posts DataFrame
        
        html_data = formatter.format_for_html(empty_context)
        assert html_data['total_posts'] == '0'


class TestExportWriter:
    """Test export writer functionality."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_writer_initialization(self, temp_output_dir):
        """Test writer initialization."""
        writer = ExportWriter(temp_output_dir)
        
        assert writer.output_dir.exists()
        assert writer.jinja_env is not None
    
    def test_write_json(self, temp_output_dir):
        """Test JSON writing."""
        writer = ExportWriter(temp_output_dir)
        data = {'test': 'data', 'number': 42}
        
        filepath = writer.write_json(data, 'test_output')
        
        assert filepath.exists()
        assert filepath.suffix == '.json'
        
        # Verify content
        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded == data
    
    def test_write_csv(self, temp_output_dir):
        """Test CSV writing."""
        writer = ExportWriter(temp_output_dir)
        
        # Create test DataFrames
        dataframes = {
            'posts': pd.DataFrame([
                {'id': '1', 'content': 'Test 1'},
                {'id': '2', 'content': 'Test 2'}
            ]),
            'stats': pd.DataFrame([
                {'metric': 'total_posts', 'value': 100},
                {'metric': 'unique_authors', 'value': 50}
            ])
        }
        
        filepaths = writer.write_csv(dataframes, 'test_output')
        
        assert len(filepaths) == 2
        assert all(fp.exists() for fp in filepaths)
        assert all(fp.suffix == '.csv' for fp in filepaths)
        
        # Verify content
        posts_df = pd.read_csv(filepaths[0])
        assert len(posts_df) == 2
        assert 'id' in posts_df.columns
    
    def test_write_html(self, temp_output_dir):
        """Test HTML writing."""
        writer = ExportWriter(temp_output_dir)
        
        # Mock template
        template_content = """
        <html>
        <head><title>{{ title }}</title></head>
        <body>
            <h1>{{ title }}</h1>
            <p>Total posts: {{ total_posts }}</p>
        </body>
        </html>
        """
        
        with patch.object(writer.jinja_env, 'get_template') as mock_get:
            mock_template = Mock()
            mock_template.render.return_value = template_content
            mock_get.return_value = mock_template
            
            data = {
                'title': 'Test Report',
                'total_posts': '100'
            }
            
            filepath = writer.write_html(data, 'test_output')
            
            assert filepath.exists()
            assert filepath.suffix == '.html'
            mock_template.render.assert_called_once()
    
    def test_write_summary(self, temp_output_dir):
        """Test summary writing."""
        writer = ExportWriter(temp_output_dir)
        context = create_test_context()
        
        filepath = writer.write_summary(context, 'test_output')
        
        assert filepath.exists()
        assert '_summary.json' in filepath.name
        
        # Verify content
        with open(filepath) as f:
            summary = json.load(f)
        
        assert 'metadata' in summary
        assert 'key_metrics' in summary
        assert 'sentiment_summary' in summary
        assert summary['key_metrics']['total_posts'] == 10
    
    def test_create_archive(self, temp_output_dir):
        """Test archive creation."""
        writer = ExportWriter(temp_output_dir)
        
        # Create test files
        test_files = []
        for i in range(3):
            filepath = writer.output_dir / f"test_{i}.txt"
            filepath.write_text(f"Test content {i}")
            test_files.append(filepath)
        
        archive_path = writer.create_archive(test_files, 'test_archive')
        
        assert archive_path is not None
        assert archive_path.exists()
        assert archive_path.suffix == '.zip'
        
        # Verify archive contents
        import zipfile
        with zipfile.ZipFile(archive_path, 'r') as zf:
            assert len(zf.namelist()) == 3
    
    def test_file_size_formatting(self):
        """Test file size formatting."""
        writer = ExportWriter()
        
        assert writer._format_file_size(500) == '500.0 B'
        assert writer._format_file_size(1536) == '1.5 KB'
        assert writer._format_file_size(1048576) == '1.0 MB'
        assert writer._format_file_size(1073741824) == '1.0 GB'
    
    def test_error_handling(self, temp_output_dir):
        """Test error handling in writers."""
        writer = ExportWriter(temp_output_dir)
        
        # Test writing to invalid path
        with patch('builtins.open', side_effect=IOError("Permission denied")):
            with pytest.raises(IOError):
                writer.write_json({'test': 'data'}, 'test')
        
        # Test empty DataFrame handling
        dataframes = {'empty': pd.DataFrame()}
        filepaths = writer.write_csv(dataframes, 'test')
        assert len(filepaths) == 0  # Should skip empty DataFrames


class TestExportIntegration:
    """Integration tests for export functionality."""
    
    def test_full_export_pipeline(self, temp_output_dir):
        """Test complete export pipeline."""
        # Create context with real data
        context = create_test_context(50)
        
        # Initialize components
        formatter = DataFormatter()
        writer = ExportWriter(temp_output_dir)
        
        # Export all formats
        exported_files = []
        
        # JSON export
        json_data = formatter.format_for_json(context)
        json_path = writer.write_json(json_data, 'integration_test')
        exported_files.append(json_path)
        
        # CSV export
        csv_data = formatter.format_for_csv(context)
        csv_paths = writer.write_csv(csv_data, 'integration_test')
        exported_files.extend(csv_paths)
        
        # HTML export (mock template)
        with patch.object(writer.jinja_env, 'get_template') as mock_get:
            mock_template = Mock()
            mock_template.render.return_value = "<html>Test Report</html>"
            mock_get.return_value = mock_template
            
            html_data = formatter.format_for_html(context)
            html_path = writer.write_html(html_data, 'integration_test')
            exported_files.append(html_path)
        
        # Summary export
        summary_path = writer.write_summary(context, 'integration_test')
        exported_files.append(summary_path)
        
        # Verify all files exist
        assert all(f.exists() for f in exported_files)
        assert len(exported_files) >= 4  # At least JSON, HTML, summary, and 1+ CSV
        
        # Create archive
        archive_path = writer.create_archive(exported_files, 'integration_test')
        assert archive_path is not None
        assert archive_path.exists()
    
    def test_large_dataset_export(self, temp_output_dir):
        """Test export with large dataset."""
        # Create large context
        context = create_test_context(1000)
        
        formatter = DataFormatter()
        writer = ExportWriter(temp_output_dir)
        
        # Time the export
        import time
        start_time = time.time()
        
        # Export JSON
        json_data = formatter.format_for_json(context)
        json_path = writer.write_json(json_data, 'large_dataset')
        
        # Export CSV
        csv_data = formatter.format_for_csv(context)
        csv_paths = writer.write_csv(csv_data, 'large_dataset')
        
        export_time = time.time() - start_time
        
        # Verify performance
        assert export_time < 10  # Should complete in under 10 seconds
        assert json_path.exists()
        assert len(csv_paths) > 0
        
        # Check file sizes are reasonable
        json_size = json_path.stat().st_size
        assert json_size > 1000  # Should have substantial content
    
    def test_export_with_missing_data(self, temp_output_dir):
        """Test export with incomplete data."""
        # Create context with missing fields
        context = {
            'posts': [create_test_post(1)],
            'statistics': {},  # Empty stats
            'time_series': {}  # Empty time series
        }
        
        formatter = DataFormatter()
        writer = ExportWriter(temp_output_dir)
        
        # Should handle gracefully
        json_data = formatter.format_for_json(context)
        json_path = writer.write_json(json_data, 'minimal')
        
        csv_data = formatter.format_for_csv(context)
        csv_paths = writer.write_csv(csv_data, 'minimal')
        
        assert json_path.exists()
        assert len(csv_paths) >= 1  # At least posts CSV
    
    def test_unicode_handling(self, temp_output_dir):
        """Test handling of Unicode content."""
        # Create posts with Unicode content
        posts = [
            create_test_post(1, content='Hello 世界 🌍'),
            create_test_post(2, content='Émojis: 😀😃😄😁'),
            create_test_post(3, content='Ñoño José François')
        ]
        
        context = create_test_context()
        context['posts'] = posts
        
        formatter = DataFormatter()
        writer = ExportWriter(temp_output_dir)
        
        # Export all formats
        json_data = formatter.format_for_json(context)
        json_path = writer.write_json(json_data, 'unicode_test')
        
        csv_data = formatter.form