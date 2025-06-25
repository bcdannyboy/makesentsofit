"""
Export writers for different file formats.
Handles writing formatted data to JSON, CSV, and HTML files.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
import logging
import shutil

logger = logging.getLogger(__name__)

class ExportWriter:
    """Write formatted data to files in various formats."""
    
    def __init__(self, output_dir: str = './output'):
        """
        Initialize export writer.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self._ensure_output_dir()
        
        # Setup Jinja2 for HTML templates
        template_dir = Path(__file__).parent / 'templates'
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        self.jinja_env.filters['format_number'] = self._format_number
        self.jinja_env.filters['format_percentage'] = self._format_percentage
        self.jinja_env.filters['truncate_text'] = self._truncate_text
        
        logger.debug(f"ExportWriter initialized with output_dir: {self.output_dir}")
    
    def _ensure_output_dir(self):
        """Ensure output directory exists."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured output directory exists: {self.output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            raise
    
    def write_json(self, data: Dict[str, Any], filename_prefix: str, 
                   indent: int = 2, ensure_ascii: bool = False) -> Path:
        """
        Write JSON file.
        
        Args:
            data: Data to write
            filename_prefix: Prefix for filename
            indent: JSON indentation level
            ensure_ascii: Whether to escape non-ASCII characters
            
        Returns:
            Path to written file
        """
        filepath = self.output_dir / f"{filename_prefix}.json"
        
        try:
            logger.info(f"Writing JSON to: {filepath}")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, 
                         default=str, sort_keys=True)
            
            # Log file size
            file_size = filepath.stat().st_size
            logger.info(f"Wrote JSON file: {filepath.name} ({self._format_file_size(file_size)})")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to write JSON file: {e}")
            raise
    
    def write_csv(self, dataframes: Dict[str, pd.DataFrame], 
                  filename_prefix: str) -> List[Path]:
        """
        Write multiple CSV files.
        
        Args:
            dataframes: Dictionary mapping names to DataFrames
            filename_prefix: Prefix for filenames
            
        Returns:
            List of paths to written files
        """
        filepaths = []
        
        for name, df in dataframes.items():
            if df.empty:
                logger.warning(f"Skipping empty DataFrame: {name}")
                continue
                
            filepath = self.output_dir / f"{filename_prefix}_{name}.csv"
            
            try:
                logger.info(f"Writing CSV to: {filepath}")
                
                # Write CSV with proper encoding and formatting
                df.to_csv(
                    filepath, 
                    index=False, 
                    encoding='utf-8',
                    date_format='%Y-%m-%d %H:%M:%S'
                )
                
                # Log file info
                file_size = filepath.stat().st_size
                rows, cols = df.shape
                logger.info(f"Wrote CSV file: {filepath.name} "
                          f"({rows} rows, {cols} columns, {self._format_file_size(file_size)})")
                
                filepaths.append(filepath)
                
            except Exception as e:
                logger.error(f"Failed to write CSV file {name}: {e}")
                # Continue with other files
        
        return filepaths
    
    def write_html(self, data: Dict[str, Any], filename_prefix: str) -> Path:
        """
        Write HTML report.
        
        Args:
            data: Data for HTML template
            filename_prefix: Prefix for filename
            
        Returns:
            Path to written file
        """
        filepath = self.output_dir / f"{filename_prefix}_report.html"
        
        try:
            logger.info(f"Writing HTML report to: {filepath}")
            
            # Load and render template
            template = self.jinja_env.get_template('report.html')
            html_content = template.render(**data)
            
            # Write HTML file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Copy CSS file if it exists
            self._copy_static_files(filename_prefix)
            
            # Log file info
            file_size = filepath.stat().st_size
            logger.info(f"Wrote HTML report: {filepath.name} ({self._format_file_size(file_size)})")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to write HTML report: {e}")
            raise
    
    def write_summary(self, context: Dict[str, Any], filename_prefix: str) -> Path:
        """
        Write a summary JSON with just key statistics.
        
        Args:
            context: Full analysis context
            filename_prefix: Prefix for filename
            
        Returns:
            Path to written file
        """
        # Extract key metrics for summary
        stats = context.get('statistics', {})
        time_series = context.get('time_series', {})
        
        summary = {
            'metadata': {
                'generated_at': context.get('start_time', ''),
                'queries': context.get('queries', []),
                'time_window_days': context.get('time_window_days', 0),
                'platforms': context.get('platforms', [])
            },
            'key_metrics': {
                'total_posts': stats.get('total_posts', 0),
                'unique_authors': stats.get('authors', {}).get('unique_authors', 0),
                'date_range_days': stats.get('date_range', {}).get('days', 0)
            },
            'sentiment_summary': {
                'distribution': stats.get('sentiment_distribution', {}).get('counts', {}),
                'percentages': stats.get('sentiment_distribution', {}).get('percentages', {}),
                'sentiment_ratio': stats.get('sentiment_distribution', {}).get('sentiment_ratio', 0),
                'dominant_sentiment': stats.get('sentiment_distribution', {}).get('dominant_sentiment', '')
            },
            'engagement_summary': {
                'total_engagement': stats.get('engagement', {}).get('total_engagement', 0),
                'avg_engagement': stats.get('engagement', {}).get('avg_engagement', 0),
                'viral_posts_count': len(stats.get('viral_posts', []))
            },
            'trends': {
                'overall_trend': time_series.get('trends', {}).get('overall_trend', ''),
                'trend_strength': time_series.get('trends', {}).get('trend_strength', 0),
                'volatility': time_series.get('sentiment_volatility', {}).get('overall', 0),
                'anomalies_count': len(time_series.get('anomalies', []))
            },
            'negative_users_summary': {
                'count': len(stats.get('negative_users', [])),
                'top_3': [
                    {
                        'author': u['author'],
                        'negative_ratio': u['negative_ratio'],
                        'post_count': u['post_count']
                    }
                    for u in stats.get('negative_users', [])[:3]
                ]
            },
            'processing_summary': context.get('deduplication', {})
        }
        
        filepath = self.output_dir / f"{filename_prefix}_summary.json"
        
        try:
            logger.info(f"Writing summary to: {filepath}")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Wrote summary file: {filepath.name}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to write summary file: {e}")
            raise
    
    def create_archive(self, files: List[Path], archive_name: str) -> Optional[Path]:
        """
        Create a ZIP archive of exported files.
        
        Args:
            files: List of file paths to archive
            archive_name: Name for the archive
            
        Returns:
            Path to archive file or None if failed
        """
        import zipfile
        
        archive_path = self.output_dir / f"{archive_name}.zip"
        
        try:
            logger.info(f"Creating archive: {archive_path}")
            
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in files:
                    if file_path.exists():
                        arcname = file_path.name
                        zipf.write(file_path, arcname)
                        logger.debug(f"Added to archive: {arcname}")
            
            archive_size = archive_path.stat().st_size
            logger.info(f"Created archive: {archive_path.name} ({self._format_file_size(archive_size)})")
            
            return archive_path
            
        except Exception as e:
            logger.error(f"Failed to create archive: {e}")
            return None
    
    def _copy_static_files(self, filename_prefix: str):
        """Copy static files (CSS, JS) for HTML reports."""
        template_dir = Path(__file__).parent / 'templates'
        static_files = ['styles.css']
        
        for static_file in static_files:
            src = template_dir / static_file
            if src.exists():
                dst = self.output_dir / f"{filename_prefix}_{static_file}"
                try:
                    shutil.copy2(src, dst)
                    logger.debug(f"Copied static file: {static_file}")
                except Exception as e:
                    logger.warning(f"Failed to copy static file {static_file}: {e}")
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def _format_number(self, value: Any) -> str:
        """Format number with thousands separator."""
        if isinstance(value, (int, float)):
            if value == int(value):
                return f"{int(value):,}"
            return f"{value:,.2f}"
        return str(value)
    
    def _format_percentage(self, value: Any) -> str:
        """Format percentage value."""
        if isinstance(value, (int, float)):
            return f"{value:.1f}%"
        return str(value)
    
    def _truncate_text(self, text: str, max_length: int = 100) -> str:
        """Truncate text to maximum length."""
        if not text or len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."