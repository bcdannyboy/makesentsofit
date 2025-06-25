#!/usr/bin/env python3
"""
MakeSenseOfIt - Social Media Sentiment Analysis CLI
Main entry point for the application.
"""
import click
import sys
import os
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.cli import validate_cli_args, format_output_summary
from src.config import Config
from src.logger import setup_logging, get_logger
from src.utils import ensure_directory, generate_timestamp

# Version
__version__ = '1.0.0'

@click.command()
@click.option('--queries', '-q', required=True, help='Comma-separated search queries')
@click.option('--time', '-t', default=7, type=int, help='Days to look back')
@click.option('--platforms', '-p', default='twitter,reddit', help='Platforms to scrape')
@click.option('--output', '-o', help='Output file prefix')
@click.option('--format', '-f', default='json', help='Output format(s): json,csv,html')
@click.option('--visualize', '-v', is_flag=True, help='Generate visualizations')
@click.option('--verbose', is_flag=True, help='Verbose output')
@click.option('--config', type=click.Path(exists=True), help='Config file path')
@click.version_option(version=__version__)
def main(queries, time, platforms, output, format, visualize, verbose, config):
    """
    Analyze sentiment across social media platforms.
    
    Examples:
        makesentsofit --queries "bitcoin,btc" --time 7
        makesentsofit -q "climate change" -t 30 -p twitter,reddit -v
        makesentsofit -q "ai" -f json,csv,html -o ai_analysis
    """
    # Setup logging first
    setup_logging(verbose)
    logger = get_logger(__name__)
    
    try:
        # Load configuration
        logger.debug(f"Loading configuration from: {config or 'defaults'}")
        cfg = Config(config_file=config)
        
        # Parse and validate arguments
        query_list = [q.strip() for q in queries.split(',')]
        platform_list = [p.strip().lower() for p in platforms.split(',')]
        format_list = [f.strip().lower() for f in format.split(',')]
        
        # Validate arguments
        validation_errors = validate_cli_args(
            queries=query_list,
            time_window=time,
            platforms=platform_list,
            formats=format_list
        )
        
        if validation_errors:
            for error in validation_errors:
                click.echo(f"❌ {error}", err=True)
            sys.exit(1)
        
        # Generate output prefix if not provided
        if not output:
            timestamp = generate_timestamp()
            output = f"analysis_{timestamp}"
        
        # Ensure output directory exists
        output_dir = Path(cfg.output_directory)
        ensure_directory(str(output_dir))
        
        # Create analysis context
        context = {
            'queries': query_list,
            'time_window_days': time,
            'platforms': platform_list,
            'output_prefix': output,
            'output_directory': str(output_dir),
            'formats': format_list,
            'visualize': visualize,
            'config': cfg,
            'version': __version__,
            'start_time': datetime.now()
        }
        
        # Display header
        click.echo("\n" + "="*50)
        click.echo("🔍 MakeSenseOfIt - Social Media Sentiment Analysis")
        click.echo("="*50)
        
        # Log configuration
        click.echo(f"\n📋 Configuration:")
        click.echo(f"  • Queries: {', '.join(query_list)}")
        click.echo(f"  • Time window: {time} days")
        click.echo(f"  • Platforms: {', '.join(platform_list)}")
        click.echo(f"  • Output formats: {', '.join(format_list)}")
        click.echo(f"  • Output prefix: {output}")
        click.echo(f"  • Visualizations: {'Yes' if visualize else 'No'}")
        
        if verbose:
            click.echo(f"\n🔧 Advanced Settings:")
            click.echo(f"  • Config file: {config or 'Using defaults'}")
            click.echo(f"  • Output directory: {output_dir}")
            click.echo(f"  • Cache directory: {cfg.cache_directory}")
            click.echo(f"  • Sentiment model: {cfg.sentiment_model}")
            
            click.echo(f"\n⚡ Rate Limits:")
            for platform in platform_list:
                limit = cfg.get_rate_limit(platform)
                click.echo(f"  • {platform}: {limit} requests/minute")
        
        # Phase 1: Configuration validation complete
        click.echo("\n✅ Configuration validated successfully!")
        click.echo("📝 Ready for Phase 2: Data Collection")
        
        # Save configuration summary
        if verbose:
            summary = format_output_summary(context)
            logger.debug(f"Analysis context:\n{summary}")
        
        return context
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        if verbose:
            logger.exception("Full traceback:")
        sys.exit(1)

if __name__ == '__main__':
    main()