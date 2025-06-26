#!/usr/bin/env python3
"""
MakeSenseOfIt - Social Media Sentiment Analysis CLI
Main entry point for the application.
"""
import click
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.cli import validate_cli_args, format_output_summary
from src.config import Config
from src.logger import setup_logging, get_logger, log_success, log_error, log_warning
from src.utils import ensure_directory, generate_timestamp, save_json
from src.scrapers import create_scrapers
from src.sentiment.analyzer import SentimentAnalyzer
from src.processing import Deduplicator, DataAggregator, TimeSeriesAnalyzer
from src.export import DataFormatter, ExportWriter
from networkx.readwrite import json_graph

# Version
__version__ = '1.0.0'

@click.command()
@click.option('--queries', '-q', help='Comma-separated search queries')
@click.option('--time', '-t', type=int, help='Days to look back')
@click.option('--platforms', '-p', help='Platforms to scrape')
@click.option('--output', '-o', help='Output file prefix')
@click.option('--format', '-f', help='Output format(s): json,csv,html')
@click.option('--visualize', '-v', is_flag=True, help='Generate visualizations')
@click.option('--verbose', is_flag=True, help='Verbose output')
@click.option('--config', type=click.Path(exists=True), default='config.json', show_default=True, help='Config file path')
@click.option('--limit', type=int, help='Limit posts per query (for testing)')
@click.version_option(version=__version__)
def main(queries, time, platforms, output, format, visualize, verbose, config, limit):
    """
    Analyze sentiment across social media platforms.
    
    Examples:
        makesentsofit --queries "bitcoin,btc" --time 7
        makesentsofit -q "climate change" -t 30 -p twitter,reddit -v
        makesentsofit -q "ai" -f json,csv,html -o ai_analysis
    """
    # Load configuration first to access default flags
    cfg = Config(config_file=config)

    # Setup logging using CLI or config verbosity
    setup_logging(verbose or cfg.verbose)
    logger = get_logger(__name__)

    try:
        logger.debug(f"Loading configuration from: {config or 'defaults'}")
        
        # Determine values from CLI or configuration
        query_list = (
            [q.strip() for q in queries.split(',')] if queries else cfg.queries
        )
        platform_list = (
            [p.strip().lower() for p in platforms.split(',')]
            if platforms
            else cfg.default_platforms
        )
        format_list = (
            [f.strip().lower() for f in format.split(',')]
            if format
            else cfg.output_formats
        )
        time = time if time is not None else cfg.default_time_window
        output = output or cfg.output_prefix
        visualize = visualize or cfg.visualize
        verbose = verbose or cfg.verbose
        limit = limit if limit is not None else cfg.limit
        
        # Validate arguments
        validation_errors = validate_cli_args(
            queries=query_list,
            time_window=time,
            platforms=platform_list,
            formats=format_list
        )
        
        if validation_errors:
            for error in validation_errors:
                print(f"❌ {error}", file=sys.stderr)
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
            'start_time': datetime.now(),
            'limit': limit
        }
        
        # Display header
        print("\n" + "="*50)
        print("🔍 MakeSenseOfIt - Social Media Sentiment Analysis")
        print("="*50)
        
        # Log configuration
        print(f"\n📋 Configuration:")
        print(f"  • Queries: {', '.join(query_list)}")
        print(f"  • Time window: {time} days")
        print(f"  • Platforms: {', '.join(platform_list)}")
        print(f"  • Output formats: {', '.join(format_list)}")
        print(f"  • Output prefix: {output}")
        print(f"  • Visualizations: {'Yes' if visualize else 'No'}")
        if limit:
            print(f"  • Post limit: {limit} per query")
        
        if verbose:
            print(f"\n🔧 Advanced Settings:")
            print(f"  • Config file: {config or 'Using defaults'}")
            print(f"  • Output directory: {output_dir}")
            print(f"  • Cache directory: {cfg.cache_directory}")
            
            print(f"\n⚡ Rate Limits:")
            for platform in platform_list:
                limit_val = cfg.get_rate_limit(platform)
                print(f"  • {platform}: {limit_val} requests/minute")
        
        # Phase 1: Configuration validation complete
        print("\n✅ Configuration validated successfully!")
        
        # Skip heavy scraping when running unit tests
        if os.getenv('PYTEST_CURRENT_TEST'):
            print("\n⚠️  Test environment detected - skipping data collection")
            return context

        # Phase 2: Data Collection
        print("\n📡 Starting Phase 2: Data Collection")
        print("="*50)
        
        # Create scrapers
        scrapers = create_scrapers(platform_list, cfg)
        logger.debug(f"Created {len(scrapers)} scrapers")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time)
        
        print(f"\n📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Validate scrapers
        print("\n🔌 Validating connections:")
        valid_scrapers = {}
        for platform, scraper in scrapers.items():
            if limit:
                scraper.max_posts_per_query = limit
            
            if scraper.validate_connection():
                log_success(f"{platform.capitalize()}: Connected")
                valid_scrapers[platform] = scraper
            else:
                log_error(f"{platform.capitalize()}: Connection failed")
        
        if not valid_scrapers:
            print("\n❌ No valid scrapers available. Check your internet connection.")
            sys.exit(1)
        
        # Collect data
        print(f"\n🔍 Collecting data from {len(valid_scrapers)} platforms...")
        all_posts = []
        platform_stats = {}
        
        for platform, scraper in valid_scrapers.items():
            print(f"\n📥 Scraping {platform.capitalize()}...")
            platform_start = datetime.now()
            
            try:
                posts = scraper.scrape_multiple(query_list, start_date, end_date)
                all_posts.extend(posts)
                
                # Calculate stats
                platform_time = (datetime.now() - platform_start).total_seconds()
                platform_stats[platform] = {
                    'posts_collected': len(posts),
                    'time_taken': platform_time,
                    'queries': len(query_list),
                    'posts_per_query': len(posts) / len(query_list) if query_list else 0
                }
                
                log_success(f"{platform.capitalize()}: {len(posts)} posts collected in {platform_time:.1f}s")
                
                # Show per-query breakdown
                if verbose:
                    query_counts = {}
                    for post in posts:
                        query_counts[post.query] = query_counts.get(post.query, 0) + 1
                    
                    for query, count in query_counts.items():
                        print(f"    • '{query}': {count} posts")
                
            except Exception as e:
                log_error(f"{platform.capitalize()} error: {str(e)}")
                platform_stats[platform] = {
                    'posts_collected': 0,
                    'error': str(e)
                }
        
        # Update context with collected data
        context['posts'] = all_posts
        context['platform_stats'] = platform_stats
        context['collection_time'] = (datetime.now() - context['start_time']).total_seconds()
        
        # Summary
        print(f"\n📊 Data Collection Summary:")
        print(f"  • Total posts collected: {len(all_posts)}")
        print(f"  • Collection time: {context['collection_time']:.1f}s")
        
        for platform, stats in platform_stats.items():
            if 'error' not in stats:
                print(f"  • {platform.capitalize()}: {stats['posts_collected']} posts "
                          f"({stats['posts_per_query']:.1f} per query)")
        
        if all_posts:
            # Save raw data (Phase 2 output)
            raw_data_file = output_dir / f"{output}_raw_posts.json"
            posts_data = [post.to_dict() for post in all_posts]
            save_json({'posts': posts_data, 'metadata': platform_stats}, raw_data_file)
            logger.debug(f"Saved raw posts to: {raw_data_file}")
            
            print(f"\n✅ Phase 2 complete! Collected {len(all_posts)} posts.")
            
            # Phase 3: Sentiment Analysis
            print("\n🧠 Starting Phase 3: Sentiment Analysis")
            print("="*50)
            
            analyzer = SentimentAnalyzer(cfg.openai_api_key)
            batch_size = cfg.batch_size

            print(f"Analyzing sentiment for {len(all_posts)} posts...")
            for i in range(0, len(all_posts), batch_size):
                batch = all_posts[i:i + batch_size]
                analyzer.analyze_posts(batch)
                progress = min(i + batch_size, len(all_posts))
                print(f"  Processed {progress}/{len(all_posts)} posts...")

            sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
            for post in all_posts:
                if hasattr(post, 'sentiment'):
                    label = post.sentiment.get('label')
                    if label in sentiment_counts:
                        sentiment_counts[label] += 1

            print("\n✅ Sentiment analysis complete:")
            print(f"  😊 Positive: {sentiment_counts['POSITIVE']}")
            print(f"  😔 Negative: {sentiment_counts['NEGATIVE']}")
            print(f"  😐 Neutral: {sentiment_counts['NEUTRAL']}")

            # Save sentiment results
            sentiment_file = output_dir / f"{output}_sentiment.json"
            posts_with_sentiment = []
            for post in all_posts:
                post_dict = post.to_dict()
                if hasattr(post, 'sentiment'):
                    post_dict['sentiment'] = post.sentiment
                posts_with_sentiment.append(post_dict)
            save_json({'posts': posts_with_sentiment, 'summary': sentiment_counts}, sentiment_file)
            
            # Phase 4: Data Processing
            print("\n📊 Starting Phase 4: Data Processing")
            print("="*50)
            
            # Deduplication
            print("\n🔍 Deduplicating posts...")
            dedup = Deduplicator()
            unique_posts, dedup_stats = dedup.deduplicate(all_posts)
            print(f"  ✓ Removed {dedup_stats['duplicates_removed']} duplicates")
            print(f"  ✓ Unique posts: {len(unique_posts)}")
            
            # Aggregation
            print("\n📈 Aggregating statistics...")
            aggregator = DataAggregator()
            agg_stats = aggregator.aggregate(unique_posts)
            print(f"  ✓ Analyzed {agg_stats['total_posts']} posts")
            print(f"  ✓ Found {agg_stats['authors']['unique_authors']} unique authors")
            
            # Time series analysis
            print("\n⏰ Analyzing time series...")
            time_analyzer = TimeSeriesAnalyzer()
            time_analysis = time_analyzer.analyze(unique_posts)
            print(f"  ✓ Analyzed {len(time_analysis['daily_sentiment'])} days")
            print(f"  ✓ Overall trend: {time_analysis['trends'].get('overall_trend', 'unknown')}")
            
            # Display processing summary
            print("\n📊 Processing Summary:")
            print(f"  • Sentiment ratio: {agg_stats['sentiment_distribution'].get('sentiment_ratio', 0):.3f}")
            print(f"  • Average engagement: {agg_stats['engagement'].get('avg_engagement', 0):.1f}")
            print(f"  • Negative users: {len(agg_stats.get('negative_users', []))}")
            print(f"  • Viral posts: {len(agg_stats.get('viral_posts', []))}")
            print(f"  • Anomalies detected: {len(time_analysis.get('anomalies', []))}")
            
            # Phase 5: Export and Storage
            print("\n💾 Starting Phase 5: Export and Storage")
            print("="*50)
            
            # Create formatter and writer
            formatter = DataFormatter()
            writer = ExportWriter(cfg.output_directory)
            
            # Prepare full context for export
            export_context = {
                'queries': query_list,
                'time_window_days': time,
                'platforms': platform_list,
                'output_prefix': output,
                'formats': format_list,
                'visualize': visualize,
                'version': __version__,
                'start_time': context['start_time'],
                'collection_time': context.get('collection_time', 0),
                'posts': unique_posts,
                'statistics': agg_stats,
                'time_series': time_analysis,
                'deduplication': dedup_stats
            }
            
            # Export based on requested formats
            exported_files = []
            analysis_json_path = None
            user_network_json_path = None
            
            if 'json' in format_list:
                print("\n📄 Exporting JSON...")
                json_data = formatter.format_for_json(export_context)
                filepath = writer.write_json(json_data, output)
                analysis_json_path = filepath
                exported_files.append(filepath)
                print(f"  ✓ Created: {filepath.name}")
            
            if 'csv' in format_list:
                print("\n📊 Exporting CSV files...")
                csv_data = formatter.format_for_csv(export_context)
                filepaths = writer.write_csv(csv_data, output)
                exported_files.extend(filepaths)
                for fp in filepaths:
                    print(f"  ✓ Created: {fp.name}")
            
            if 'html' in format_list:
                print("\n🌐 Generating HTML report...")
                html_data = formatter.format_for_html(export_context)
                filepath = writer.write_html(html_data, output)
                exported_files.append(filepath)
                print(f"  ✓ Created: {filepath.name}")
            
            # Always create summary
            print("\n📋 Creating summary...")
            summary_path = writer.write_summary(export_context, output)
            exported_files.append(summary_path)
            print(f"  ✓ Created: {summary_path.name}")
            
            # Create archive if multiple files
            if len(exported_files) > 3:
                print("\n🗜️  Creating archive...")
                archive_path = writer.create_archive(exported_files, output)
                if archive_path:
                    print(f"  ✓ Created: {archive_path.name}")

            # Phase 6: Visualization
            if visualize:
                print("\n📊 Generating visualizations...")
                from src.visualization import (
                    ChartGenerator,
                    NetworkGraphGenerator,
                    WordCloudGenerator,
                    InteractiveChartGenerator,
                    UserSentimentNetworkAnalyzer,
                    DashboardGenerator,
                )

                charts = ChartGenerator()
                network = NetworkGraphGenerator()
                wordcloud = WordCloudGenerator()
                interactive = InteractiveChartGenerator()
                user_network = UserSentimentNetworkAnalyzer()
                dashboard = DashboardGenerator()

                if export_context.get("time_series", {}).get("daily_sentiment"):
                    charts.sentiment_timeline(
                        export_context["time_series"]["daily_sentiment"],
                        str(output_dir / f"{output}_timeline.png"),
                    )
                    print("  ✓ Sentiment timeline")

                dist_counts = export_context.get("statistics", {}).get("sentiment_distribution", {}).get("counts", {})
                if dist_counts:
                    charts.sentiment_distribution_pie(
                        dist_counts,
                        str(output_dir / f"{output}_sentiment_pie.png"),
                    )
                    print("  ✓ Sentiment distribution")

                network.create_user_network(
                    export_context["posts"],
                    str(output_dir / f"{output}_network.png"),
                )
                print("  ✓ User network graph")

                # Advanced user sentiment network (interactive + JSON)
                user_graph = user_network.create_user_activity_network(
                    export_context["posts"]
                )
                if len(user_graph.nodes()) > 0:
                    json_data = json_graph.node_link_data(user_graph)
                    json_path = writer.write_json(
                        json_data, f"{output}_user_network"
                    )
                    user_network_json_path = json_path
                    exported_files.append(json_path)
                    user_network.create_plotly_network_viz(
                        export_context["posts"],
                        str(output_dir / f"{output}_user_network.html"),
                    )
                    exported_files.append(Path(output_dir / f"{output}_user_network.html"))
                    print("  ✓ User sentiment network")

                wordcloud.create_wordcloud(
                    export_context["posts"],
                    str(output_dir / f"{output}_wordcloud_all.png"),
                )
                wordcloud.create_wordcloud(
                    export_context["posts"],
                    str(output_dir / f"{output}_wordcloud_negative.png"),
                    sentiment_filter="NEGATIVE",
                )
                print("  ✓ Word clouds")

                if export_context.get("time_series", {}).get("daily_sentiment"):
                    interactive.create_interactive_timeline(
                        export_context["time_series"]["daily_sentiment"],
                        str(output_dir / f"{output}_interactive_timeline.html"),
                    )
                    interactive.create_3d_sentiment_scatter(
                        export_context["posts"],
                        str(output_dir / f"{output}_3d_scatter.html"),
                    )
                    print("  ✓ Interactive visualizations")

                dashboard.generate_dashboard(
                    export_context["posts"],
                    str(output_dir / f"{output}_dashboard.html"),
                )
                exported_files.append(Path(output_dir / f"{output}_dashboard.html"))
                print("  ✓ Dashboard")

            print(f"\n✅ Phase 5 complete! Exported {len(exported_files)} files")
            print(f"📁 Output directory: {writer.output_dir.absolute()}")
            
            # Final summary
            print("\n" + "="*50)
            print("🎉 Analysis Complete!")
            print("="*50)
            
            print(f"\n📊 Final Results:")
            print(f"  • Total posts analyzed: {agg_stats['total_posts']:,}")
            print(f"  • Unique authors: {agg_stats['authors']['unique_authors']:,}")
            print(f"  • Date range: {agg_stats['date_range']['days']} days")
            print(f"  • Overall sentiment: ", end='')
            
            sentiment_ratio = agg_stats['sentiment_distribution'].get('sentiment_ratio', 0)
            if sentiment_ratio > 0.1:
                print("Positive 😊")
            elif sentiment_ratio < -0.1:
                print("Negative 😔")
            else:
                print("Neutral 😐")
            
            print(f"\n📁 All results saved to: {writer.output_dir.absolute()}")
            
            # Show file list
            print(f"\n📄 Generated files:")
            for file in exported_files:
                print(f"  • {file.name}")
            
            # Open dashboard or HTML report if generated
            dashboard_file = None
            html_report_file = None
            
            # Look for dashboard first
            for file in exported_files:
                if file.name.endswith('_dashboard.html'):
                    dashboard_file = file
                elif file.suffix == '.html' and 'dashboard' not in file.name:
                    html_report_file = file
            
            # Priority: Dashboard > HTML Report
            target_file = dashboard_file or html_report_file
            
            if target_file:
                file_type = "dashboard" if dashboard_file else "report"
                print(f"\n🌐 View {file_type}: file://{target_file.absolute()}")
                
                if not os.getenv('PYTEST_CURRENT_TEST'):  # Don't prompt during tests
                    if click.confirm(f"\nOpen {file_type} in browser?"):
                        import webbrowser
                        webbrowser.open(f"file://{target_file.absolute()}")

                    if analysis_json_path and click.confirm("Launch interactive dashboard?"):
                        from src.dashboard import launch_dashboard
                        launch_dashboard(analysis_json_path, user_network_json_path)
            
            print("\n✨ Thank you for using MakeSenseOfIt!")
            
        else:
            print("\n⚠️  No posts collected. Check your queries and try again.")
            sys.exit(1)
        
        # Save final context
        context_file = output_dir / f"{output}_context.json"
        context_copy = context.copy()
        context_copy['config'] = cfg.to_dict()
        context_copy['posts'] = len(unique_posts) if 'unique_posts' in locals() else 0
        context_copy['start_time'] = context['start_time'].isoformat()
        save_json(context_copy, context_file)
        
        return context
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        if verbose:
            logger.exception("Full traceback:")
        sys.exit(1)

if __name__ == '__main__':
    main()