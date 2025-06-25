"""
Data formatting module for exports.
Prepares data for different output formats (JSON, CSV, HTML).
"""
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataFormatter:
    """Format data for different export types."""
    
    def __init__(self):
        """Initialize data formatter."""
        self.timestamp = datetime.now()
        self.timestamp_str = self.timestamp.strftime('%Y-%m-%d_%H-%M-%S')
        logger.debug(f"DataFormatter initialized at {self.timestamp_str}")
    
    def format_for_json(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format data for JSON export.
        
        Args:
            context: Analysis context with all data
            
        Returns:
            Dictionary ready for JSON serialization
        """
        logger.info("Formatting data for JSON export")
        
        # Extract posts if they exist
        posts_data = []
        if 'posts' in context and isinstance(context['posts'], list):
            for post in context['posts']:
                post_dict = post.to_dict() if hasattr(post, 'to_dict') else {}
                
                # Add sentiment if available
                if hasattr(post, 'sentiment') and post.sentiment:
                    post_dict['sentiment'] = post.sentiment
                    
                posts_data.append(post_dict)
        
        # Build comprehensive JSON structure
        json_data = {
            'metadata': {
                'generated_at': self.timestamp.isoformat(),
                'generated_timestamp': self.timestamp_str,
                'version': context.get('version', '1.0.0'),
                'analysis_parameters': {
                    'queries': context.get('queries', []),
                    'time_window_days': context.get('time_window_days', 0),
                    'platforms': context.get('platforms', []),
                    'start_time': context.get('start_time').isoformat() if context.get('start_time') else None,
                    'collection_time': context.get('collection_time', 0)
                }
            },
            'summary': {
                'total_posts': len(posts_data),
                'date_range': context.get('statistics', {}).get('date_range', {}),
                'platforms': context.get('statistics', {}).get('by_platform', {}),
                'queries': context.get('statistics', {}).get('by_query', {})
            },
            'statistics': context.get('statistics', {}),
            'time_series': context.get('time_series', {}),
            'deduplication': context.get('deduplication', {}),
            'posts': posts_data
        }
        
        logger.info(f"Formatted JSON with {len(posts_data)} posts")
        return json_data
    
    def format_for_csv(self, context: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Format data for CSV export (multiple files).
        
        Args:
            context: Analysis context with all data
            
        Returns:
            Dictionary mapping filename suffixes to DataFrames
        """
        logger.info("Formatting data for CSV export")
        dataframes = {}
        
        # 1. Posts DataFrame
        posts_data = []
        posts = context.get('posts', [])
        
        for post in posts:
            # Extract basic fields
            post_data = {
                'id': post.id,
                'platform': post.platform,
                'author': post.author,
                'author_id': post.author_id,
                'timestamp': post.timestamp.isoformat() if hasattr(post.timestamp, 'isoformat') else str(post.timestamp),
                'date': post.timestamp.date() if hasattr(post.timestamp, 'date') else post.timestamp,
                'hour': post.timestamp.hour if hasattr(post.timestamp, 'hour') else 0,
                'query': post.query,
                'url': post.url,
                'content': self._truncate_text(post.content, 500),
                'content_length': len(post.content),
                'title': self._truncate_text(post.title, 200) if post.title else ''
            }
            
            # Add sentiment if available
            if hasattr(post, 'sentiment') and post.sentiment:
                post_data.update({
                    'sentiment': post.sentiment.get('label', ''),
                    'sentiment_score': post.sentiment.get('score', 0),
                    'sentiment_method': post.sentiment.get('method', '')
                })
            else:
                post_data.update({
                    'sentiment': '',
                    'sentiment_score': 0,
                    'sentiment_method': ''
                })
            
            # Add engagement metrics
            if post.engagement:
                post_data['likes'] = post.engagement.get('likes', post.engagement.get('score', 0))
                post_data['shares'] = post.engagement.get('retweets', post.engagement.get('num_comments', 0))
                post_data['replies'] = post.engagement.get('replies', 0)
                post_data['engagement_total'] = sum(post.engagement.values())
            else:
                post_data.update({
                    'likes': 0,
                    'shares': 0,
                    'replies': 0,
                    'engagement_total': 0
                })
            
            # Add metadata
            metadata = post.metadata or {}
            post_data['hashtag_count'] = len(metadata.get('hashtags', []))
            post_data['mention_count'] = len(metadata.get('mentions', []))
            post_data['has_media'] = bool(metadata.get('media_types', []))
            
            posts_data.append(post_data)
        
        # Always return a posts DataFrame even if empty so downstream
        # processing can rely on its existence. This mirrors the behaviour
        # expected by the tests for empty contexts.
        dataframes['posts'] = pd.DataFrame(posts_data)
        if posts_data:
            logger.debug(f"Created posts DataFrame with {len(posts_data)} rows")
        else:
            logger.debug("Created empty posts DataFrame")
        
        # 2. Daily Statistics DataFrame
        stats = context.get('statistics', {})
        time_series = context.get('time_series', {})
        
        if time_series.get('daily_sentiment'):
            daily_data = []
            for date_str, data in time_series['daily_sentiment'].items():
                daily_data.append({
                    'date': date_str,
                    'total_posts': data.get('total', 0),
                    'positive': data.get('positive', 0),
                    'negative': data.get('negative', 0),
                    'neutral': data.get('neutral', 0),
                    'sentiment_ratio': data.get('sentiment_ratio', 0),
                    'positive_ratio': data.get('positive_ratio', 0),
                    'avg_sentiment_score': data.get('avg_sentiment', 0),
                    'total_engagement': data.get('total_engagement', 0),
                    'avg_engagement': data.get('avg_engagement', 0),
                    'unique_authors': data.get('unique_authors', 0)
                })
            
            if daily_data:
                dataframes['daily_statistics'] = pd.DataFrame(daily_data)
                logger.debug(f"Created daily statistics DataFrame with {len(daily_data)} rows")
        
        # 3. Summary Statistics DataFrame
        if stats:
            summary_data = {
                'metric': [],
                'value': [],
                'category': []
            }
            
            # Basic metrics
            summary_data['metric'].append('total_posts')
            summary_data['value'].append(stats.get('total_posts', 0))
            summary_data['category'].append('general')
            
            summary_data['metric'].append('unique_authors')
            summary_data['value'].append(stats.get('authors', {}).get('unique_authors', 0))
            summary_data['category'].append('general')
            
            summary_data['metric'].append('date_range_days')
            summary_data['value'].append(stats.get('date_range', {}).get('days', 0))
            summary_data['category'].append('general')
            
            # Sentiment metrics
            sent_dist = stats.get('sentiment_distribution', {})
            if sent_dist:
                for sentiment, count in sent_dist.get('counts', {}).items():
                    summary_data['metric'].append(f'sentiment_{sentiment.lower()}')
                    summary_data['value'].append(count)
                    summary_data['category'].append('sentiment')
                
                summary_data['metric'].append('sentiment_ratio')
                summary_data['value'].append(sent_dist.get('sentiment_ratio', 0))
                summary_data['category'].append('sentiment')
            
            # Engagement metrics
            engagement = stats.get('engagement', {})
            for metric in ['total_likes', 'total_shares', 'avg_engagement', 'max_engagement']:
                if metric in engagement:
                    summary_data['metric'].append(metric)
                    summary_data['value'].append(engagement[metric])
                    summary_data['category'].append('engagement')
            
            if summary_data['metric']:
                dataframes['summary_statistics'] = pd.DataFrame(summary_data)
                logger.debug(f"Created summary statistics DataFrame with {len(summary_data['metric'])} metrics")
        
        # 4. Negative Users DataFrame
        negative_users = stats.get('negative_users', [])
        if negative_users:
            neg_users_data = []
            for user in negative_users:
                neg_users_data.append({
                    'author': user['author'],
                    'platform': user['platform'],
                    'post_count': user['post_count'],
                    'negative_ratio': user['negative_ratio'],
                    'negative_posts': user['negative_posts'],
                    'positive_posts': user['positive_posts'],
                    'neutral_posts': user['neutral_posts'],
                    'avg_sentiment_score': user['avg_sentiment_score'],
                    'avg_engagement': user['avg_engagement']
                })
            
            dataframes['negative_users'] = pd.DataFrame(neg_users_data)
            logger.debug(f"Created negative users DataFrame with {len(neg_users_data)} users")
        
        # 5. Viral Posts DataFrame
        viral_posts = stats.get('viral_posts', [])
        if viral_posts:
            viral_data = []
            for vpost in viral_posts:
                viral_data.append({
                    'id': vpost['id'],
                    'platform': vpost['platform'],
                    'author': vpost['author'],
                    'engagement': vpost['engagement'],
                    'likes': vpost['likes'],
                    'shares': vpost['shares'],
                    'sentiment': vpost['sentiment'],
                    'content_preview': vpost['content_preview'],
                    'timestamp': vpost['timestamp'],
                    'url': vpost['url']
                })
            
            dataframes['viral_posts'] = pd.DataFrame(viral_data)
            logger.debug(f"Created viral posts DataFrame with {len(viral_data)} posts")
        
        # 6. Hashtags DataFrame
        hashtags = stats.get('hashtags', {}).get('top_hashtags', {})
        if hashtags:
            hashtag_data = [
                {'hashtag': tag, 'count': count}
                for tag, count in hashtags.items()
            ]
            dataframes['hashtags'] = pd.DataFrame(hashtag_data)
            logger.debug(f"Created hashtags DataFrame with {len(hashtag_data)} tags")
        
        # 7. Anomalies DataFrame
        anomalies = time_series.get('anomalies', [])
        if anomalies:
            anomaly_data = []
            for anomaly in anomalies:
                anomaly_data.append({
                    'date': anomaly['date'],
                    'type': anomaly['type'],
                    'z_score': anomaly['z_score'],
                    'severity': anomaly['severity'],
                    'post_count': anomaly['post_count'],
                    'sentiment_value': anomaly['sentiment_value']
                })
            
            dataframes['anomalies'] = pd.DataFrame(anomaly_data)
            logger.debug(f"Created anomalies DataFrame with {len(anomaly_data)} anomalies")
        
        logger.info(f"Created {len(dataframes)} DataFrames for CSV export")
        return dataframes
    
    def format_for_html(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format data for HTML report.
        
        Args:
            context: Analysis context with all data
            
        Returns:
            Dictionary with data prepared for HTML template
        """
        logger.info("Formatting data for HTML export")
        
        stats = context.get('statistics', {})
        time_series = context.get('time_series', {})
        
        # Prepare chart data
        chart_data = self._prepare_chart_data(stats, time_series)
        
        # Format numbers for display
        def format_num(n):
            """Format number with thousands separator."""
            if isinstance(n, (int, float)):
                return f"{n:,}" if n == int(n) else f"{n:,.2f}"
            return str(n)
        
        # Build HTML context
        html_data = {
            'title': f"Sentiment Analysis Report - {self.timestamp_str}",
            'generated_at': self.timestamp.strftime('%B %d, %Y at %I:%M %p'),
            'timestamp': self.timestamp_str,
            
            # Analysis parameters
            'queries': context.get('queries', []),
            'time_window': context.get('time_window_days', 0),
            'platforms': context.get('platforms', []),
            
            # Summary statistics
            'total_posts': format_num(stats.get('total_posts', 0)),
            'unique_authors': format_num(stats.get('authors', {}).get('unique_authors', 0)),
            'date_range': stats.get('date_range', {}),
            'date_range_formatted': self._format_date_range(stats.get('date_range', {})),
            
            # Sentiment distribution
            'sentiment_distribution': stats.get('sentiment_distribution', {}),
            'sentiment_counts': {
                k: format_num(v) for k, v in 
                stats.get('sentiment_distribution', {}).get('counts', {}).items()
            },
            'sentiment_percentages': stats.get('sentiment_distribution', {}).get('percentages', {}),
            'sentiment_ratio': stats.get('sentiment_distribution', {}).get('sentiment_ratio', 0),
            
            # Engagement statistics
            'engagement_stats': {
                k: format_num(v) for k, v in 
                stats.get('engagement', {}).items()
            },
            
            # Top negative users (limited to 20)
            'top_negative_users': stats.get('negative_users', [])[:20],
            'negative_users_count': len(stats.get('negative_users', [])),
            
            # Viral posts (limited to 10)
            'viral_posts': stats.get('viral_posts', [])[:10],
            'viral_posts_count': len(stats.get('viral_posts', [])),
            
            # Time series data
            'time_series_data': time_series.get('daily_sentiment', {}),
            'trends': time_series.get('trends', {}),
            'anomalies': time_series.get('anomalies', [])[:10],
            'volatility': time_series.get('sentiment_volatility', {}),
            
            # Chart data for JavaScript
            'chart_data': chart_data,
            
            # Platform breakdown
            'platform_stats': stats.get('by_platform', {}),
            'platform_sentiment': stats.get('sentiment_by_platform', {}),
            
            # Query breakdown
            'query_stats': stats.get('by_query', {}),
            'query_sentiment': stats.get('sentiment_by_query', {}),
            
            # Temporal patterns
            'temporal_patterns': stats.get('temporal', {}),
            
            # Top hashtags
            'top_hashtags': list(stats.get('hashtags', {}).get('top_hashtags', {}).items())[:20],
            
            # Authors analysis
            'author_stats': stats.get('authors', {}),
            'most_active_authors': list(stats.get('authors', {}).get('most_active', {}).items())[:10],
            
            # Processing metadata
            'deduplication_stats': context.get('deduplication', {}),
            'processing_time': context.get('collection_time', 0),
            
            # Helper functions for template
            'format_number': format_num,
            'format_percentage': lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else str(x)
        }
        
        logger.info("Formatted data for HTML export")
        return html_data
    
    def _truncate_text(self, text: Optional[str], max_length: int = 100) -> str:
        """
        Truncate text to maximum length.
        
        Args:
            text: Text to truncate
            max_length: Maximum length
            
        Returns:
            Truncated text
        """
        if not text:
            return ""
        
        text = str(text).strip()
        if len(text) <= max_length:
            return text
        
        return text[:max_length - 3] + "..."
    
    def _format_date_range(self, date_range: Dict[str, Any]) -> str:
        """Format date range for display."""
        if not date_range:
            return "No date range"
        
        try:
            start = datetime.fromisoformat(date_range.get('start', ''))
            end = datetime.fromisoformat(date_range.get('end', ''))
            
            # Same day
            if start.date() == end.date():
                return start.strftime('%B %d, %Y')
            # Same month
            elif start.month == end.month and start.year == end.year:
                return f"{start.strftime('%B %d')} - {end.strftime('%d, %Y')}"
            # Same year
            elif start.year == end.year:
                return f"{start.strftime('%B %d')} - {end.strftime('%B %d, %Y')}"
            # Different years
            else:
                return f"{start.strftime('%B %d, %Y')} - {end.strftime('%B %d, %Y')}"
        except:
            return f"{date_range.get('days', 0)} days"
    
    def _prepare_chart_data(self, stats: Dict[str, Any], time_series: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for JavaScript charts."""
        chart_data = {}
        
        # 1. Sentiment pie chart data
        sent_dist = stats.get('sentiment_distribution', {}).get('counts', {})
        if sent_dist:
            chart_data['sentiment_pie'] = {
                'labels': list(sent_dist.keys()),
                'data': list(sent_dist.values()),
                'colors': {
                    'POSITIVE': '#4CAF50',
                    'NEGATIVE': '#f44336',
                    'NEUTRAL': '#9E9E9E'
                }
            }
        
        # 2. Time series line chart data
        daily_sentiment = time_series.get('daily_sentiment', {})
        if daily_sentiment:
            dates = sorted(daily_sentiment.keys())
            
            chart_data['sentiment_timeline'] = {
                'dates': dates,
                'positive': [daily_sentiment[d]['positive'] for d in dates],
                'negative': [daily_sentiment[d]['negative'] for d in dates],
                'neutral': [daily_sentiment[d]['neutral'] for d in dates],
                'engagement': [daily_sentiment[d]['total_engagement'] for d in dates],
                'sentiment_ratio': [daily_sentiment[d]['sentiment_ratio'] for d in dates]
            }
        
        # 3. Hourly activity heatmap data
        temporal = stats.get('temporal', {})
        if temporal.get('posts_by_hour'):
            chart_data['hourly_activity'] = {
                'hours': list(range(24)),
                'counts': [temporal['posts_by_hour'].get(h, 0) for h in range(24)]
            }
        
        # 4. Platform comparison data
        platform_stats = stats.get('by_platform', {})
        if platform_stats:
            chart_data['platform_comparison'] = {
                'platforms': list(platform_stats.keys()),
                'counts': list(platform_stats.values())
            }
        
        # 5. Top hashtags bar chart
        hashtags = stats.get('hashtags', {}).get('top_hashtags', {})
        if hashtags:
            top_10 = list(hashtags.items())[:10]
            chart_data['top_hashtags'] = {
                'hashtags': [h[0] for h in top_10],
                'counts': [h[1] for h in top_10]
            }
        
        return chart_data