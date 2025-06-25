"""
Data aggregation module.
Generates comprehensive statistics from collected posts.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataAggregator:
    """
    Aggregate statistics from posts to provide insights.
    
    Generates statistics on:
    - Sentiment distribution
    - Engagement metrics
    - Author behavior
    - Temporal patterns
    - Content characteristics
    """
    
    def __init__(self, negative_threshold: float = 0.6,
                 min_posts_for_analysis: int = 3):
        """
        Initialize aggregator.
        
        Args:
            negative_threshold: Ratio of negative posts to flag user as negative
            min_posts_for_analysis: Minimum posts required for user analysis
        """
        self.negative_threshold = negative_threshold
        self.min_posts_for_analysis = min_posts_for_analysis
    
    def aggregate(self, posts: List['Post']) -> Dict[str, Any]:
        """
        Generate comprehensive statistics from posts.
        
        Args:
            posts: List of posts to analyze
            
        Returns:
            Dictionary containing all statistics
        """
        if not posts:
            logger.warning("No posts to aggregate")
            return self._empty_stats()
        
        logger.info(f"Aggregating statistics for {len(posts)} posts")
        
        # Convert to DataFrame for easier analysis
        df = self._posts_to_dataframe(posts)
        
        # Generate statistics
        stats = {
            'total_posts': len(df),
            'date_range': self._calculate_date_range(df),
            'by_platform': self._aggregate_by_platform(df),
            'by_query': self._aggregate_by_query(df),
            'sentiment_distribution': self._analyze_sentiment_distribution(df),
            'sentiment_by_platform': self._analyze_sentiment_by_platform(df),
            'sentiment_by_query': self._analyze_sentiment_by_query(df),
            'engagement': self._analyze_engagement(df),
            'authors': self._analyze_authors(df),
            'content': self._analyze_content(df),
            'temporal': self._analyze_temporal_patterns(df),
            'hashtags': self._analyze_hashtags(posts),
            'mentions': self._analyze_mentions(posts),
            'negative_users': self._identify_negative_users(df),
            'viral_posts': self._identify_viral_posts(df, posts),
            'sentiment_leaders': self._identify_sentiment_leaders(df)
        }
        
        logger.info("Aggregation complete")
        return stats
    
    def _empty_stats(self) -> Dict[str, Any]:
        """Return empty statistics structure."""
        return {
            'total_posts': 0,
            'date_range': {},
            'by_platform': {},
            'by_query': {},
            'sentiment_distribution': {},
            'sentiment_by_platform': {},
            'sentiment_by_query': {},
            'engagement': {},
            'authors': {},
            'content': {},
            'temporal': {},
            'hashtags': {},
            'mentions': {},
            'negative_users': [],
            'viral_posts': [],
            'sentiment_leaders': {}
        }
    
    def _posts_to_dataframe(self, posts: List['Post']) -> pd.DataFrame:
        """Convert posts to pandas DataFrame."""
        data = []
        
        for post in posts:
            # Extract sentiment info
            sentiment_label = 'UNKNOWN'
            sentiment_score = 0.0
            sentiment_method = 'unknown'
            
            if hasattr(post, 'sentiment') and post.sentiment:
                sentiment_label = post.sentiment.get('label', 'UNKNOWN')
                sentiment_score = post.sentiment.get('score', 0.0)
                sentiment_method = post.sentiment.get('method', 'unknown')
            
            # Calculate engagement score
            if post.engagement is not None:
                engagement_score = sum(post.engagement.values())
            else:
                engagement_score = 0
            
            # Extract metadata
            hashtag_count = len(post.metadata.get('hashtags', []))
            mention_count = len(post.metadata.get('mentions', []))
            has_media = bool(post.metadata.get('media_types', []))
            
            data.append({
                'id': post.id,
                'platform': post.platform,
                'author': post.author,
                'author_id': post.author_id,
                'timestamp': post.timestamp,
                'query': post.query,
                'sentiment': sentiment_label,
                'sentiment_score': sentiment_score,
                'sentiment_method': sentiment_method,
                'engagement_total': engagement_score,
                'likes': post.engagement.get('likes', post.engagement.get('score', 0)) if post.engagement else 0,
                'shares': post.engagement.get('retweets', post.engagement.get('num_comments', 0)) if post.engagement else 0,
                'content_length': len(post.content),
                'has_title': bool(post.title),
                'hashtag_count': hashtag_count,
                'mention_count': mention_count,
                'has_media': has_media,
                'hour': post.timestamp.hour,
                'day_of_week': post.timestamp.strftime('%A'),
                'date': post.timestamp.date()
            })
        
        return pd.DataFrame(data)
    
    def _calculate_date_range(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate date range statistics."""
        return {
            'start': df['timestamp'].min().isoformat(),
            'end': df['timestamp'].max().isoformat(),
            'days': (df['timestamp'].max() - df['timestamp'].min()).days + 1,
            'total_hours': int((df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600)
        }
    
    def _aggregate_by_platform(self, df: pd.DataFrame) -> Dict[str, int]:
        """Aggregate posts by platform."""
        return df['platform'].value_counts().to_dict()
    
    def _aggregate_by_query(self, df: pd.DataFrame) -> Dict[str, int]:
        """Aggregate posts by query."""
        return df['query'].value_counts().to_dict()
    
    def _analyze_sentiment_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall sentiment distribution."""
        sentiment_counts = df['sentiment'].value_counts().to_dict()
        total = len(df)
        
        # Calculate percentages
        sentiment_percentages = {
            sentiment: (count / total * 100) if total > 0 else 0
            for sentiment, count in sentiment_counts.items()
        }
        
        # Calculate sentiment ratio (positive - negative / total)
        positive = sentiment_counts.get('POSITIVE', 0)
        negative = sentiment_counts.get('NEGATIVE', 0)
        sentiment_ratio = (positive - negative) / total if total > 0 else 0
        
        return {
            'counts': sentiment_counts,
            'percentages': sentiment_percentages,
            'sentiment_ratio': sentiment_ratio,
            'dominant_sentiment': max(sentiment_counts.items(), key=lambda x: x[1])[0] if sentiment_counts else 'UNKNOWN'
        }
    
    def _analyze_sentiment_by_platform(self, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """Analyze sentiment distribution by platform."""
        result = {}
        
        for platform in df['platform'].unique():
            platform_df = df[df['platform'] == platform]
            result[platform] = platform_df['sentiment'].value_counts().to_dict()
        
        return result
    
    def _analyze_sentiment_by_query(self, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """Analyze sentiment distribution by query."""
        result = {}
        
        for query in df['query'].unique():
            query_df = df[df['query'] == query]
            result[query] = query_df['sentiment'].value_counts().to_dict()
        
        return result
    
    def _analyze_engagement(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze engagement metrics."""
        return {
            'total_likes': int(df['likes'].sum()),
            'total_shares': int(df['shares'].sum()),
            'total_engagement': int(df['engagement_total'].sum()),
            'avg_likes': float(df['likes'].mean()),
            'avg_shares': float(df['shares'].mean()),
            'avg_engagement': float(df['engagement_total'].mean()),
            'median_engagement': float(df['engagement_total'].median()),
            'max_likes': int(df['likes'].max()),
            'max_shares': int(df['shares'].max()),
            'max_engagement': int(df['engagement_total'].max()),
            'engagement_std': float(df['engagement_total'].std()),
            'high_engagement_posts': int((df['engagement_total'] > df['engagement_total'].mean() + df['engagement_total'].std()).sum()),
            'zero_engagement_posts': int((df['engagement_total'] == 0).sum())
        }
    
    def _analyze_authors(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze author statistics."""
        author_counts = df['author'].value_counts()
        
        return {
            'unique_authors': df['author'].nunique(),
            'most_active': author_counts.head(20).to_dict(),
            'posts_per_author': {
                'mean': float(author_counts.mean()),
                'median': float(author_counts.median()),
                'max': int(author_counts.max()),
                'std': float(author_counts.std())
            },
            'single_post_authors': int((author_counts == 1).sum()),
            'prolific_authors': int((author_counts >= 10).sum()),
            'author_concentration': float(author_counts.head(10).sum() / len(df)) if len(df) > 0 else 0
        }
    
    def _analyze_content(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze content characteristics."""
        return {
            'avg_length': float(df['content_length'].mean()),
            'median_length': float(df['content_length'].median()),
            'max_length': int(df['content_length'].max()),
            'min_length': int(df['content_length'].min()),
            'with_hashtags': int((df['hashtag_count'] > 0).sum()),
            'with_mentions': int((df['mention_count'] > 0).sum()),
            'with_media': int(df['has_media'].sum()),
            'with_title': int(df['has_title'].sum()),
            'hashtag_usage_rate': float((df['hashtag_count'] > 0).mean()),
            'mention_usage_rate': float((df['mention_count'] > 0).mean()),
            'avg_hashtags_per_post': float(df['hashtag_count'].mean()),
            'avg_mentions_per_post': float(df['mention_count'].mean())
        }
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in posts."""
        # Posts by hour
        posts_by_hour = df['hour'].value_counts().sort_index().to_dict()
        
        # Posts by day of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        posts_by_day = df['day_of_week'].value_counts().reindex(day_order, fill_value=0).to_dict()
        
        # Posts by date
        posts_by_date = df.groupby('date').size().to_dict()
        posts_by_date = {str(k): v for k, v in posts_by_date.items()}  # Convert date to string
        
        # Peak times
        peak_hour = int(df['hour'].mode()[0]) if not df.empty else None
        peak_day = df['day_of_week'].mode()[0] if not df.empty else None
        
        # Activity patterns
        hourly_avg = df.groupby('hour').size().mean()
        high_activity_hours = df['hour'].value_counts()[df['hour'].value_counts() > hourly_avg].index.tolist()
        
        return {
            'posts_by_hour': posts_by_hour,
            'posts_by_day_of_week': posts_by_day,
            'posts_by_date': posts_by_date,
            'peak_hour': peak_hour,
            'peak_day': peak_day,
            'high_activity_hours': high_activity_hours,
            'weekend_posts': int(df[df['day_of_week'].isin(['Saturday', 'Sunday'])].shape[0]),
            'weekday_posts': int(df[~df['day_of_week'].isin(['Saturday', 'Sunday'])].shape[0]),
            'night_posts': int(df[df['hour'].between(0, 6)].shape[0]),
            'day_posts': int(df[df['hour'].between(7, 18)].shape[0]),
            'evening_posts': int(df[df['hour'].between(19, 23)].shape[0])
        }
    
    def _analyze_hashtags(self, posts: List['Post']) -> Dict[str, Any]:
        """Analyze hashtag usage."""
        all_hashtags = []
        
        for post in posts:
            hashtags = post.metadata.get('hashtags', [])
            all_hashtags.extend([h.lower() for h in hashtags])
        
        if not all_hashtags:
            return {'top_hashtags': {}, 'unique_hashtags': 0, 'total_hashtags': 0}
        
        hashtag_counts = Counter(all_hashtags)
        
        return {
            'top_hashtags': dict(hashtag_counts.most_common(20)),
            'unique_hashtags': len(hashtag_counts),
            'total_hashtags': len(all_hashtags),
            'hashtag_frequency': {
                'single_use': sum(1 for count in hashtag_counts.values() if count == 1),
                'trending': sum(1 for count in hashtag_counts.values() if count >= 10)
            }
        }
    
    def _analyze_mentions(self, posts: List['Post']) -> Dict[str, Any]:
        """Analyze user mentions."""
        all_mentions = []
        
        for post in posts:
            mentions = post.metadata.get('mentions', [])
            all_mentions.extend([m.lower() for m in mentions])
        
        if not all_mentions:
            return {'top_mentions': {}, 'unique_mentions': 0, 'total_mentions': 0}
        
        mention_counts = Counter(all_mentions)
        
        return {
            'top_mentions': dict(mention_counts.most_common(20)),
            'unique_mentions': len(mention_counts),
            'total_mentions': len(all_mentions),
            'mention_frequency': {
                'single_mention': sum(1 for count in mention_counts.values() if count == 1),
                'frequently_mentioned': sum(1 for count in mention_counts.values() if count >= 5)
            }
        }
    
    def _identify_negative_users(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify consistently negative users."""
        # Group by author
        author_sentiment = df.groupby('author').agg({
            'sentiment': lambda x: x.value_counts().to_dict(),
            'id': 'count',
            'sentiment_score': 'mean',
            'platform': 'first',
            'engagement_total': 'mean'
        }).rename(columns={'id': 'post_count'})
        
        # Filter authors with minimum posts
        author_sentiment = author_sentiment[author_sentiment['post_count'] >= self.min_posts_for_analysis]
        
        negative_users = []
        
        for author, row in author_sentiment.iterrows():
            sentiments = row['sentiment']
            negative_count = sentiments.get('NEGATIVE', 0)
            total_count = sum(sentiments.values())
            
            if total_count > 0:
                negative_ratio = negative_count / total_count
                
                if negative_ratio >= self.negative_threshold:
                    negative_users.append({
                        'author': author,
                        'platform': row['platform'],
                        'post_count': int(row['post_count']),
                        'negative_ratio': float(negative_ratio),
                        'negative_posts': int(negative_count),
                        'positive_posts': int(sentiments.get('POSITIVE', 0)),
                        'neutral_posts': int(sentiments.get('NEUTRAL', 0)),
                        'avg_sentiment_score': float(row['sentiment_score']),
                        'avg_engagement': float(row['engagement_total']),
                        'sentiment_breakdown': sentiments
                    })
        
        # Sort by negativity ratio
        negative_users.sort(key=lambda x: x['negative_ratio'], reverse=True)
        
        return negative_users[:50]  # Top 50 negative users
    
    def _identify_viral_posts(self, df: pd.DataFrame, posts: List['Post']) -> List[Dict[str, Any]]:
        """Identify viral or highly engaging posts."""
        if len(df) == 0:
            return []
            
        # Use percentile-based approach for better outlier detection
        # Viral threshold is either 95th percentile or mean + 2*std, whichever is lower
        percentile_95 = df['engagement_total'].quantile(0.95)
        mean_plus_2std = df['engagement_total'].mean() + (2 * df['engagement_total'].std())
        
        # For small datasets, use a more lenient approach
        if len(df) < 10:
            # Use 90th percentile for small datasets
            viral_threshold = df['engagement_total'].quantile(0.90)
        else:
            viral_threshold = min(percentile_95, mean_plus_2std)
        
        # Ensure threshold is at least 2x the median to avoid false positives
        median_engagement = df['engagement_total'].median()
        viral_threshold = max(viral_threshold, median_engagement * 2)
        
        # Find viral posts
        viral_df = df[df['engagement_total'] >= viral_threshold].copy()
        viral_df = viral_df.sort_values('engagement_total', ascending=False).head(20)
        
        viral_posts = []
        for idx, row in viral_df.iterrows():
            # Find the original post
            original_post = next((p for p in posts if p.id == row['id'] and p.platform == row['platform']), None)
            
            if original_post:
                viral_posts.append({
                    'id': row['id'],
                    'platform': row['platform'],
                    'author': row['author'],
                    'engagement': int(row['engagement_total']),
                    'likes': int(row['likes']),
                    'shares': int(row['shares']),
                    'sentiment': row['sentiment'],
                    'content_preview': original_post.content[:200] + '...' if len(original_post.content) > 200 else original_post.content,
                    'timestamp': row['timestamp'].isoformat(),
                    'url': original_post.url
                })
        
        return viral_posts
    
    def _identify_sentiment_leaders(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Identify sentiment leaders (influential positive/negative voices)."""
        # Authors with high engagement
        author_stats = df.groupby('author').agg({
            'engagement_total': ['sum', 'mean'],
            'sentiment': lambda x: x.mode()[0] if not x.empty else 'UNKNOWN',
            'id': 'count'
        })
        
        author_stats.columns = ['total_engagement', 'avg_engagement', 'dominant_sentiment', 'post_count']
        
        # Filter for authors with multiple posts
        author_stats = author_stats[author_stats['post_count'] >= 2]
        
        # Identify leaders by sentiment
        sentiment_leaders = {
            'positive': [],
            'negative': [],
            'neutral': []
        }
        
        for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            sentiment_authors = author_stats[author_stats['dominant_sentiment'] == sentiment]
            top_authors = sentiment_authors.nlargest(10, 'total_engagement')
            
            for author, row in top_authors.iterrows():
                sentiment_leaders[sentiment.lower()].append({
                    'author': author,
                    'total_engagement': int(row['total_engagement']),
                    'avg_engagement': float(row['avg_engagement']),
                    'post_count': int(row['post_count']),
                    'influence_score': float(row['total_engagement'] / df['engagement_total'].sum()) if df['engagement_total'].sum() > 0 else 0
                })
        
        return sentiment_leaders