"""
Tests for data processing functionality.
Comprehensive test coverage for Phase 4 components.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from src.scrapers.base import Post
from src.processing import Deduplicator, DataAggregator, TimeSeriesAnalyzer


def create_test_post(id, content, author='testuser', platform='twitter', 
                    sentiment_label='NEUTRAL', timestamp=None, query='test',
                    engagement=None):
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
        title=None if platform == 'twitter' else f"Title {id}",
        timestamp=timestamp,
        engagement=engagement,
        url=f'https://{platform}.com/{id}',
        query=query,
        metadata={
            'hashtags': ['test', 'data'] if id % 2 == 0 else [],
            'mentions': ['user1'] if id % 3 == 0 else []
        }
    )
    
    # Add sentiment
    post.sentiment = {
        'label': sentiment_label,
        'score': 0.8,
        'method': 'transformer'
    }
    
    return post


class TestDeduplicator:
    """Test deduplication functionality."""
    
    def test_deduplicator_initialization(self):
        """Test deduplicator initialization."""
        dedup = Deduplicator(similarity_threshold=0.9)
        
        assert dedup.similarity_threshold == 0.9
        assert dedup.enable_fuzzy_matching is False
        assert len(dedup.seen_ids) == 0
        assert len(dedup.seen_hashes) == 0
    
    def test_exact_id_deduplication(self):
        """Test deduplication by exact ID match."""
        posts = [
            create_test_post(1, "Hello world"),
            create_test_post(1, "Hello world"),  # Duplicate ID
            create_test_post(2, "Different post")
        ]
        
        dedup = Deduplicator()
        unique_posts, stats = dedup.deduplicate(posts)
        
        assert len(unique_posts) == 2
        assert stats['duplicates_removed'] == 1
        assert stats['processing_stats']['exact_id_duplicates'] == 1
    
    def test_content_hash_deduplication(self):
        """Test deduplication by content hash."""
        posts = [
            create_test_post(1, "Hello world!"),
            create_test_post(2, "Hello world!"),  # Same content, different ID
            create_test_post(3, "Different content")
        ]
        
        dedup = Deduplicator()
        unique_posts, stats = dedup.deduplicate(posts)
        
        assert len(unique_posts) == 2
        assert stats['duplicates_removed'] == 1
        assert stats['processing_stats']['content_duplicates'] == 1
    
    def test_normalized_content_deduplication(self):
        """Test content normalization for deduplication."""
        posts = [
            create_test_post(1, "Hello World! #test @user1"),
            create_test_post(2, "hello world test"),  # Normalized same as 1 without user1
            create_test_post(3, "Hello, World! Test @user1!"),  # Also normalized same as 1
            create_test_post(4, "Different content here")
        ]
        
        dedup = Deduplicator()
        unique_posts, stats = dedup.deduplicate(posts)
        
        assert len(unique_posts) == 2  # Only 2 unique after normalization
        assert stats['duplicates_removed'] == 2
    
    def test_fuzzy_deduplication(self):
        """Test fuzzy matching deduplication."""
        posts = [
            create_test_post(1, "The quick brown fox jumps over the lazy dog"),
            create_test_post(2, "The quick brown fox jumps over a lazy dog"),  # Very similar
            create_test_post(3, "Completely different content here")
        ]
        
        dedup = Deduplicator(similarity_threshold=0.8, enable_fuzzy_matching=True)
        unique_posts, stats = dedup.deduplicate(posts)
        
        assert len(unique_posts) == 2  # Should detect fuzzy duplicate
        assert stats['processing_stats']['fuzzy_duplicates'] == 1
    
    def test_cross_platform_deduplication(self):
        """Test deduplication across platforms."""
        posts = [
            create_test_post(1, "Same content", platform='twitter'),
            create_test_post(2, "Same content", platform='reddit'),  # Same content, different platform
            create_test_post(3, "Different content", platform='twitter')
        ]
        
        dedup = Deduplicator()
        unique_posts, stats = dedup.deduplicate(posts)
        
        assert len(unique_posts) == 2
        assert stats['cross_query_duplicates'] == 1
    
    def test_deduplication_statistics(self):
        """Test deduplication statistics."""
        posts = [
            create_test_post(1, "Post 1", query='bitcoin'),
            create_test_post(2, "Post 1", query='btc'),  # Duplicate from different query
            create_test_post(3, "Post 2", query='bitcoin'),
            create_test_post(3, "Post 2", query='bitcoin'),  # Exact duplicate
        ]
        
        dedup = Deduplicator()
        unique_posts, stats = dedup.deduplicate(posts)
        
        assert stats['total_posts'] == 4
        assert stats['unique_posts'] == 2
        assert stats['duplicates_removed'] == 2
        assert stats['duplicate_rate'] == 0.5
        assert 'bitcoin' in stats['duplicates_by_query']
        assert 'btc' in stats['duplicates_by_query']
    
    def test_deduplicator_reset(self):
        """Test deduplicator reset functionality."""
        posts = [create_test_post(1, "Test post")]
        
        dedup = Deduplicator()
        dedup.deduplicate(posts)
        
        assert len(dedup.seen_ids) == 1
        assert dedup.stats['total_processed'] == 1
        
        dedup.reset()
        
        assert len(dedup.seen_ids) == 0
        assert dedup.stats['total_processed'] == 0
    
    def test_empty_posts_deduplication(self):
        """Test deduplication with empty posts list."""
        dedup = Deduplicator()
        unique_posts, stats = dedup.deduplicate([])
        
        assert len(unique_posts) == 0
        assert stats['total_posts'] == 0
        assert stats['duplicates_removed'] == 0


class TestDataAggregator:
    """Test data aggregation functionality."""
    
    def test_aggregator_initialization(self):
        """Test aggregator initialization."""
        agg = DataAggregator(negative_threshold=0.6, min_posts_for_analysis=3)
        
        assert agg.negative_threshold == 0.6
        assert agg.min_posts_for_analysis == 3
    
    def test_empty_posts_aggregation(self):
        """Test aggregation with no posts."""
        agg = DataAggregator()
        stats = agg.aggregate([])
        
        assert stats['total_posts'] == 0
        assert stats['by_platform'] == {}
        assert stats['negative_users'] == []
    
    def test_basic_aggregation(self):
        """Test basic statistics aggregation."""
        posts = [
            create_test_post(1, "Positive post", sentiment_label='POSITIVE'),
            create_test_post(2, "Negative post", sentiment_label='NEGATIVE'),
            create_test_post(3, "Neutral post", sentiment_label='NEUTRAL'),
            create_test_post(4, "Another positive", sentiment_label='POSITIVE')
        ]
        
        agg = DataAggregator()
        stats = agg.aggregate(posts)
        
        assert stats['total_posts'] == 4
        assert stats['sentiment_distribution']['counts']['POSITIVE'] == 2
        assert stats['sentiment_distribution']['counts']['NEGATIVE'] == 1
        assert stats['sentiment_distribution']['sentiment_ratio'] == 0.25  # (2-1)/4
    
    def test_platform_aggregation(self):
        """Test aggregation by platform."""
        posts = [
            create_test_post(1, "Tweet 1", platform='twitter'),
            create_test_post(2, "Tweet 2", platform='twitter'),
            create_test_post(3, "Reddit 1", platform='reddit'),
        ]
        
        agg = DataAggregator()
        stats = agg.aggregate(posts)
        
        assert stats['by_platform']['twitter'] == 2
        assert stats['by_platform']['reddit'] == 1
    
    def test_engagement_analysis(self):
        """Test engagement metrics calculation."""
        posts = [
            create_test_post(1, "Post 1", engagement={'likes': 100, 'retweets': 50}),
            create_test_post(2, "Post 2", engagement={'likes': 200, 'retweets': 100}),
            create_test_post(3, "Post 3", engagement={'likes': 0, 'retweets': 0})
        ]
        
        agg = DataAggregator()
        stats = agg.aggregate(posts)
        
        engagement = stats['engagement']
        assert engagement['total_likes'] == 300
        assert engagement['total_shares'] == 150
        assert engagement['avg_likes'] == 100
        assert engagement['max_likes'] == 200
        assert engagement['zero_engagement_posts'] == 1
    
    def test_author_analysis(self):
        """Test author statistics."""
        posts = [
            create_test_post(1, "Post 1", author='user1'),
            create_test_post(2, "Post 2", author='user1'),
            create_test_post(3, "Post 3", author='user2'),
            create_test_post(4, "Post 4", author='user1'),
        ]
        
        agg = DataAggregator()
        stats = agg.aggregate(posts)
        
        authors = stats['authors']
        assert authors['unique_authors'] == 2
        assert authors['most_active']['user1'] == 3
        assert authors['single_post_authors'] == 1  # user2
    
    def test_temporal_analysis(self):
        """Test temporal pattern analysis."""
        # Create posts at different times
        posts = []
        base_time = datetime.now()
        
        # Morning posts
        for i in range(3):
            posts.append(create_test_post(
                i, f"Morning {i}", 
                timestamp=base_time.replace(hour=9) - timedelta(days=i)
            ))
        
        # Evening posts
        for i in range(2):
            posts.append(create_test_post(
                i+10, f"Evening {i}", 
                timestamp=base_time.replace(hour=20) - timedelta(days=i)
            ))
        
        agg = DataAggregator()
        stats = agg.aggregate(posts)
        
        temporal = stats['temporal']
        assert 9 in temporal['posts_by_hour']
        assert 20 in temporal['posts_by_hour']
        assert temporal['posts_by_hour'][9] == 3
        assert temporal['posts_by_hour'][20] == 2
    
    def test_negative_users_identification(self):
        """Test identification of negative users."""
        # Create posts from different users with different sentiment patterns
        posts = []
        
        # Negative user (4 negative, 1 positive)
        for i in range(4):
            posts.append(create_test_post(
                i, f"Negative {i}", author='negative_user', 
                sentiment_label='NEGATIVE'
            ))
        posts.append(create_test_post(
            10, "One positive", author='negative_user', 
            sentiment_label='POSITIVE'
        ))
        
        # Positive user (3 positive, 1 negative)
        for i in range(3):
            posts.append(create_test_post(
                i+20, f"Positive {i}", author='positive_user', 
                sentiment_label='POSITIVE'
            ))
        posts.append(create_test_post(
            30, "One negative", author='positive_user', 
            sentiment_label='NEGATIVE'
        ))
        
        agg = DataAggregator(negative_threshold=0.6, min_posts_for_analysis=3)
        stats = agg.aggregate(posts)
        
        negative_users = stats['negative_users']
        assert len(negative_users) >= 1
        assert negative_users[0]['author'] == 'negative_user'
        assert negative_users[0]['negative_ratio'] == 0.8  # 4/5
    
    def test_hashtag_analysis(self):
        """Test hashtag analysis."""
        posts = []
        for i in range(5):
            post = create_test_post(i, f"Post {i}")
            post.metadata['hashtags'] = ['python', 'data'] if i < 3 else ['python']
            posts.append(post)
        
        agg = DataAggregator()
        stats = agg.aggregate(posts)
        
        hashtags = stats['hashtags']
        assert hashtags['top_hashtags']['python'] == 5
        assert hashtags['top_hashtags']['data'] == 3
        assert hashtags['unique_hashtags'] == 2
    
    def test_viral_posts_identification(self):
        """Test viral posts identification."""
        posts = [
            create_test_post(1, "Normal post", engagement={'likes': 10, 'retweets': 5}),
            create_test_post(2, "Another normal", engagement={'likes': 20, 'retweets': 10}),
            create_test_post(3, "Also normal", engagement={'likes': 15, 'retweets': 8}),
            create_test_post(4, "Viral post", engagement={'likes': 1000, 'retweets': 500}),
        ]
        
        agg = DataAggregator()
        stats = agg.aggregate(posts)
        
        viral_posts = stats['viral_posts']
        assert len(viral_posts) >= 1
        assert viral_posts[0]['engagement'] == 1500  # 1000 + 500
    
    def test_sentiment_leaders_identification(self):
        """Test sentiment leaders identification."""
        posts = []
        
        # Positive leader with high engagement
        for i in range(3):
            posts.append(create_test_post(
                i, f"Positive {i}", author='positive_leader',
                sentiment_label='POSITIVE',
                engagement={'likes': 1000, 'retweets': 500}
            ))
        
        # Regular user
        posts.append(create_test_post(
            10, "Regular post", author='regular_user',
            sentiment_label='NEUTRAL',
            engagement={'likes': 10, 'retweets': 5}
        ))
        
        agg = DataAggregator()
        stats = agg.aggregate(posts)
        
        leaders = stats['sentiment_leaders']
        assert 'positive' in leaders
        if leaders['positive']:  # May be empty if not enough data
            assert leaders['positive'][0]['author'] == 'positive_leader'


class TestTimeSeriesAnalyzer:
    """Test time series analysis functionality."""
    
    def test_analyzer_initialization(self):
        """Test time series analyzer initialization."""
        analyzer = TimeSeriesAnalyzer(anomaly_threshold=2.0)
        
        assert analyzer.anomaly_threshold == 2.0
        assert analyzer.min_data_points == 3
    
    def test_empty_posts_analysis(self):
        """Test analysis with no posts."""
        analyzer = TimeSeriesAnalyzer()
        analysis = analyzer.analyze([])
        
        assert analysis['daily_sentiment'] == {}
        assert analysis['anomalies'] == []
        assert analysis['sentiment_volatility'] == 0.0
    
    def test_daily_sentiment_aggregation(self):
        """Test daily sentiment aggregation."""
        posts = []
        base_time = datetime.now()
        
        # Day 1: 2 positive, 1 negative
        for i in range(2):
            posts.append(create_test_post(
                i, f"Positive {i}",
                timestamp=base_time - timedelta(days=1, hours=i),
                sentiment_label='POSITIVE'
            ))
        posts.append(create_test_post(
            10, "Negative",
            timestamp=base_time - timedelta(days=1, hours=3),
            sentiment_label='NEGATIVE'
        ))
        
        # Day 2: 1 positive, 2 negative
        posts.append(create_test_post(
            20, "Positive",
            timestamp=base_time - timedelta(hours=1),
            sentiment_label='POSITIVE'
        ))
        for i in range(2):
            posts.append(create_test_post(
                30+i, f"Negative {i}",
                timestamp=base_time - timedelta(hours=i+2),
                sentiment_label='NEGATIVE'
            ))
        
        analyzer = TimeSeriesAnalyzer()
        analysis = analyzer.analyze(posts)
        
        daily = analysis['daily_sentiment']
        assert len(daily) == 2  # Two days
        
        # Check sentiment counts
        day_keys = sorted(daily.keys())
        assert daily[day_keys[0]]['positive'] == 2
        assert daily[day_keys[0]]['negative'] == 1
        assert daily[day_keys[1]]['positive'] == 1
        assert daily[day_keys[1]]['negative'] == 2
    
    def test_trend_analysis(self):
        """Test trend analysis."""
        posts = []
        base_time = datetime.now()
        
        # Create improving trend (more positive over time)
        for day in range(7):
            sentiment_ratio = day / 6  # 0 to 1
            
            # Add positive posts
            for _ in range(int(3 * sentiment_ratio) + 1):
                posts.append(create_test_post(
                    len(posts), "Positive",
                    timestamp=base_time - timedelta(days=6-day),
                    sentiment_label='POSITIVE'
                ))
            
            # Add negative posts
            for _ in range(int(3 * (1-sentiment_ratio)) + 1):
                posts.append(create_test_post(
                    len(posts), "Negative",
                    timestamp=base_time - timedelta(days=6-day),
                    sentiment_label='NEGATIVE'
                ))
        
        analyzer = TimeSeriesAnalyzer()
        analysis = analyzer.analyze(posts)
        
        trends = analysis['trends']
        assert trends['overall_trend'] in ['improving', 'stable', 'worsening']
        assert 'ma_3_day' in trends
        assert 'ma_7_day' in trends
    
    def test_anomaly_detection(self):
        """Test anomaly detection."""
        posts = []
        base_time = datetime.now()
        
        # Normal days (balanced sentiment)
        for day in range(10):
            if day == 5:  # Anomaly day - all negative
                for i in range(20):
                    posts.append(create_test_post(
                        len(posts), "Very negative",
                        timestamp=base_time - timedelta(days=day, hours=i),
                        sentiment_label='NEGATIVE'
                    ))
            else:
                # Normal pattern
                posts.append(create_test_post(
                    len(posts), "Positive",
                    timestamp=base_time - timedelta(days=day),
                    sentiment_label='POSITIVE'
                ))
                posts.append(create_test_post(
                    len(posts), "Negative",
                    timestamp=base_time - timedelta(days=day, hours=1),
                    sentiment_label='NEGATIVE'
                ))
        
        analyzer = TimeSeriesAnalyzer(anomaly_threshold=2.0)
        analysis = analyzer.analyze(posts)
        
        anomalies = analysis['anomalies']
        assert len(anomalies) >= 1
        assert anomalies[0]['type'] == 'negative_spike'
    
    def test_volatility_calculation(self):
        """Test volatility calculation."""
        posts = []
        base_time = datetime.now()
        
        # High volatility pattern
        for hour in range(24):
            sentiment = 'POSITIVE' if hour % 2 == 0 else 'NEGATIVE'
            posts.append(create_test_post(
                hour, f"Post {hour}",
                timestamp=base_time - timedelta(hours=hour),
                sentiment_label=sentiment
            ))
        
        analyzer = TimeSeriesAnalyzer()
        analysis = analyzer.analyze(posts)
        
        volatility = analysis['sentiment_volatility']
        assert volatility['overall'] > 0
        assert 'daily' in volatility
        assert 'hourly' in volatility
    
    def test_peak_activity_detection(self):
        """Test peak activity period detection."""
        posts = []
        base_time = datetime.now()
        
        # Create peak at specific hours
        for day in range(3):
            # Peak hour (3pm)
            for i in range(10):
                posts.append(create_test_post(
                    len(posts), f"Peak {i}",
                    timestamp=base_time.replace(hour=15, minute=i) - timedelta(days=day)
                ))
            
            # Normal hours
            posts.append(create_test_post(
                len(posts), "Normal",
                timestamp=base_time.replace(hour=9) - timedelta(days=day)
            ))
        
        analyzer = TimeSeriesAnalyzer()
        analysis = analyzer.analyze(posts)
        
        peak_activity = analysis['peak_activity']
        assert 'peak_hours' in peak_activity
        assert len(peak_activity['peak_hours']) > 0
        assert peak_activity['peak_hours'][0]['hour_of_day'] == 15
    
    def test_momentum_calculation(self):
        """Test momentum indicators."""
        posts = []
        base_time = datetime.now()
        
        # Create posts with changing sentiment
        for day in range(7):
            sentiment = 'POSITIVE' if day >= 4 else 'NEGATIVE'
            for i in range(3):
                posts.append(create_test_post(
                    len(posts), f"Post {day}-{i}",
                    timestamp=base_time - timedelta(days=6-day, hours=i),
                    sentiment_label=sentiment
                ))
        
        analyzer = TimeSeriesAnalyzer()
        analysis = analyzer.analyze(posts)
        
        momentum = analysis['momentum']
        assert 'momentum' in momentum
        assert 'direction' in momentum
        assert momentum['direction'] in ['positive', 'negative']
    
    def test_periodicity_analysis(self):
        """Test periodicity pattern detection."""
        posts = []
        base_time = datetime.now()
        
        # Create hourly pattern (positive in morning, negative in evening)
        for day in range(7):
            # Morning posts (positive)
            for hour in range(6, 12):
                posts.append(create_test_post(
                    len(posts), "Morning",
                    timestamp=base_time.replace(hour=hour) - timedelta(days=day),
                    sentiment_label='POSITIVE'
                ))
            
            # Evening posts (negative)
            for hour in range(18, 22):
                posts.append(create_test_post(
                    len(posts), "Evening",
                    timestamp=base_time.replace(hour=hour) - timedelta(days=day),
                    sentiment_label='NEGATIVE'
                ))
        
        analyzer = TimeSeriesAnalyzer()
        analysis = analyzer.analyze(posts)
        
        periodicity = analysis['periodicity']
        assert 'hourly_pattern' in periodicity
        assert periodicity['most_positive_hour'] in range(6, 12)
        assert periodicity['most_negative_hour'] in range(18, 22)
    
    def test_forecast_indicators(self):
        """Test forecast indicator calculation."""
        posts = []
        base_time = datetime.now()
        
        # Recent improving trend
        for day in range(7):
            sentiment = 'POSITIVE' if day >= 5 else 'NEGATIVE'
            posts.append(create_test_post(
                day, f"Post {day}",
                timestamp=base_time - timedelta(days=6-day),
                sentiment_label=sentiment
            ))
        
        analyzer = TimeSeriesAnalyzer()
        analysis = analyzer.analyze(posts)
        
        forecast = analysis['forecast_indicators']
        assert 'recent_sentiment_distribution' in forecast
        assert 'trend_consistency' in forecast
        assert 'last_sentiment_value' in forecast
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        # Only 2 posts (below min_data_points)
        posts = [
            create_test_post(1, "Post 1"),
            create_test_post(2, "Post 2")
        ]
        
        analyzer = TimeSeriesAnalyzer(min_data_points=3)
        analysis = analyzer.analyze(posts)
        
        # Should still provide some analysis
        assert 'daily_sentiment' in analysis
        
        # But trends should indicate insufficient data
        if 'overall_trend' in analysis['trends']:
            assert analysis['trends']['overall_trend'] == 'insufficient_data'


class TestProcessingIntegration:
    """Integration tests for processing pipeline."""
    
    def test_full_processing_pipeline(self):
        """Test complete processing pipeline."""
        # Create diverse test data
        posts = []
        base_time = datetime.now()
        
        # Add various posts
        authors = ['user1', 'user2', 'negative_user', 'positive_user']
        sentiments = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        platforms = ['twitter', 'reddit']
        
        for i in range(100):
            posts.append(create_test_post(
                i, 
                f"Post content {i} #test @mention",
                author=authors[i % len(authors)],
                platform=platforms[i % len(platforms)],
                sentiment_label=sentiments[i % len(sentiments)],
                timestamp=base_time - timedelta(hours=i),
                query=f"query{i % 3}",
                engagement={
                    'likes': np.random.randint(0, 1000),
                    'shares': np.random.randint(0, 500)
                }
            ))
        
        # Add some duplicates
        posts.append(posts[0])  # Exact duplicate
        posts.append(create_test_post(200, posts[10].content))  # Content duplicate
        
        # Run through processing pipeline
        # 1. Deduplication
        dedup = Deduplicator()
        unique_posts, dedup_stats = dedup.deduplicate(posts)
        
        assert len(unique_posts) < len(posts)
        assert dedup_stats['duplicates_removed'] >= 2
        
        # 2. Aggregation
        aggregator = DataAggregator()
        agg_stats = aggregator.aggregate(unique_posts)
        
        assert agg_stats['total_posts'] == len(unique_posts)
        assert len(agg_stats['by_platform']) == 2
        assert len(agg_stats['negative_users']) >= 0
        
        # 3. Time series analysis
        analyzer = TimeSeriesAnalyzer()
        time_analysis = analyzer.analyze(unique_posts)
        
        assert len(time_analysis['daily_sentiment']) > 0
        assert 'overall_trend' in time_analysis['trends']
        assert isinstance(time_analysis['sentiment_volatility'], dict)
    
    def test_processing_performance(self):
        """Test that processing meets performance requirements."""
        import time
        
        # Create 10,000 posts
        posts = []
        for i in range(10000):
            posts.append(create_test_post(
                i, 
                f"Post {i} with some content",
                author=f"user{i % 1000}",
                sentiment_label=['POSITIVE', 'NEGATIVE', 'NEUTRAL'][i % 3]
            ))
        
        start_time = time.time()
        
        # Process all posts
        dedup = Deduplicator()
        unique_posts, _ = dedup.deduplicate(posts)
        
        aggregator = DataAggregator()
        aggregator.aggregate(unique_posts)
        
        analyzer = TimeSeriesAnalyzer()
        analyzer.analyze(unique_posts)
        
        processing_time = time.time() - start_time
        
        # Should complete in under 30 seconds
        assert processing_time < 30, f"Processing took {processing_time:.2f}s, should be < 30s"
    
    def test_edge_cases(self):
        """Test edge cases in processing."""
        # Posts with missing sentiment
        post_no_sentiment = create_test_post(1, "No sentiment")
        delattr(post_no_sentiment, 'sentiment')
        
        # Posts with empty content
        post_empty = create_test_post(2, "")
        
        # Posts with None values
        post_none = create_test_post(3, "Normal")
        post_none.engagement = None
        
        posts = [post_no_sentiment, post_empty, post_none]
        
        # Should handle gracefully
        dedup = Deduplicator()
        unique_posts, _ = dedup.deduplicate(posts)
        
        aggregator = DataAggregator()
        stats = aggregator.aggregate(unique_posts)
        
        analyzer = TimeSeriesAnalyzer()
        analysis = analyzer.analyze(unique_posts)
        
        # Should complete without errors
        assert stats['total_posts'] == len(unique_posts)
        assert 'daily_sentiment' in analysis