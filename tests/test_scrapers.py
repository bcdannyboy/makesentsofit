"""
Tests for scraper implementations.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import time

from src.scrapers import create_scrapers
from src.scrapers.base import Post, BaseScraper
from src.scrapers.rate_limiter import RateLimiter, AdaptiveRateLimiter
from src.scrapers.twitter import TwitterScraper
from src.scrapers.reddit import RedditScraper
from src.config import Config


class TestPost:
    """Test Post dataclass."""
    
    def test_post_creation(self):
        """Test creating a Post object."""
        post = Post(
            id='123',
            platform='twitter',
            author='testuser',
            author_id='456',
            content='Test content #hashtag @mention',
            title=None,
            timestamp=datetime.now(),
            engagement={'likes': 10, 'retweets': 5},
            url='https://twitter.com/test/123',
            query='test query',
            metadata={'lang': 'en'}
        )
        
        assert post.id == '123'
        assert post.platform == 'twitter'
        assert post.author == 'testuser'
        assert post.get_engagement_score() == 15
        assert 'hashtag' in post.content
    
    def test_post_to_dict(self):
        """Test converting Post to dictionary."""
        timestamp = datetime.now()
        post = Post(
            id='123',
            platform='reddit',
            author='redditor',
            author_id=None,
            content='Test post',
            title='Test Title',
            timestamp=timestamp,
            engagement={'score': 100},
            url='https://reddit.com/r/test/123',
            query='test'
        )
        
        post_dict = post.to_dict()
        
        assert post_dict['id'] == '123'
        assert post_dict['platform'] == 'reddit'
        assert post_dict['title'] == 'Test Title'
        assert isinstance(post_dict['timestamp'], str)
        assert post_dict['engagement']['score'] == 100
    
    def test_post_string_timestamp(self):
        """Test Post with string timestamp."""
        post = Post(
            id='123',
            platform='twitter',
            author='user',
            author_id='456',
            content='Test',
            title=None,
            timestamp='2024-01-01T12:00:00',
            engagement={},
            url='https://example.com',
            query='test'
        )
        
        assert isinstance(post.timestamp, datetime)
    
    def test_post_invalid_data(self):
        """Test Post with invalid data types."""
        # Should handle non-dict engagement
        post = Post(
            id='123',
            platform='twitter',
            author='user',
            author_id='456',
            content='Test',
            title=None,
            timestamp=datetime.now(),
            engagement=None,  # Invalid
            url='https://example.com',
            query='test'
        )
        
        assert isinstance(post.engagement, dict)
        assert post.engagement == {}


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    def test_rate_limiter_basic(self):
        """Test basic rate limiting."""
        limiter = RateLimiter(calls_per_minute=60)
        
        # First call should not wait
        wait_time = limiter.wait_if_needed()
        assert wait_time == 0.0
        
        # Stats should be updated
        stats = limiter.get_stats()
        assert stats['total_calls'] == 1
        assert stats['remaining_calls'] == 59
    
    def test_rate_limiter_burst(self):
        """Test burst limiting."""
        limiter = RateLimiter(calls_per_minute=60, burst_size=5)
        
        # Make 5 quick calls (burst limit)
        for _ in range(5):
            limiter.wait_if_needed()
        
        # 6th call should wait
        start = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start
        
        # Should have waited at least min_interval
        assert elapsed >= (60.0 / 60) - 0.1  # Allow small margin
    
    def test_rate_limiter_window(self):
        """Test sliding window behavior."""
        limiter = RateLimiter(calls_per_minute=10)
        
        # Make 10 calls
        for _ in range(10):
            limiter.wait_if_needed()
        
        assert limiter.get_remaining_calls() == 0
        
        # Wait a bit and check cleanup
        time.sleep(0.1)
        
        # Old timestamps should be cleaned up over time
        with limiter.lock:
            limiter._cleanup_old_timestamps(time.time() + 61)
        
        assert len(limiter.call_times) == 0
    
    def test_rate_limiter_stats(self):
        """Test rate limiter statistics."""
        limiter = RateLimiter(calls_per_minute=120)
        
        # Make some calls
        for _ in range(5):
            limiter.wait_if_needed()
            time.sleep(0.01)
        
        stats = limiter.get_stats()
        assert stats['total_calls'] == 5
        assert stats['current_rate'] > 0
        assert stats['remaining_calls'] == 115
        assert stats['total_wait_time'] >= 0
    
    def test_adaptive_rate_limiter(self):
        """Test adaptive rate limiter."""
        limiter = AdaptiveRateLimiter(
            calls_per_minute=60,
            min_rate=10,
            max_rate=120
        )
        
        # Record successes
        for _ in range(20):
            limiter.record_success()
        
        # Should potentially increase rate
        limiter._maybe_adjust_rate()
        
        # Record errors
        for _ in range(10):
            limiter.record_error()
        
        # Should potentially decrease rate
        limiter._maybe_adjust_rate()
        
        # Rate limit error should immediately reduce
        initial_rate = limiter.calls_per_minute
        limiter.record_error(is_rate_limit=True)
        assert limiter.calls_per_minute < initial_rate


class TestBaseScraper:
    """Test base scraper functionality."""
    
    def test_scraper_initialization(self):
        """Test scraper initialization."""
        rate_limiter = RateLimiter()
        
        # Create a concrete implementation for testing
        class TestScraper(BaseScraper):
            def scrape(self, query, start_date, end_date):
                return []
            
            def validate_connection(self):
                return True
        
        scraper = TestScraper(rate_limiter)
        
        assert scraper.rate_limiter == rate_limiter
        assert scraper.posts_collected == 0
        assert scraper.errors_count == 0
    
    def test_scraper_multiple_queries(self):
        """Test scraping multiple queries."""
        rate_limiter = RateLimiter()
        
        class TestScraper(BaseScraper):
            def scrape(self, query, start_date, end_date):
                # Return different posts for different queries
                if query == 'test1':
                    return [Mock(spec=Post) for _ in range(3)]
                elif query == 'test2':
                    return [Mock(spec=Post) for _ in range(2)]
                return []
            
            def validate_connection(self):
                return True
        
        scraper = TestScraper(rate_limiter)
        
        queries = ['test1', 'test2', 'test3']
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        posts = scraper.scrape_multiple(queries, start_date, end_date)
        
        assert len(posts) == 5  # 3 + 2 + 0
    
    def test_scraper_error_handling(self):
        """Test scraper error handling."""
        rate_limiter = RateLimiter()
        
        class TestScraper(BaseScraper):
            def scrape(self, query, start_date, end_date):
                if query == 'error':
                    raise Exception("Test error")
                return [Mock(spec=Post)]
            
            def validate_connection(self):
                return True
        
        scraper = TestScraper(rate_limiter)
        
        queries = ['good', 'error', 'good2']
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()
        
        # Should continue despite error
        posts = scraper.scrape_multiple(queries, start_date, end_date)
        
        assert len(posts) == 2  # Only non-error queries
        assert scraper.errors_count == 1
        assert scraper.last_error == "Test error"
    
    def test_date_validation(self):
        """Test date range validation."""
        rate_limiter = RateLimiter()
        
        class TestScraper(BaseScraper):
            def scrape(self, query, start_date, end_date):
                self._validate_date_range(start_date, end_date)
                return []
            
            def validate_connection(self):
                return True
        
        scraper = TestScraper(rate_limiter)
        
        # Invalid date range
        with pytest.raises(ValueError):
            scraper.scrape('test', datetime.now(), datetime.now() - timedelta(days=1))
    
    def test_query_cleaning(self):
        """Test query cleaning."""
        rate_limiter = RateLimiter()
        
        class TestScraper(BaseScraper):
            def scrape(self, query, start_date, end_date):
                return []
            
            def validate_connection(self):
                return True
        
        scraper = TestScraper(rate_limiter)
        
        # Test various problematic queries
        assert scraper._clean_query('  test  ') == 'test'
        assert scraper._clean_query('test\nquery') == 'test query'
        assert scraper._clean_query('test\t\rquery') == 'test query'
        assert scraper._clean_query('test    query') == 'test query'


class TestTwitterScraper:
    """Test Twitter scraper functionality."""
    
    @patch('src.scrapers.twitter.SNSCRAPE_AVAILABLE', False)
    def test_twitter_no_snscrape(self):
        """Test Twitter scraper without snscrape."""
        rate_limiter = RateLimiter()
        
        with pytest.raises(ImportError):
            TwitterScraper(rate_limiter)
    
    @patch('src.scrapers.twitter.sntwitter')
    @patch('src.scrapers.twitter.SNSCRAPE_AVAILABLE', True)
    def test_twitter_connection_validation(self, mock_sntwitter):
        """Test Twitter connection validation."""
        rate_limiter = RateLimiter()
        scraper = TwitterScraper(rate_limiter)
        
        # Mock successful connection
        mock_scraper = Mock()
        mock_scraper.get_items.return_value = iter([Mock()])
        mock_sntwitter.TwitterSearchScraper.return_value = mock_scraper
        
        assert scraper.validate_connection() is True
        
        # Mock failed connection
        mock_sntwitter.TwitterSearchScraper.side_effect = Exception("Connection error")
        assert scraper.validate_connection() is False
    
    @patch('src.scrapers.twitter.sntwitter')
    @patch('src.scrapers.twitter.SNSCRAPE_AVAILABLE', True)
    def test_twitter_scraping(self, mock_sntwitter):
        """Test Twitter scraping."""
        rate_limiter = RateLimiter()
        scraper = TwitterScraper(rate_limiter, max_tweets_per_query=5)
        
        # Create mock tweets
        mock_tweets = []
        for i in range(10):
            tweet = Mock()
            tweet.id = str(i)
            tweet.date = datetime.now() - timedelta(hours=i)
            tweet.user.username = f'user{i}'
            tweet.user.id = str(100 + i)
            tweet.rawContent = f'Test tweet {i} #test @mention'
            tweet.likeCount = i * 10
            tweet.retweetCount = i * 5
            tweet.replyCount = i * 2
            tweet.quoteCount = i
            tweet.url = f'https://twitter.com/user{i}/status/{i}'
            tweet.lang = 'en'
            tweet.hashtags = []
            tweet.mentionedUsers = []
            tweet.retweetedTweet = None
            tweet.quotedTweet = None
            tweet.inReplyToTweetId = None
            mock_tweets.append(tweet)
        
        # Mock scraper
        mock_scraper = Mock()
        mock_scraper.get_items.return_value = iter(mock_tweets)
        mock_sntwitter.TwitterSearchScraper.return_value = mock_scraper
        
        # Scrape
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        posts = scraper.scrape('test query', start_date, end_date)
        
        # Should respect max limit
        assert len(posts) == 5
        
        # Check first post
        first_post = posts[0]
        assert first_post.id == '0'
        assert first_post.platform == 'twitter'
        assert first_post.author == 'user0'
        assert '#test' in first_post.content
        assert first_post.engagement['likes'] == 0
    
    def test_twitter_date_query_building(self):
        """Test Twitter date query building."""
        rate_limiter = RateLimiter()
        
        with patch('src.scrapers.twitter.SNSCRAPE_AVAILABLE', True):
            scraper = TwitterScraper(rate_limiter)
            
            start = datetime(2024, 1, 1)
            end = datetime(2024, 1, 7)
            
            query = scraper._build_date_query('bitcoin', start, end)
            
            assert 'bitcoin' in query
            assert 'since:2024-01-01' in query
            assert 'until:2024-01-07' in query
    
    def test_twitter_metadata_extraction(self):
        """Test Twitter metadata extraction."""
        rate_limiter = RateLimiter()
        
        with patch('src.scrapers.twitter.SNSCRAPE_AVAILABLE', True):
            scraper = TwitterScraper(rate_limiter)
            
            # Test hashtag extraction
            tweet = Mock()
            tweet.hashtags = None
            tweet.rawContent = 'Test #bitcoin #crypto'
            
            hashtags = scraper._extract_hashtags(tweet)
            assert 'bitcoin' in hashtags
            assert 'crypto' in hashtags
            
            # Test mention extraction
            tweet.mentionedUsers = None
            tweet.rawContent = 'Hey @user1 and @user2!'
            
            mentions = scraper._extract_mentions(tweet)
            assert 'user1' in mentions
            assert 'user2' in mentions


class TestRedditScraper:
    """Test Reddit scraper functionality."""
    
    @patch('src.scrapers.reddit.PRAW_AVAILABLE', False)
    def test_reddit_no_praw(self):
        """Test Reddit scraper without PRAW."""
        rate_limiter = RateLimiter()
        
        with pytest.raises(ImportError):
            RedditScraper(rate_limiter)
    
    @patch('src.scrapers.reddit.praw')
    @patch('src.scrapers.reddit.PRAW_AVAILABLE', True)
    def test_reddit_initialization(self, mock_praw):
        """Test Reddit scraper initialization."""
        rate_limiter = RateLimiter()
        
        mock_reddit = Mock()
        mock_praw.Reddit.return_value = mock_reddit
        
        scraper = RedditScraper(rate_limiter, subreddits=['python', 'programming'])
        
        assert scraper.subreddits == ['python', 'programming']
        assert mock_reddit.read_only is True
    
    @patch('src.scrapers.reddit.praw')
    @patch('src.scrapers.reddit.PRAW_AVAILABLE', True)
    def test_reddit_connection_validation(self, mock_praw):
        """Test Reddit connection validation."""
        rate_limiter = RateLimiter()
        
        mock_reddit = Mock()
        mock_subreddit = Mock()
        mock_subreddit.id = 'test123'
        mock_reddit.subreddit.return_value = mock_subreddit
        mock_praw.Reddit.return_value = mock_reddit
        
        scraper = RedditScraper(rate_limiter)
        
        assert scraper.validate_connection() is True
        
        # Test failed connection
        mock_reddit.subreddit.side_effect = Exception("Connection error")
        assert scraper.validate_connection() is False
    
    @patch('src.scrapers.reddit.praw')
    @patch('src.scrapers.reddit.PRAW_AVAILABLE', True) 
    def test_reddit_scraping(self, mock_praw):
        """Test Reddit scraping."""
        rate_limiter = RateLimiter()
        
        # Create mock submissions
        mock_submissions = []
        for i in range(5):
            submission = Mock()
            submission.id = f'post{i}'
            submission.created_utc = (datetime.now() - timedelta(hours=i)).timestamp()
            submission.author = Mock()
            submission.author.name = f'redditor{i}'
            submission.title = f'Test Post {i}'
            submission.selftext = f'This is test content {i}'
            submission.score = 100 + i * 10
            submission.upvote_ratio = 0.95
            submission.num_comments = 20 + i
            submission.permalink = f'/r/test/comments/{i}/'
            submission.subreddit = Mock()
            submission.subreddit.display_name = 'test'
            submission.is_self = True
            submission.is_video = False
            submission.over_18 = False
            submission.spoiler = False
            submission.stickied = False
            submission.locked = False
            submission.link_flair_text = 'Discussion'
            mock_submissions.append(submission)
        
        # Mock Reddit and subreddit
        mock_reddit = Mock()
        mock_subreddit = Mock()
        mock_subreddit.search.return_value = mock_submissions
        mock_reddit.subreddit.return_value = mock_subreddit
        mock_praw.Reddit.return_value = mock_reddit
        
        scraper = RedditScraper(rate_limiter, subreddits=['test'])
        
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()
        posts = scraper.scrape('test query', start_date, end_date)
        
        assert len(posts) == 5
        
        # Check first post
        first_post = posts[0]
        assert first_post.id == 'post0'
        assert first_post.platform == 'reddit'
        assert first_post.author == 'redditor0'
        assert first_post.title == 'Test Post 0'
        assert first_post.metadata['subreddit'] == 'test'
    
    def test_reddit_time_filter(self):
        """Test Reddit time filter selection."""
        rate_limiter = RateLimiter()
        
        with patch('src.scrapers.reddit.PRAW_AVAILABLE', True):
            with patch('src.scrapers.reddit.praw'):
                scraper = RedditScraper(rate_limiter)
                
                # Test different date ranges
                start = datetime.now()
                
                # 1 day
                assert scraper._get_time_filter(
                    start - timedelta(days=1), start
                ) == 'day'
                
                # 1 week
                assert scraper._get_time_filter(
                    start - timedelta(days=7), start
                ) == 'week'
                
                # 1 month
                assert scraper._get_time_filter(
                    start - timedelta(days=30), start
                ) == 'month'
                
                # 1 year
                assert scraper._get_time_filter(
                    start - timedelta(days=365), start
                ) == 'year'
                
                # More than 1 year
                assert scraper._get_time_filter(
                    start - timedelta(days=400), start
                ) == 'all'
    
    @patch('src.scrapers.reddit.praw')
    @patch('src.scrapers.reddit.PRAW_AVAILABLE', True)
    def test_reddit_deleted_author(self, mock_praw):
        """Test handling deleted Reddit authors."""
        rate_limiter = RateLimiter()
        
        # Create submission with deleted author
        submission = Mock()
        submission.id = 'deleted_post'
        submission.created_utc = datetime.now().timestamp()
        submission.author = None  # Deleted
        submission.title = 'Deleted Author Post'
        submission.selftext = 'Content'
        submission.score = 50
        submission.upvote_ratio = 0.8
        submission.num_comments = 10
        submission.permalink = '/r/test/deleted/'
        submission.subreddit = Mock()
        submission.subreddit.display_name = 'test'
        
        # Mock Reddit
        mock_reddit = Mock()
        mock_subreddit = Mock()
        mock_subreddit.search.return_value = [submission]
        mock_reddit.subreddit.return_value = mock_subreddit
        mock_praw.Reddit.return_value = mock_reddit
        
        scraper = RedditScraper(rate_limiter)
        
        posts = scraper.scrape('test', datetime.now() - timedelta(days=1), datetime.now())
        
        assert len(posts) == 1
        assert posts[0].author == '[deleted]'
        assert posts[0].author_id is None


class TestScraperFactory:
    """Test scraper factory function."""
    
    @patch('src.scrapers.reddit.praw')
    @patch('src.scrapers.reddit.PRAW_AVAILABLE', True)
    @patch('src.scrapers.twitter.sntwitter')
    @patch('src.scrapers.twitter.SNSCRAPE_AVAILABLE', True)
    def test_create_scrapers(self, mock_sntwitter, mock_praw):
        """Test creating scrapers."""
        config = Config()
        
        # Create scrapers for both platforms
        scrapers = create_scrapers(['twitter', 'reddit'], config)
        
        assert 'twitter' in scrapers
        assert 'reddit' in scrapers
        assert isinstance(scrapers['twitter'], TwitterScraper)
        assert isinstance(scrapers['reddit'], RedditScraper)
    
    def test_create_scrapers_unknown_platform(self):
        """Test creating scrapers with unknown platform."""
        config = Config()
        
        # Should skip unknown platforms
        scrapers = create_scrapers(['twitter', 'facebook', 'instagram'], config)
        
        # Only twitter should be created (assuming snscrape is available)
        assert len(scrapers) <= 1  # Depends on whether snscrape is actually installed


# Integration tests
class TestScraperIntegration:
    """Integration tests for scrapers."""
    
    @pytest.mark.integration
    @patch('src.scrapers.twitter.sntwitter')
    @patch('src.scrapers.twitter.SNSCRAPE_AVAILABLE', True)
    @patch('src.scrapers.reddit.praw')
    @patch('src.scrapers.reddit.PRAW_AVAILABLE', True)
    def test_full_scraping_workflow(self, mock_praw, mock_sntwitter):
        """Test complete scraping workflow."""
        config = Config()
        
        # Mock Twitter data
        mock_tweet = Mock()
        mock_tweet.id = 'tw1'
        mock_tweet.date = datetime.now()
        mock_tweet.user = Mock(username='twitteruser', id='123')
        mock_tweet.rawContent = 'Twitter post'
        mock_tweet.likeCount = 100
        mock_tweet.retweetCount = 50
        mock_tweet.replyCount = 10
        mock_tweet.quoteCount = 5
        mock_tweet.url = 'https://twitter.com/test/tw1'
        mock_tweet.lang = 'en'
        mock_tweet.hashtags = []
        mock_tweet.mentionedUsers = []
        mock_tweet.retweetedTweet = None
        
        mock_twitter_scraper = Mock()
        mock_twitter_scraper.get_items.return_value = [mock_tweet]
        mock_sntwitter.TwitterSearchScraper.return_value = mock_twitter_scraper
        
        # Mock Reddit data
        mock_submission = Mock()
        mock_submission.id = 'rd1'
        mock_submission.created_utc = datetime.now().timestamp()
        mock_submission.author = Mock(name='redditor')
        mock_submission.title = 'Reddit Post'
        mock_submission.selftext = 'Reddit content'
        mock_submission.score = 200
        mock_submission.upvote_ratio = 0.9
        mock_submission.num_comments = 30
        mock_submission.permalink = '/r/test/rd1'
        mock_submission.subreddit = Mock(display_name='test')
        mock_submission.is_self = True
        mock_submission.is_video = False
        mock_submission.over_18 = False
        mock_submission.spoiler = False
        mock_submission.stickied = False
        mock_submission.locked = False
        
        mock_reddit = Mock()
        mock_subreddit = Mock()
        mock_subreddit.search.return_value = [mock_submission]
        mock_reddit.subreddit.return_value = mock_subreddit
        mock_praw.Reddit.return_value = mock_reddit
        
        # Create scrapers
        scrapers = create_scrapers(['twitter', 'reddit'], config)
        
        # Scrape data
        all_posts = []
        queries = ['test query']
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()
        
        for platform, scraper in scrapers.items():
            posts = scraper.scrape_multiple(queries, start_date, end_date)
            all_posts.extend(posts)
        
        # Verify results
        assert len(all_posts) == 2
        
        # Find posts by platform
        twitter_posts = [p for p in all_posts if p.platform == 'twitter']
        reddit_posts = [p for p in all_posts if p.platform == 'reddit']
        
        assert len(twitter_posts) == 1
        assert len(reddit_posts) == 1
        
        # Verify Twitter post
        tw_post = twitter_posts[0]
        assert tw_post.id == 'tw1'
        assert tw_post.author == 'twitteruser'
        assert tw_post.engagement['likes'] == 100
        
        # Verify Reddit post
        rd_post = reddit_posts[0]
        assert rd_post.id == 'rd1'
        assert rd_post.author == 'redditor'
        assert rd_post.title == 'Reddit Post'
        assert rd_post.engagement['score'] == 200