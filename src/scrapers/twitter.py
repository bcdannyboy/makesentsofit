"""
Twitter/X scraper implementation using snscrape.
"""
import logging
import ssl
import urllib3
from datetime import datetime
from typing import List, Optional, Dict, Any
import re

# Disable SSL warnings and verification for demo purposes
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set environment variable to disable SSL verification globally
import os
os.environ['PYTHONHTTPSVERIFY'] = '0'

try:
    import snscrape.modules.twitter as sntwitter
    import snscrape.base
    import requests
    import ssl
    
    SNSCRAPE_AVAILABLE = True
    
    # Global SSL context with verification disabled
    _ssl_context = ssl.create_default_context()
    _ssl_context.check_hostname = False
    _ssl_context.verify_mode = ssl.CERT_NONE
    
    # Monkey patch requests to disable SSL verification globally
    original_request = requests.Session.request
    
    def patched_request(self, method, url, **kwargs):
        kwargs['verify'] = False
        return original_request(self, method, url, **kwargs)
    
    requests.Session.request = patched_request
    
    # Patch urllib3 to use no-verify SSL context
    import urllib3.poolmanager
    original_poolmanager_init = urllib3.poolmanager.PoolManager.__init__
    
    def patched_poolmanager_init(self, *args, **kwargs):
        kwargs['ssl_context'] = _ssl_context
        return original_poolmanager_init(self, *args, **kwargs)
    
    urllib3.poolmanager.PoolManager.__init__ = patched_poolmanager_init
    
    # Patch HTTPSConnectionPool
    import urllib3.connectionpool
    original_https_init = urllib3.connectionpool.HTTPSConnectionPool.__init__
    
    def patched_https_init(self, *args, **kwargs):
        kwargs['ssl_context'] = _ssl_context
        return original_https_init(self, *args, **kwargs)
    
    urllib3.connectionpool.HTTPSConnectionPool.__init__ = patched_https_init
    
except ImportError:
    SNSCRAPE_AVAILABLE = False
    sntwitter = None

from .base import BaseScraper, Post
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

class TwitterScraper(BaseScraper):
    """
    Twitter/X scraper using snscrape library.
    
    This scraper uses snscrape which doesn't require API credentials
    and can access historical tweets without the limitations of the
    official Twitter API.
    """
    
    def __init__(self, rate_limiter: RateLimiter, 
                 max_tweets_per_query: Optional[int] = None):
        """
        Initialize Twitter scraper.
        
        Args:
            rate_limiter: Rate limiter instance
            max_tweets_per_query: Maximum tweets to collect per query
        """
        super().__init__(rate_limiter)
        self.max_tweets_per_query = max_tweets_per_query
        
        if not SNSCRAPE_AVAILABLE:
            logger.error("snscrape is not installed. Install with: pip install snscrape")
            raise ImportError("snscrape is required for Twitter scraping")
    
    def validate_connection(self) -> bool:
        """
        Test connection to Twitter.
        
        Returns:
            True if connection is valid
        """
        if not SNSCRAPE_AVAILABLE:
            return False
        
        try:
            # Try a minimal search to test connection using new mode syntax
            scraper = sntwitter.TwitterSearchScraper(
                'test',
                mode=sntwitter.TwitterSearchScraperMode.TOP
            )
            
            # Try to get just one tweet
            for i, tweet in enumerate(scraper.get_items()):
                if i >= 0:  # Just check if we can iterate
                    break
            
            logger.debug("Twitter connection validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Twitter connection validation failed: {e}")
            return False
    
    def scrape(self, query: str, start_date: datetime, 
               end_date: datetime) -> List[Post]:
        """
        Scrape Twitter for posts matching query.
        
        Args:
            query: Search query
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            List of posts
        """
        # Validate date range
        self._validate_date_range(start_date, end_date)
        
        # Clean query
        query = self._clean_query(query)
        
        # Build Twitter search query with date filters
        date_query = self._build_date_query(query, start_date, end_date)
        
        logger.info(f"Starting Twitter scrape: '{query}' from {start_date.date()} to {end_date.date()}")
        
        posts = []
        tweet_count = 0
        
        try:
            # Create scraper instance
            scraper = sntwitter.TwitterSearchScraper(date_query)
            
            # Iterate through tweets
            for tweet in scraper.get_items():
                # Rate limiting
                self.rate_limiter.wait_if_needed()
                
                # Check if we've reached the limit
                if self.max_tweets_per_query and tweet_count >= self.max_tweets_per_query:
                    logger.info(f"Reached max tweets limit: {self.max_tweets_per_query}")
                    break
                
                # Double-check date bounds (snscrape sometimes returns outside range)
                if not self._is_within_date_range(tweet.date, start_date, end_date):
                    continue
                
                # Convert tweet to Post object
                post = self._tweet_to_post(tweet, query)
                posts.append(post)
                
                tweet_count += 1
                self.posts_collected += 1
                
                # Progress logging
                if tweet_count % 100 == 0:
                    logger.debug(f"Collected {tweet_count} tweets for '{query}'...")
            
            logger.info(f"Completed Twitter scrape: {tweet_count} tweets for '{query}'")
            
        except Exception as e:
            logger.error(f"Error during Twitter scrape for '{query}': {e}")
            self.errors_count += 1
            self.last_error = str(e)
            raise
        
        return posts
    
    def _build_date_query(self, query: str, start_date: datetime, 
                          end_date: datetime) -> str:
        """
        Build Twitter search query with date filters.
        
        Args:
            query: Base search query
            start_date: Start date
            end_date: End date
            
        Returns:
            Query string with date filters
        """
        # Format dates for Twitter search
        since = start_date.strftime('%Y-%m-%d')
        until = end_date.strftime('%Y-%m-%d')
        
        # Build query with date range
        date_query = f"{query} since:{since} until:{until}"
        
        return date_query
    
    def _is_within_date_range(self, tweet_date: datetime, 
                              start_date: datetime, end_date: datetime) -> bool:
        """
        Check if tweet date is within specified range.
        
        Args:
            tweet_date: Tweet timestamp
            start_date: Start of range
            end_date: End of range
            
        Returns:
            True if within range
        """
        # Handle timezone-naive comparison
        if tweet_date.tzinfo and not start_date.tzinfo:
            tweet_date = tweet_date.replace(tzinfo=None)
        
        return start_date <= tweet_date <= end_date
    
    def _tweet_to_post(self, tweet: Any, query: str) -> Post:
        """
        Convert snscrape Tweet object to Post object.
        
        Args:
            tweet: snscrape Tweet object
            query: Search query that found this tweet
            
        Returns:
            Post object
        """
        # Extract engagement metrics
        engagement = {
            'likes': getattr(tweet, 'likeCount', 0) or 0,
            'retweets': getattr(tweet, 'retweetCount', 0) or 0,
            'replies': getattr(tweet, 'replyCount', 0) or 0,
            'quotes': getattr(tweet, 'quoteCount', 0) or 0
        }
        
        # Extract metadata
        metadata = {
            'lang': getattr(tweet, 'lang', None),
            'hashtags': self._extract_hashtags(tweet),
            'mentions': self._extract_mentions(tweet),
            'urls': self._extract_urls(tweet),
            'is_retweet': bool(getattr(tweet, 'retweetedTweet', None)),
            'is_quote': bool(getattr(tweet, 'quotedTweet', None)),
            'is_reply': bool(getattr(tweet, 'inReplyToTweetId', None)),
            'media_types': self._extract_media_types(tweet),
            'coordinates': self._extract_coordinates(tweet)
        }
        
        # Create Post object
        post = Post(
            id=str(tweet.id),
            platform='twitter',
            author=tweet.user.username if tweet.user else 'unknown',
            author_id=str(tweet.user.id) if tweet.user else None,
            content=getattr(tweet, 'rawContent', '') or '',
            title=None,  # Twitter doesn't have titles
            timestamp=tweet.date.replace(tzinfo=None) if tweet.date.tzinfo else tweet.date,
            engagement=engagement,
            url=str(tweet.url) if hasattr(tweet, 'url') else f"https://twitter.com/i/status/{tweet.id}",
            query=query,
            metadata=metadata
        )
        
        return post
    
    def _extract_hashtags(self, tweet: Any) -> List[str]:
        """Extract hashtags from tweet."""
        if hasattr(tweet, 'hashtags') and tweet.hashtags:
            return [tag.text for tag in tweet.hashtags]
        
        # Fallback: extract from content
        content = getattr(tweet, 'rawContent', '') or ''
        return re.findall(r'#(\w+)', content)
    
    def _extract_mentions(self, tweet: Any) -> List[str]:
        """Extract user mentions from tweet."""
        if hasattr(tweet, 'mentionedUsers') and tweet.mentionedUsers:
            return [user.username for user in tweet.mentionedUsers if hasattr(user, 'username')]
        
        # Fallback: extract from content
        content = getattr(tweet, 'rawContent', '') or ''
        return re.findall(r'@(\w+)', content)
    
    def _extract_urls(self, tweet: Any) -> List[str]:
        """Extract URLs from tweet."""
        if hasattr(tweet, 'outlinks') and tweet.outlinks:
            try:
                return list(tweet.outlinks)
            except TypeError:
                return []
        
        # Fallback: extract from content
        content = getattr(tweet, 'rawContent', '') or ''
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+' 
        return re.findall(url_pattern, content)
    
    def _extract_media_types(self, tweet: Any) -> List[str]:
        """Extract media types from tweet."""
        media_types = []
        
        if hasattr(tweet, 'media') and tweet.media:
            try:
                for media_item in tweet.media:
                    if hasattr(media_item, 'type'):
                        media_types.append(media_item.type)
            except TypeError:
                return media_types
        
        return media_types
    
    def _extract_coordinates(self, tweet: Any) -> Optional[Dict[str, float]]:
        """Extract coordinates if available."""
        if hasattr(tweet, 'coordinates') and tweet.coordinates:
            return {
                'lat': tweet.coordinates.latitude,
                'lon': tweet.coordinates.longitude
            }
        return None


class TwitterAdvancedScraper(TwitterScraper):
    """
    Advanced Twitter scraper with additional features.
    
    Includes user timeline scraping, advanced filtering,
    and conversation thread collection.
    """
    
    def scrape_user_timeline(self, username: str, start_date: datetime,
                            end_date: datetime, include_replies: bool = False) -> List[Post]:
        """
        Scrape tweets from a specific user's timeline.
        
        Args:
            username: Twitter username (without @)
            start_date: Start date
            end_date: End date
            include_replies: Whether to include replies
            
        Returns:
            List of posts from user
        """
        logger.info(f"Scraping timeline for @{username}")
        
        posts = []
        tweet_count = 0
        
        try:
            # Create user scraper
            scraper = sntwitter.TwitterUserScraper(username)
            
            for tweet in scraper.get_items():
                # Rate limiting
                self.rate_limiter.wait_if_needed()
                
                # Check date range
                if not self._is_within_date_range(tweet.date, start_date, end_date):
                    if tweet.date < start_date:
                        break  # Tweets are in reverse chronological order
                    continue
                
                # Skip replies if not requested
                if not include_replies and getattr(tweet, 'inReplyToTweetId', None):
                    continue
                
                # Convert to Post
                post = self._tweet_to_post(tweet, f"@{username}")
                posts.append(post)
                
                tweet_count += 1
                
                if self.max_tweets_per_query and tweet_count >= self.max_tweets_per_query:
                    break
            
            logger.info(f"Collected {tweet_count} tweets from @{username}")
            
        except Exception as e:
            logger.error(f"Error scraping user timeline @{username}: {e}")
            raise
        
        return posts
    
    def scrape_conversation(self, tweet_id: str) -> List[Post]:
        """
        Scrape an entire conversation thread.
        
        Args:
            tweet_id: ID of a tweet in the conversation
            
        Returns:
            List of posts in the conversation
        """
        logger.info(f"Scraping conversation for tweet {tweet_id}")
        
        # This would require more complex implementation
        # For now, return empty list as placeholder
        logger.warning("Conversation scraping not yet implemented")
        return []