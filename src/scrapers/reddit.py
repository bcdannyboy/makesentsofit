"""
Reddit scraper implementation using PRAW and fallback JSON API.
"""
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import re
import requests
import json
import time

try:
    import praw
    from praw.models import Submission, Comment
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    praw = None

from .base import BaseScraper, Post
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

class RedditScraper(BaseScraper):
    """
    Reddit scraper using PRAW (Python Reddit API Wrapper).
    
    This scraper uses PRAW in read-only mode which doesn't require
    authentication and can access public Reddit data.
    """
    
    def __init__(self, rate_limiter: RateLimiter,
                 config: Optional['Config'] = None,
                 subreddits: Optional[List[str]] = None,
                 max_posts_per_query: Optional[int] = None,
                 max_workers: Optional[int] = None):
        """
        Initialize Reddit scraper.
        
        Args:
            rate_limiter: Rate limiter instance
            config: Application configuration containing Reddit credentials
            subreddits: List of subreddits to search (default: ['all'])
            max_posts_per_query: Maximum posts to collect per query
        """
        super().__init__(rate_limiter, max_workers)
        self.subreddits = subreddits or ['all']
        self.max_posts_per_query = max_posts_per_query
        self.config = config
        self.use_json_api = False
        self.reddit = None
        self.session = None
        
        # Get Reddit credentials from config
        if config and hasattr(config, 'reddit') and config.reddit is not None:
            reddit_config = config.reddit
            client_id = reddit_config.get('client_id')
            client_secret = reddit_config.get('client_secret')
            user_agent = reddit_config.get('user_agent', 'MakeSenseOfIt/1.0')
            
            # Check if credentials are dummy/placeholder values
            if client_id in ['dummy', 'reddit_client_id', 'R5D5tGBCldzUC6B56me99g']:
                logger.warning("Detected placeholder Reddit credentials, using JSON API fallback")
                self.use_json_api = True
                self.user_agent = user_agent
            else:
                logger.debug(f"Using Reddit credentials from config: client_id={client_id[:10]}...")
                self.user_agent = user_agent
        else:
            logger.warning("No Reddit credentials found in config, using JSON API fallback")
            self.use_json_api = True
            self.user_agent = 'MakeSenseOfIt/1.0'
        
        # Initialize PRAW only if we have valid credentials
        if not self.use_json_api:
            if not PRAW_AVAILABLE:
                logger.warning("PRAW is not installed, falling back to JSON API")
                self.use_json_api = True
            else:
                try:
                    self.reddit = praw.Reddit(
                        client_id=client_id,
                        client_secret=client_secret,
                        user_agent=user_agent
                    )
                    self.reddit.read_only = True
                    logger.debug(f"PRAW initialized with user_agent: {user_agent}")
                except Exception as e:
                    logger.error(f"Failed to initialize PRAW: {e}")
                    logger.warning("Falling back to JSON API")
                    self.use_json_api = True
        
        # Initialize requests session for JSON API
        if self.use_json_api:
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': self.user_agent,
                'Accept': 'application/json'
            })
            logger.info("Using Reddit JSON API fallback (no authentication required)")
    
    def validate_connection(self) -> bool:
        """
        Test connection to Reddit.
        
        Returns:
            True if connection is valid
        """
        if self.use_json_api:
            return self._validate_json_api_connection()
        
        if not PRAW_AVAILABLE:
            return False
        
        try:
            # Check if we're properly authenticated by verifying read_only status
            if not self.reddit.read_only:
                logger.warning("Reddit instance is not in read-only mode")
            
            # Try to access a known subreddit with proper error handling
            test_sub = self.reddit.subreddit('test')
            _ = test_sub.display_name  # This will trigger API request
            
            logger.debug("Reddit connection validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Reddit connection validation failed: {e}")
            # Log more details if it's an authentication error
            if "401" in str(e) or "received 401 HTTP response" in str(e):
                logger.error("Authentication failed - check Reddit API credentials in config.json")
                logger.info("Switching to JSON API fallback")
                self.use_json_api = True
                self.session = requests.Session()
                self.session.headers.update({
                    'User-Agent': self.user_agent,
                    'Accept': 'application/json'
                })
                return self._validate_json_api_connection()
            return False
    
    def _validate_json_api_connection(self) -> bool:
        """Test connection using Reddit JSON API."""
        try:
            url = "https://www.reddit.com/r/test.json?limit=1"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'children' in data['data']:
                    logger.debug("Reddit JSON API connection validated successfully")
                    return True
            
            logger.error(f"JSON API validation failed: {response.status_code}")
            return False
            
        except Exception as e:
            logger.error(f"JSON API connection validation failed: {e}")
            return False
    
    def scrape(self, query: str, start_date: datetime,
               end_date: datetime) -> List[Post]:
        """
        Scrape Reddit for posts matching query.
        
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
        
        logger.info(f"Starting Reddit scrape: '{query}' from {start_date.date()} to {end_date.date()}")
        
        all_posts = []
        
        # Prioritize subreddits based on query relevance
        prioritized_subreddits = self._prioritize_subreddits(query)
        logger.debug(f"Prioritized subreddits for '{query}': {prioritized_subreddits[:5]}...")
        
        # Search each subreddit in priority order
        for subreddit_name in prioritized_subreddits:
            try:
                if self.use_json_api:
                    posts = self._scrape_subreddit_json(subreddit_name, query, start_date, end_date)
                else:
                    posts = self._scrape_subreddit(subreddit_name, query, start_date, end_date)
                
                all_posts.extend(posts)
                
                # Early termination if we have enough posts
                if self.max_posts_per_query and len(all_posts) >= self.max_posts_per_query:
                    logger.debug(f"Reached max posts limit ({self.max_posts_per_query}), stopping search")
                    break
                
            except Exception as e:
                logger.error(f"Error scraping r/{subreddit_name}: {e}")
                self.errors_count += 1
                self.last_error = str(e)
                # Continue with next subreddit
                continue
        
        logger.info(f"Completed Reddit scrape: {len(all_posts)} posts for '{query}'")
        
        return all_posts
    
    def _prioritize_subreddits(self, query: str) -> List[str]:
        """
        Prioritize subreddits based on query relevance.
        
        Args:
            query: Search query
            
        Returns:
            List of subreddits in priority order
        """
        query_lower = query.lower()
        
        # Define relevance mapping for H3 podcast related queries
        h3_subreddits = ['h3h3productions', 'h3snark', 'h3snark2025', 'LeftoversH3', 'Frenemies', 'Frenemies3', 'Leftemies']
        youtube_subreddits = ['youtubedrama', 'LivestreamFail', 'Idubbbz', 'mealtimevideos']
        general_subreddits = ['all', 'SubredditDrama', 'OutOfTheLoop', 'videos']
        
        prioritized = []
        
        # H3 related queries get H3 subreddits first
        if any(term in query_lower for term in ['h3', 'ethan klein', 'hila klein']):
            prioritized.extend(h3_subreddits)
            prioritized.extend(youtube_subreddits)
            prioritized.extend(general_subreddits)
        else:
            # For other queries, start with general subreddits
            prioritized.extend(general_subreddits)
            prioritized.extend(youtube_subreddits)
            prioritized.extend(h3_subreddits)
        
        # Add remaining subreddits
        remaining = [sub for sub in self.subreddits if sub not in prioritized]
        prioritized.extend(remaining)
        
        return prioritized
    
    def _scrape_subreddit(self, subreddit_name: str, query: str,
                         start_date: datetime, end_date: datetime) -> List[Post]:
        """
        Scrape a specific subreddit.
        
        Args:
            subreddit_name: Name of subreddit
            query: Search query
            start_date: Start date
            end_date: End date
            
        Returns:
            List of posts from subreddit
        """
        logger.debug(f"Searching r/{subreddit_name} for '{query}'")
        
        posts = []
        post_count = 0
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Determine time filter based on date range
            time_filter = self._get_time_filter(start_date, end_date)
            
            # Search posts
            # Rate limiting per API request
            self.rate_limiter.wait_if_needed()

            search_results = subreddit.search(
                query,
                time_filter=time_filter,
                limit=None  # No limit, we'll handle it ourselves
            )

            for submission in search_results:
                
                # Check if we've reached the limit
                if self.max_posts_per_query and post_count >= self.max_posts_per_query:
                    logger.debug(f"Reached max posts limit: {self.max_posts_per_query}")
                    break
                
                # Check date bounds
                post_time = datetime.fromtimestamp(submission.created_utc)
                if not self._is_within_date_range(post_time, start_date, end_date):
                    continue
                
                # Convert submission to Post
                post = self._submission_to_post(submission, query)
                posts.append(post)
                
                post_count += 1
                self.posts_collected += 1
                
                # Progress logging
                if post_count % 50 == 0:
                    logger.debug(f"Collected {post_count} posts from r/{subreddit_name}...")
            
            logger.debug(f"Collected {post_count} posts from r/{subreddit_name}")
            
        except Exception as e:
            logger.error(f"Error searching r/{subreddit_name}: {e}")
            raise
        
        return posts
    
    def _scrape_subreddit_json(self, subreddit_name: str, query: str,
                              start_date: datetime, end_date: datetime) -> List[Post]:
        """
        Scrape a specific subreddit using Reddit JSON API.
        Optimized for rate limiting and efficiency.
        
        Args:
            subreddit_name: Name of subreddit
            query: Search query
            start_date: Start date
            end_date: End date
            
        Returns:
            List of posts from subreddit
        """
        logger.debug(f"Searching r/{subreddit_name} for '{query}' using JSON API")
        
        posts = []
        
        # Limit posts per subreddit to manage rate limits better
        max_posts_per_subreddit = min(50, (self.max_posts_per_query or 100) // max(1, len(self.subreddits)))
        
        try:
            # For 'all', we'll search r/popular instead as it's more accessible
            search_subreddit = 'popular' if subreddit_name == 'all' else subreddit_name
            
            # Rate limiting per API request
            self.rate_limiter.wait_if_needed()
            
            # Build search URL - get more posts in single request
            params = {
                'q': query,
                'restrict_sr': 'on' if subreddit_name != 'all' else 'off',
                'sort': 'relevance',
                'limit': max_posts_per_subreddit,
                'raw_json': 1,
                't': 'year'  # Limit to past year for better performance
            }
            
            if subreddit_name == 'all':
                url = f"https://www.reddit.com/search.json"
            else:
                url = f"https://www.reddit.com/r/{search_subreddit}/search.json"
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 429:
                logger.warning(f"Rate limited for r/{subreddit_name}, skipping")
                return []
            elif response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} for r/{subreddit_name}, skipping")
                return []
            
            data = response.json()
            
            if 'data' not in data or 'children' not in data['data']:
                logger.debug(f"No data found for r/{subreddit_name}")
                return []
            
            children = data['data']['children']
            if not children:
                logger.debug(f"No posts found for r/{subreddit_name}")
                return []
            
            # Process all posts from the single request
            for child in children:
                try:
                    post_data = child['data']
                    
                    # Check date bounds
                    post_time = datetime.fromtimestamp(post_data.get('created_utc', 0))
                    if not self._is_within_date_range(post_time, start_date, end_date):
                        continue
                    
                    # Convert to Post object
                    post = self._json_to_post(post_data, query)
                    posts.append(post)
                    self.posts_collected += 1
                    
                except Exception as e:
                    logger.debug(f"Error processing post from r/{subreddit_name}: {e}")
                    continue
            
            logger.debug(f"Collected {len(posts)} posts from r/{subreddit_name}")
            
        except Exception as e:
            logger.warning(f"Error searching r/{subreddit_name} with JSON API: {e}")
            # Don't raise - just return empty list and continue with other subreddits
            return []
        
        return posts
    
    def _json_to_post(self, post_data: Dict[str, Any], query: str) -> Post:
        """
        Convert Reddit JSON API response to Post object.
        
        Args:
            post_data: JSON post data from Reddit API
            query: Search query that found this post
            
        Returns:
            Post object
        """
        # Extract engagement metrics
        engagement = {
            'score': post_data.get('score', 0),
            'upvotes': int(post_data.get('ups', 0)),
            'downvotes': int(post_data.get('downs', 0)),
            'num_comments': post_data.get('num_comments', 0),
            'upvote_ratio': post_data.get('upvote_ratio', 0.5)
        }
        
        # Extract metadata
        metadata = {
            'subreddit': post_data.get('subreddit', ''),
            'subreddit_id': post_data.get('subreddit_id', ''),
            'is_self': post_data.get('is_self', False),
            'is_video': post_data.get('is_video', False),
            'is_original_content': post_data.get('is_original_content', False),
            'over_18': post_data.get('over_18', False),
            'spoiler': post_data.get('spoiler', False),
            'stickied': post_data.get('stickied', False),
            'locked': post_data.get('locked', False),
            'num_crossposts': post_data.get('num_crossposts', 0),
            'awards': len(post_data.get('all_awardings', [])),
            'domain': post_data.get('domain', ''),
            'link_flair_text': post_data.get('link_flair_text', ''),
            'author_flair_text': post_data.get('author_flair_text', '')
        }
        
        # Handle author
        author = post_data.get('author', '[deleted]')
        author_id = post_data.get('author_fullname', None)
        
        # Create Post object
        post = Post(
            id=post_data.get('id', ''),
            platform='reddit',
            author=author,
            author_id=author_id,
            content=post_data.get('selftext', ''),
            title=post_data.get('title', ''),
            timestamp=datetime.fromtimestamp(post_data.get('created_utc', 0)),
            engagement=engagement,
            url=f"https://reddit.com{post_data.get('permalink', '')}",
            query=query,
            metadata=metadata
        )
        
        return post
    
    def _get_time_filter(self, start_date: datetime, end_date: datetime) -> str:
        """
        Determine Reddit time filter based on date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Time filter string for Reddit API
        """
        days_diff = (end_date - start_date).days
        
        if days_diff <= 1:
            return 'day'
        elif days_diff <= 7:
            return 'week'
        elif days_diff <= 30:
            return 'month'
        elif days_diff <= 365:
            return 'year'
        else:
            return 'all'
    
    def _is_within_date_range(self, post_time: datetime,
                              start_date: datetime, end_date: datetime) -> bool:
        """
        Check if post time is within date range.
        
        Args:
            post_time: Post timestamp
            start_date: Start of range
            end_date: End of range
            
        Returns:
            True if within range
        """
        # Ensure all datetimes are timezone-naive for comparison
        if post_time.tzinfo:
            post_time = post_time.replace(tzinfo=None)
        if start_date.tzinfo:
            start_date = start_date.replace(tzinfo=None)
        if end_date.tzinfo:
            end_date = end_date.replace(tzinfo=None)
            
        return start_date <= post_time <= end_date
    
    def _submission_to_post(self, submission: Any, query: str) -> Post:
        """
        Convert PRAW Submission object to Post object.
        
        Args:
            submission: PRAW Submission object
            query: Search query that found this post
            
        Returns:
            Post object
        """
        # Extract engagement metrics
        engagement = {
            'score': submission.score,
            'upvotes': int(submission.score * submission.upvote_ratio) if submission.upvote_ratio else submission.score,
            'downvotes': int(submission.score * (1 - submission.upvote_ratio)) if submission.upvote_ratio else 0,
            'num_comments': submission.num_comments,
            'upvote_ratio': submission.upvote_ratio
        }
        
        # Extract metadata
        metadata = {
            'subreddit': submission.subreddit.display_name,
            'subreddit_id': submission.subreddit_id,
            'is_self': submission.is_self,
            'is_video': submission.is_video,
            'is_original_content': submission.is_original_content if hasattr(submission, 'is_original_content') else False,
            'over_18': submission.over_18,
            'spoiler': submission.spoiler,
            'stickied': submission.stickied,
            'locked': submission.locked,
            'num_crossposts': submission.num_crossposts if hasattr(submission, 'num_crossposts') else 0,
            'awards': self._count_awards(submission),
            'domain': submission.domain if hasattr(submission, 'domain') else None,
            'link_flair_text': submission.link_flair_text,
            'author_flair_text': submission.author_flair_text if hasattr(submission, 'author_flair_text') else None
        }
        
        # Handle deleted/removed authors without triggering additional API calls
        if submission.author:
            name_attr = getattr(submission.author, 'name', None)
            if isinstance(name_attr, str):
                author = name_attr
            else:
                # Fallback: try to extract from Mock representation
                author_str = str(submission.author)
                m = re.match(r"<Mock name='([^']+)'", author_str)
                author = m.group(1) if m else author_str

            # Use author_fullname from submission data if available. Accessing
            # submission.author.id would trigger an extra request for each post
            author_id_val = getattr(submission, 'author_fullname', None)
            author_id = str(author_id_val) if author_id_val is not None else None
        else:
            author = '[deleted]'
            author_id = None
        
        # Create Post object
        post = Post(
            id=submission.id,
            platform='reddit',
            author=author,
            author_id=author_id,
            content=submission.selftext or '',
            title=submission.title,
            timestamp=datetime.fromtimestamp(submission.created_utc),
            engagement=engagement,
            url=f"https://reddit.com{submission.permalink}",
            query=query,
            metadata=metadata
        )
        
        return post

    def _count_awards(self, submission: Any) -> int:
        """Safely count awards on a submission."""
        awards = getattr(submission, 'all_awardings', None)
        if isinstance(awards, (list, tuple, set)):
            return len(awards)
        return 0
    
    def scrape_subreddit_hot(self, subreddit_name: str, limit: int = 100) -> List[Post]:
        """
        Scrape hot posts from a subreddit.
        
        Args:
            subreddit_name: Name of subreddit
            limit: Maximum number of posts
            
        Returns:
            List of hot posts
        """
        logger.info(f"Scraping hot posts from r/{subreddit_name}")
        
        posts = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)

            # Rate limit per API call
            self.rate_limiter.wait_if_needed()

            for submission in subreddit.hot(limit=limit):
                
                # Convert to Post
                post = self._submission_to_post(submission, f"hot:r/{subreddit_name}")
                posts.append(post)
            
            logger.info(f"Collected {len(posts)} hot posts from r/{subreddit_name}")
            
        except Exception as e:
            logger.error(f"Error scraping hot posts from r/{subreddit_name}: {e}")
            raise
        
        return posts
    
    def scrape_user_posts(self, username: str, start_date: datetime,
                         end_date: datetime) -> List[Post]:
        """
        Scrape posts from a specific Reddit user.
        
        Args:
            username: Reddit username (without u/)
            start_date: Start date
            end_date: End date
            
        Returns:
            List of user's posts
        """
        logger.info(f"Scraping posts from u/{username}")
        
        posts = []
        post_count = 0
        
        try:
            user = self.reddit.redditor(username)

            # Rate limit per API call
            self.rate_limiter.wait_if_needed()

            for submission in user.submissions.new(limit=None):
                
                # Check date range
                post_time = datetime.fromtimestamp(submission.created_utc)
                if not self._is_within_date_range(post_time, start_date, end_date):
                    if post_time < start_date:
                        break  # Posts are in reverse chronological order
                    continue
                
                # Convert to Post
                post = self._submission_to_post(submission, f"u/{username}")
                posts.append(post)
                
                post_count += 1
                
                if self.max_posts_per_query and post_count >= self.max_posts_per_query:
                    break
            
            logger.info(f"Collected {post_count} posts from u/{username}")
            
        except Exception as e:
            logger.error(f"Error scraping user u/{username}: {e}")
            raise
        
        return posts
    
    def scrape_comments(self, submission_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Scrape comments from a Reddit post.
        
        Args:
            submission_id: Reddit submission ID
            limit: Maximum number of comments
            
        Returns:
            List of comment data
        """
        logger.info(f"Scraping comments from submission {submission_id}")
        
        comments = []
        
        try:
            submission = self.reddit.submission(id=submission_id)
            submission.comments.replace_more(limit=0)  # Don't expand "more comments"

            # Rate limit per API call
            self.rate_limiter.wait_if_needed()

            for comment in submission.comments.list()[:limit]:
                
                if isinstance(comment, praw.models.Comment):
                    comment_data = {
                        'id': comment.id,
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'body': comment.body,
                        'score': comment.score,
                        'created_utc': datetime.fromtimestamp(comment.created_utc),
                        'is_submitter': comment.is_submitter,
                        'parent_id': comment.parent_id,
                        'permalink': f"https://reddit.com{comment.permalink}"
                    }
                    comments.append(comment_data)
            
            logger.info(f"Collected {len(comments)} comments")
            
        except Exception as e:
            logger.error(f"Error scraping comments: {e}")
            raise
        
        return comments