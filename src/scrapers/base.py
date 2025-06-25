"""
Base scraper class and data structures.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from threading import Lock

logger = logging.getLogger(__name__)

@dataclass
class Post:
    """
    Unified post structure across all platforms.
    Represents a single social media post with all relevant metadata.
    """
    id: str
    platform: str
    author: str
    author_id: Optional[str]
    content: str
    title: Optional[str]
    timestamp: datetime
    engagement: Dict[str, int]  # likes, shares, comments, etc.
    url: str
    query: str  # Which search query found this post
    metadata: Dict[str, Any] = field(default_factory=dict)  # Platform-specific data
    
    def __post_init__(self):
        """Validate and normalize post data."""
        # Ensure engagement is a dict
        if not isinstance(self.engagement, dict):
            self.engagement = {}
        
        # Ensure metadata is a dict
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        
        # Ensure timestamp is datetime
        if isinstance(self.timestamp, str):
            try:
                self.timestamp = datetime.fromisoformat(self.timestamp)
            except:
                logger.warning(f"Failed to parse timestamp: {self.timestamp}")
                self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert post to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of the post
        """
        return {
            'id': self.id,
            'platform': self.platform,
            'author': self.author,
            'author_id': self.author_id,
            'content': self.content,
            'title': self.title,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'engagement': self.engagement,
            'url': self.url,
            'query': self.query,
            'metadata': self.metadata
        }
    
    def get_engagement_score(self) -> int:
        """
        Calculate total engagement score.
        
        Returns:
            Sum of all engagement metrics
        """
        if self.engagement is None:
            return 0
        return sum(self.engagement.values())
    
    def __str__(self) -> str:
        """String representation of post."""
        content_preview = self.content[:50] + '...' if len(self.content) > 50 else self.content
        return f"Post({self.platform}:{self.id}, @{self.author}, '{content_preview}')"
    
    def __repr__(self) -> str:
        """Detailed representation of post."""
        return f"Post(id={self.id!r}, platform={self.platform!r}, author={self.author!r})"


class BaseScraper(ABC):
    """
    Abstract base class for platform scrapers.
    All platform-specific scrapers must inherit from this class.
    """
    
    def __init__(self, rate_limiter: 'RateLimiter', max_workers: Optional[int] = None):
        """
        Initialize base scraper.

        Args:
            rate_limiter: Rate limiter instance
        """
        self.rate_limiter = rate_limiter
        self.max_workers = max_workers or max(1, (os.cpu_count() or 1))
        self._lock = Lock()
        self.posts_collected = 0
        self.errors_count = 0
        self.last_error = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def scrape(self, query: str, start_date: datetime, end_date: datetime) -> List[Post]:
        """
        Scrape posts for a query within date range.
        
        Args:
            query: Search query string
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            List of Post objects
        """
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """
        Test if scraper can connect to platform.
        
        Returns:
            True if connection is valid, False otherwise
        """
        pass
    
    def scrape_multiple(self, queries: List[str], start_date: datetime, 
                       end_date: datetime) -> List[Post]:
        """
        Scrape multiple queries and combine results.
        
        Args:
            queries: List of search queries
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            Combined list of posts from all queries
        """
        all_posts = []

        def run_task(args):
            i, query = args
            self.logger.info(f"Scraping query {i}/{len(queries)}: '{query}'")

            try:
                posts = self.scrape(query, start_date, end_date)
                self.logger.info(f"Collected {len(posts)} posts for '{query}'")
                return posts
            except Exception as e:
                with self._lock:
                    self.errors_count += 1
                    self.last_error = str(e)
                self.logger.error(f"Error scraping '{query}': {e}")
                return []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(run_task, enumerate(queries, 1))
            for posts in results:
                all_posts.extend(posts)

        self.logger.info(
            f"Total posts collected: {len(all_posts)} from {len(queries)} queries")

        return all_posts
    
    def reset_stats(self):
        """Reset collection statistics."""
        self.posts_collected = 0
        self.errors_count = 0
        self.last_error = None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get scraper statistics.
        
        Returns:
            Dictionary with scraper stats
        """
        return {
            'posts_collected': self.posts_collected,
            'errors_count': self.errors_count,
            'last_error': self.last_error,
            'scraper_type': self.__class__.__name__
        }
    
    def _validate_date_range(self, start_date: datetime, end_date: datetime):
        """
        Validate date range parameters.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Raises:
            ValueError: If date range is invalid
        """
        if start_date >= end_date:
            raise ValueError(f"Start date must be before end date: {start_date} >= {end_date}")
        
        if end_date > datetime.now():
            self.logger.warning(f"End date is in the future: {end_date}. Using current time.")
        
        # Check if range is too large (more than 1 year)
        if (end_date - start_date).days > 365:
            self.logger.warning("Date range exceeds 365 days. Results may be limited.")
    
    def _clean_query(self, query: str) -> str:
        """
        Clean and normalize search query.
        
        Args:
            query: Raw query string
            
        Returns:
            Cleaned query string
        """
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Remove dangerous characters that might break searches
        query = query.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        
        return query.strip()