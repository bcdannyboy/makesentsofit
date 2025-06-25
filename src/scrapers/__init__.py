"""
Scraper modules for social media platforms.
"""
from typing import Dict, List
import logging

from .base import BaseScraper, Post
from .twitter import TwitterScraper
from .reddit import RedditScraper
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

__all__ = ['BaseScraper', 'Post', 'TwitterScraper', 'RedditScraper', 
           'RateLimiter', 'create_scrapers']

def create_scrapers(
    platforms: List[str], 
    config: 'Config'
) -> Dict[str, BaseScraper]:
    """
    Create scraper instances for specified platforms.
    
    Args:
        platforms: List of platform names
        config: Application configuration
        
    Returns:
        Dictionary mapping platform names to scraper instances
    """
    scrapers = {}
    
    for platform in platforms:
        platform = platform.lower()
        
        if platform == 'twitter':
            rate_limiter = RateLimiter(config.get_rate_limit('twitter'))
            try:
                scrapers['twitter'] = TwitterScraper(rate_limiter)
                logger.debug(
                    f"Created Twitter scraper with rate limit: {config.get_rate_limit('twitter')}/min")
            except ImportError as e:
                logger.warning(f"Skipping Twitter scraper: {e}")
                continue
            
        elif platform == 'reddit':
            rate_limiter = RateLimiter(config.get_rate_limit('reddit'))
            subreddits = getattr(config, 'reddit_subreddits', ['all'])
            try:
                scrapers['reddit'] = RedditScraper(rate_limiter, config, subreddits)
                logger.debug(f"Created Reddit scraper for subreddits: {subreddits}")
            except ImportError as e:
                logger.warning(f"Skipping Reddit scraper: {e}")
                continue
            
        else:
            logger.warning(f"Unknown platform: {platform}")
    
    return scrapers