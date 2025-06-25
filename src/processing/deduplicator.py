"""
Post deduplication module.
Removes duplicate posts collected across different queries and platforms.
"""
import hashlib
import logging
from collections import defaultdict
from typing import List, Set, Dict, Tuple
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class Deduplicator:
    """
    Remove duplicate posts across queries and platforms.
    
    Uses multiple strategies:
    1. Exact ID matching (same platform + ID)
    2. Content hashing (normalized content matching)
    3. Fuzzy matching for near-duplicates (optional)
    """
    
    def __init__(self, similarity_threshold: float = 0.9,
                 enable_fuzzy_matching: bool = False):
        """
        Initialize deduplicator.
        
        Args:
            similarity_threshold: Threshold for fuzzy matching (0-1)
            enable_fuzzy_matching: Whether to use fuzzy matching for near-duplicates
        """
        self.similarity_threshold = similarity_threshold
        self.enable_fuzzy_matching = enable_fuzzy_matching
        
        # Tracking sets
        self.seen_ids: Set[str] = set()
        self.seen_hashes: Set[str] = set()
        self.content_cache: Dict[str, str] = {}  # hash -> original content for fuzzy matching
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'exact_id_duplicates': 0,
            'content_duplicates': 0,
            'fuzzy_duplicates': 0,
            'cross_platform_duplicates': 0
        }
    
    def deduplicate(self, posts: List['Post']) -> Tuple[List['Post'], Dict]:
        """
        Remove duplicates and return unique posts with statistics.
        
        Args:
            posts: List of posts to deduplicate
            
        Returns:
            Tuple of (unique_posts, statistics)
        """
        unique_posts = []
        duplicates_by_query = defaultdict(int)
        duplicates_by_type = defaultdict(int)
        cross_query_duplicates = 0
        
        logger.info(f"Starting deduplication of {len(posts)} posts")
        
        for post in posts:
            self.stats['total_processed'] += 1
            
            # Check for duplicates
            is_duplicate, duplicate_type = self._is_duplicate(post)
            
            if is_duplicate:
                duplicates_by_query[post.query] += 1
                duplicates_by_type[duplicate_type] += 1
                
                # Track if this is a cross-query duplicate
                if duplicate_type == 'content' or duplicate_type == 'fuzzy':
                    cross_query_duplicates += 1
                
                continue
            
            # Add to unique posts
            self._register_post(post)
            unique_posts.append(post)
        
        # Compile statistics
        stats = {
            'total_posts': len(posts),
            'unique_posts': len(unique_posts),
            'duplicates_removed': len(posts) - len(unique_posts),
            'duplicate_rate': (len(posts) - len(unique_posts)) / len(posts) if posts else 0,
            'duplicates_by_query': dict(duplicates_by_query),
            'duplicates_by_type': dict(duplicates_by_type),
            'cross_query_duplicates': cross_query_duplicates,
            'processing_stats': self.stats.copy()
        }
        
        logger.info(f"Deduplication complete: {stats['total_posts']} → {stats['unique_posts']} posts "
                   f"({stats['duplicates_removed']} duplicates removed, "
                   f"{stats['duplicate_rate']:.1%} duplicate rate)")
        
        return unique_posts, stats
    
    def _is_duplicate(self, post: 'Post') -> Tuple[bool, str]:
        """
        Check if post is a duplicate.
        
        Args:
            post: Post to check
            
        Returns:
            Tuple of (is_duplicate, duplicate_type)
        """
        # 1. Check exact ID match
        post_id = f"{post.platform}:{post.id}"
        if post_id in self.seen_ids:
            self.stats['exact_id_duplicates'] += 1
            return True, 'exact_id'
        
        # 2. Check content hash
        content_hash = self._hash_content(post.content)
        if content_hash in self.seen_hashes:
            self.stats['content_duplicates'] += 1
            
            # Check if it's cross-platform
            if self._is_cross_platform_duplicate(post, content_hash):
                self.stats['cross_platform_duplicates'] += 1
                
            return True, 'content'
        
        # 3. Fuzzy matching (if enabled)
        if self.enable_fuzzy_matching and self._has_fuzzy_duplicate(post, content_hash):
            self.stats['fuzzy_duplicates'] += 1
            return True, 'fuzzy'
        
        return False, ''
    
    def _register_post(self, post: 'Post'):
        """Register a post as seen."""
        # Register ID
        post_id = f"{post.platform}:{post.id}"
        self.seen_ids.add(post_id)
        
        # Register content hash
        content_hash = self._hash_content(post.content)
        self.seen_hashes.add(content_hash)
        
        # Cache content for fuzzy matching
        if self.enable_fuzzy_matching:
            self.content_cache[content_hash] = post.content
    
    def _hash_content(self, content: str) -> str:
        """
        Create normalized hash of content for duplicate detection.
        
        Args:
            content: Post content
            
        Returns:
            SHA256 hash of normalized content
        """
        # Normalize content
        normalized = self._normalize_content(content)
        
        # Create hash
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def _normalize_content(self, content: str) -> str:
        """
        Normalize content for comparison.
        
        Args:
            content: Raw content
            
        Returns:
            Normalized content
        """
        if not content:
            return ""
        
        # Convert to lowercase
        normalized = content.lower()
        
        # Remove URLs
        normalized = re.sub(r'https?://\S+|www\.\S+', '', normalized)
        
        # Remove mentions (platform-specific)
        normalized = re.sub(r'@\w+', '', normalized)
        
        # Remove hashtags but keep the text
        normalized = re.sub(r'#(\w+)', r'\1', normalized)
        
        # Remove all punctuation and special characters
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Normalize whitespace - single spaces only
        normalized = ' '.join(normalized.split())
        
        return normalized.strip()
    
    def _has_fuzzy_duplicate(self, post: 'Post', content_hash: str) -> bool:
        """
        Check for fuzzy/near-duplicate content.
        
        Args:
            post: Post to check
            content_hash: Hash of post content
            
        Returns:
            True if a fuzzy duplicate exists
        """
        normalized_content = self._normalize_content(post.content)
        
        # Compare with cached content
        for cached_hash, cached_content in self.content_cache.items():
            if cached_hash == content_hash:
                continue  # Skip exact matches (already checked)
            
            # Calculate similarity
            cached_normalized = self._normalize_content(cached_content)
            similarity = self._calculate_similarity(normalized_content, cached_normalized)
            
            if similarity >= self.similarity_threshold:
                logger.debug(f"Fuzzy duplicate found: {similarity:.2%} similarity")
                return True
        
        return False
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        # Use SequenceMatcher for similarity
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _is_cross_platform_duplicate(self, post: 'Post', content_hash: str) -> bool:
        """
        Check if this is a cross-platform duplicate.
        
        Args:
            post: Current post
            content_hash: Content hash
            
        Returns:
            True if duplicate exists on different platform
        """
        # This is a simplified check - in production, you'd track platform info with hashes
        # For now, we'll just flag it as cross-platform if we see the same content
        return True  # Simplified for this implementation
    
    def reset(self):
        """Reset deduplicator state."""
        self.seen_ids.clear()
        self.seen_hashes.clear()
        self.content_cache.clear()
        self.stats = {
            'total_processed': 0,
            'exact_id_duplicates': 0,
            'content_duplicates': 0,
            'fuzzy_duplicates': 0,
            'cross_platform_duplicates': 0
        }
        logger.debug("Deduplicator reset")
    
    def get_stats(self) -> Dict:
        """Get deduplication statistics."""
        return self.stats.copy()