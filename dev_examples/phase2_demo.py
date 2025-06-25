#!/usr/bin/env python3
"""
Phase 2 Demo: Data Collection
Demonstrates the scraping functionality of MakeSenseOfIt.
"""
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.scrapers import create_scrapers
from src.logger import setup_logging, get_logger


def demo_basic_scraping():
    """Demonstrate basic scraping functionality."""
    print("=== Phase 2 Demo: Basic Scraping ===\n")
    
    # Setup
    setup_logging(verbose=True)
    logger = get_logger(__name__)
    config = Config()
    
    # Parameters
    platforms = ['twitter', 'reddit']
    queries = ['python programming', 'machine learning']
    days_back = 1
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"🔍 Queries: {', '.join(queries)}")
    print(f"🌐 Platforms: {', '.join(platforms)}\n")
    
    # Create scrapers
    scrapers = create_scrapers(platforms, config)
    
    # Test connections
    print("🔌 Testing connections:")
    valid_scrapers = {}
    for platform, scraper in scrapers.items():
        if scraper.validate_connection():
            print(f"  ✅ {platform}: Connected")
            valid_scrapers[platform] = scraper
        else:
            print(f"  ❌ {platform}: Connection failed")
    
    if not valid_scrapers:
        print("\n❌ No valid scrapers available!")
        return
    
    # Collect data
    print("\n📡 Collecting data...")
    all_posts = []
    
    for platform, scraper in valid_scrapers.items():
        print(f"\n📥 Scraping {platform}...")
        
        # Limit posts for demo
        scraper.max_posts_per_query = 5
        
        try:
            posts = scraper.scrape_multiple(queries, start_date, end_date)
            all_posts.extend(posts)
            print(f"  ✅ Collected {len(posts)} posts from {platform}")
            
            # Show sample post
            if posts:
                sample = posts[0]
                print(f"\n  Sample post from {platform}:")
                print(f"    Author: @{sample.author}")
                print(f"    Content: {sample.content[:100]}...")
                print(f"    Engagement: {sample.get_engagement_score()} total")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Summary
    print(f"\n📊 Total posts collected: {len(all_posts)}")
    
    # Platform breakdown
    platform_counts = {}
    for post in all_posts:
        platform_counts[post.platform] = platform_counts.get(post.platform, 0) + 1
    
    print("\n📈 Posts by platform:")
    for platform, count in platform_counts.items():
        print(f"  • {platform}: {count}")
    
    # Query breakdown
    query_counts = {}
    for post in all_posts:
        query_counts[post.query] = query_counts.get(post.query, 0) + 1
    
    print("\n🔍 Posts by query:")
    for query, count in query_counts.items():
        print(f"  • '{query}': {count}")
    
    return all_posts


def demo_rate_limiting():
    """Demonstrate rate limiting functionality."""
    print("\n\n=== Phase 2 Demo: Rate Limiting ===\n")
    
    from src.scrapers.rate_limiter import RateLimiter
    import time
    
    # Create rate limiter with low limit for demo
    limiter = RateLimiter(calls_per_minute=30)  # 0.5 calls per second
    
    print("⏱️  Testing rate limiter (30 calls/minute)...")
    print("Making 5 rapid API calls:\n")
    
    for i in range(5):
        start = time.time()
        wait_time = limiter.wait_if_needed()
        elapsed = time.time() - start
        
        print(f"Call {i+1}: Waited {wait_time:.3f}s (total: {elapsed:.3f}s)")
        
        # Show stats
        stats = limiter.get_stats()
        print(f"  Current rate: {stats['current_rate']:.1f}/min")
        print(f"  Remaining: {stats['remaining_calls']} calls\n")
    
    print("✅ Rate limiting prevents exceeding API limits!")


def demo_error_handling():
    """Demonstrate error handling and resilience."""
    print("\n\n=== Phase 2 Demo: Error Handling ===\n")
    
    from src.scrapers.base import BaseScraper
    from src.scrapers.rate_limiter import RateLimiter
    
    class FaultyScraper(BaseScraper):
        """Scraper that simulates errors."""
        
        def validate_connection(self):
            return True
        
        def scrape(self, query, start_date, end_date):
            if 'error' in query.lower():
                raise Exception(f"Simulated error for query: {query}")
            return []
    
    # Create scraper
    scraper = FaultyScraper(RateLimiter())
    
    # Test with mixed queries
    queries = ['good query', 'error query', 'another good query']
    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()
    
    print("🧪 Testing error resilience with queries:")
    for q in queries:
        print(f"  • {q}")
    
    print("\n📡 Scraping...")
    posts = scraper.scrape_multiple(queries, start_date, end_date)
    
    print(f"\n✅ Completed despite errors!")
    print(f"  • Successful queries: {3 - scraper.errors_count}")
    print(f"  • Failed queries: {scraper.errors_count}")
    print(f"  • Last error: {scraper.last_error}")


def demo_post_analysis():
    """Demonstrate post data analysis."""
    print("\n\n=== Phase 2 Demo: Post Analysis ===\n")
    
    # Use collected posts from basic demo
    posts = demo_basic_scraping()
    
    if not posts:
        print("No posts to analyze!")
        return
    
    print(f"\n\n📊 Analyzing {len(posts)} posts...\n")
    
    # Time distribution
    print("📅 Posts by hour:")
    hour_counts = {}
    for post in posts:
        hour = post.timestamp.hour
        hour_counts[hour] = hour_counts.get(hour, 0) + 1
    
    for hour in sorted(hour_counts.keys()):
        print(f"  {hour:02d}:00 - {'█' * hour_counts[hour]} ({hour_counts[hour]})")
    
    # Engagement stats
    print("\n💬 Engagement statistics:")
    if posts:
        engagements = [p.get_engagement_score() for p in posts]
        print(f"  • Average: {sum(engagements) / len(engagements):.1f}")
        print(f"  • Maximum: {max(engagements)}")
        print(f"  • Minimum: {min(engagements)}")
    
    # Top posts
    print("\n🏆 Top 3 posts by engagement:")
    top_posts = sorted(posts, key=lambda p: p.get_engagement_score(), reverse=True)[:3]
    
    for i, post in enumerate(top_posts, 1):
        print(f"\n{i}. {post.platform} by @{post.author}")
        print(f"   Score: {post.get_engagement_score()}")
        print(f"   Content: {post.content[:100]}...")


if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════╗
    ║     MakeSenseOfIt - Phase 2 Demo         ║
    ║        Data Collection Testing           ║
    ╚══════════════════════════════════════════╝
    """)
    
    # Run demos
    try:
        # Basic scraping
        posts = demo_basic_scraping()
        
        # Rate limiting
        demo_rate_limiting()
        
        # Error handling
        demo_error_handling()
        
        # Post analysis (if we have posts)
        if posts:
            demo_post_analysis()
        
        print("\n\n✅ Phase 2 demo completed successfully!")
        print("📝 Next: Phase 3 - Sentiment Analysis")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()