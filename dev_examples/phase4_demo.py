#!/usr/bin/env python3
"""
Phase 4 Demo: Data Processing
Demonstrates the data processing functionality of MakeSenseOfIt.
"""
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.scrapers.base import Post
from src.processing import Deduplicator, DataAggregator, TimeSeriesAnalyzer
from src.logger import setup_logging, get_logger


def create_demo_posts():
    """Create demo posts with various characteristics."""
    posts = []
    base_time = datetime.now()
    
    # Create posts with different patterns
    authors = {
        'positive_influencer': ('POSITIVE', 0.8, 1000),  # (sentiment, ratio, engagement)
        'negative_troll': ('NEGATIVE', 0.9, 50),
        'neutral_observer': ('NEUTRAL', 0.7, 100),
        'mixed_user': ('MIXED', 0.5, 200),
    }
    
    # Different content templates for variety
    content_templates = [
        "Just discovered {} - absolutely mind-blowing! #{}",
        "Why is nobody talking about {}? This needs more attention. {}",
        "Breaking: Major developments in {} sector today. Check this out: {}",
        "{} is overrated. Change my mind. Here's why: {}",
        "Thread 🧵: Everything you need to know about {} in 2025. {}",
        "Unpopular opinion: {} is actually getting worse, not better. {}",
        "Can we please stop pretending {} is revolutionary? It's just {}.",
        "Today I learned something incredible about {}. Did you know {}?",
        "Hot take: {} will be obsolete by next year because {}",
        "The future of {} is here and it's called {}. Game changer!",
    ]
    
    topics = [
        ("AI technology", "artificialintelligence", "machine learning is evolving rapidly"),
        ("cryptocurrency", "crypto", "blockchain adoption is increasing"),
        ("climate change", "climateaction", "renewable energy is the solution"),
        ("space exploration", "space", "Mars colonization is closer than ever"),
        ("quantum computing", "quantum", "solving impossible problems"),
        ("biotechnology", "biotech", "CRISPR changes everything"),
        ("renewable energy", "cleanenergy", "solar costs dropping fast"),
        ("virtual reality", "VR", "metaverse is the next internet"),
        ("electric vehicles", "EVs", "Tesla isn't the only player"),
        ("social media", "socialmedia", "algorithms control everything"),
    ]
    
    # Generate posts over 7 days with varied content
    for day in range(7):
        for hour in [9, 12, 15, 18, 21]:  # Peak hours
            # Use different topic and template combinations
            topic_idx = (day * 5 + hour) % len(topics)
            template_idx = (day + hour) % len(content_templates)
            
            topic, hashtag, detail = topics[topic_idx]
            template = content_templates[template_idx]
            
            for author, (primary_sentiment, ratio, base_engagement) in authors.items():
                # Determine sentiment based on author's tendency
                import random
                if primary_sentiment == 'MIXED':
                    sentiment = random.choice(['POSITIVE', 'NEGATIVE', 'NEUTRAL'])
                else:
                    sentiment = primary_sentiment if random.random() < ratio else random.choice(['POSITIVE', 'NEGATIVE', 'NEUTRAL'])
                
                # Create varied content
                content = template.format(topic, detail)
                
                # Add author-specific flavor
                if author == 'positive_influencer':
                    content += " 🚀💫 #innovation #future"
                elif author == 'negative_troll':
                    content += " This is getting ridiculous. Wake up people!"
                elif author == 'neutral_observer':
                    content += " Interesting development to watch."
                elif author == 'mixed_user':
                    content += " What do you all think?"
                
                # Create post
                post = Post(
                    id=f"{day}_{hour}_{author}_{topic_idx}",
                    platform='twitter' if hour % 2 == 0 else 'reddit',
                    author=author,
                    author_id=f"id_{author}",
                    content=content,
                    title=f"{topic} Discussion - Day {day}" if hour % 2 == 1 else None,
                    timestamp=base_time - timedelta(days=6-day, hours=23-hour),
                    engagement={
                        'likes': base_engagement + random.randint(-50, 200),
                        'shares': base_engagement // 2 + random.randint(-25, 100)
                    },
                    url=f"https://example.com/{day}/{hour}/{author}",
                    query='demo_query',
                    metadata={
                        'hashtags': [hashtag, 'trending'] if hour in [12, 18] else [hashtag],
                        'mentions': ['@expert1', '@expert2'] if author == 'positive_influencer' else []
                    }
                )
                
                # Add sentiment
                post.sentiment = {
                    'label': sentiment,
                    'score': 0.7 + random.random() * 0.3,
                    'method': 'transformer'
                }
                
                posts.append(post)
    
    # Add some duplicates for deduplication demo
    posts.append(posts[0])  # Exact duplicate
    posts.append(Post(
        id='duplicate_content',
        platform='reddit',
        author='copycat',
        author_id='id_copycat',
        content=posts[10].content,  # Content duplicate
        title='Different title',
        timestamp=base_time,
        engagement={'likes': 10, 'shares': 5},
        url='https://example.com/duplicate',
        query='demo_query',
        metadata={}
    ))
    
    # Add a viral post
    viral_post = Post(
        id='viral_post',
        platform='twitter',
        author='viral_user',
        author_id='id_viral',
        content='This is going viral! Amazing content that everyone loves!',
        title=None,
        timestamp=base_time - timedelta(days=3),
        engagement={'likes': 10000, 'shares': 5000},
        url='https://example.com/viral',
        query='demo_query',
        metadata={'hashtags': ['viral', 'trending']}
    )
    viral_post.sentiment = {'label': 'POSITIVE', 'score': 0.95, 'method': 'transformer'}
    posts.append(viral_post)
    
    return posts


def demo_deduplication(posts):
    """Demonstrate deduplication functionality."""
    print("\n=== Phase 4 Demo: Deduplication ===\n")
    
    print(f"📥 Starting with {len(posts)} posts")
    
    # Initialize deduplicator
    dedup = Deduplicator(similarity_threshold=0.85, enable_fuzzy_matching=True)
    
    # Run deduplication
    unique_posts, stats = dedup.deduplicate(posts)
    
    print(f"\n📊 Deduplication Results:")
    print(f"  • Original posts: {stats['total_posts']}")
    print(f"  • Unique posts: {stats['unique_posts']}")
    print(f"  • Duplicates removed: {stats['duplicates_removed']}")
    print(f"  • Duplicate rate: {stats['duplicate_rate']:.1%}")
    
    print(f"\n🔍 Duplicate Types:")
    for dup_type, count in stats['duplicates_by_type'].items():
        print(f"  • {dup_type}: {count}")
    
    print(f"\n📈 Processing Stats:")
    for stat, value in stats['processing_stats'].items():
        print(f"  • {stat}: {value}")
    
    return unique_posts


def demo_aggregation(posts):
    """Demonstrate data aggregation functionality."""
    print("\n\n=== Phase 4 Demo: Data Aggregation ===\n")
    
    # Initialize aggregator
    aggregator = DataAggregator(negative_threshold=0.6, min_posts_for_analysis=3)
    
    # Run aggregation
    stats = aggregator.aggregate(posts)
    
    print("📊 Aggregation Results:\n")
    
    # Basic stats
    print(f"📈 Basic Statistics:")
    print(f"  • Total posts: {stats['total_posts']}")
    print(f"  • Date range: {stats['date_range']['days']} days")
    print(f"  • Unique authors: {stats['authors']['unique_authors']}")
    
    # Platform breakdown
    print(f"\n🌐 Posts by Platform:")
    for platform, count in stats['by_platform'].items():
        print(f"  • {platform}: {count}")
    
    # Sentiment distribution
    print(f"\n😊 Sentiment Distribution:")
    sent_dist = stats['sentiment_distribution']
    for sentiment, percentage in sent_dist['percentages'].items():
        count = sent_dist['counts'][sentiment]
        print(f"  • {sentiment}: {count} posts ({percentage:.1f}%)")
    print(f"  • Sentiment ratio: {sent_dist['sentiment_ratio']:.3f}")
    
    # Engagement metrics
    print(f"\n💬 Engagement Metrics:")
    eng = stats['engagement']
    print(f"  • Total engagement: {eng['total_engagement']:,}")
    print(f"  • Average engagement: {eng['avg_engagement']:.1f}")
    print(f"  • Max engagement: {eng['max_engagement']:,}")
    print(f"  • High engagement posts: {eng['high_engagement_posts']}")
    
    # Author analysis
    print(f"\n👥 Author Analysis:")
    authors = stats['authors']
    print(f"  • Unique authors: {authors['unique_authors']}")
    print(f"  • Posts per author: {authors['posts_per_author']['mean']:.1f} (avg)")
    print(f"  • Most active authors:")
    for author, count in list(authors['most_active'].items())[:3]:
        print(f"    - {author}: {count} posts")
    
    # Negative users
    print(f"\n⚠️  Negative Users:")
    if stats['negative_users']:
        for user in stats['negative_users'][:3]:
            print(f"  • {user['author']}:")
            print(f"    - Negative ratio: {user['negative_ratio']:.1%}")
            print(f"    - Total posts: {user['post_count']}")
            print(f"    - Sentiment: {user['negative_posts']} negative, "
                  f"{user['positive_posts']} positive, {user['neutral_posts']} neutral")
    else:
        print("  • No consistently negative users found")
    
    # Viral posts
    print(f"\n🔥 Viral Posts:")
    if stats['viral_posts']:
        for post in stats['viral_posts'][:3]:
            print(f"  • Post by @{post['author']}:")
            print(f"    - Engagement: {post['engagement']:,}")
            print(f"    - Sentiment: {post['sentiment']}")
            print(f"    - Preview: {post['content_preview'][:50]}...")
    else:
        print("  • No viral posts detected")
    
    # Hashtags
    print(f"\n#️⃣  Top Hashtags:")
    if stats['hashtags']['top_hashtags']:
        for tag, count in list(stats['hashtags']['top_hashtags'].items())[:5]:
            print(f"  • #{tag}: {count} uses")
    
    return stats


def demo_time_series(posts):
    """Demonstrate time series analysis functionality."""
    print("\n\n=== Phase 4 Demo: Time Series Analysis ===\n")
    
    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer(anomaly_threshold=2.0)
    
    # Run analysis
    analysis = analyzer.analyze(posts)
    
    print("⏰ Time Series Analysis Results:\n")
    
    # Daily sentiment
    print("📅 Daily Sentiment Summary:")
    daily = analysis['daily_sentiment']
    if daily:
        for date, data in sorted(daily.items())[-3:]:  # Last 3 days
            print(f"\n  {date}:")
            print(f"    • Posts: {data['total']}")
            print(f"    • Sentiment: {data['positive']} pos, {data['negative']} neg, {data['neutral']} neu")
            print(f"    • Sentiment ratio: {data['sentiment_ratio']:.3f}")
            print(f"    • Engagement: {data['total_engagement']:,}")
    
    # Trends
    print(f"\n📈 Sentiment Trends:")
    trends = analysis['trends']
    if trends:
        print(f"  • Overall trend: {trends['overall_trend']}")
        print(f"  • Trend strength: {trends['trend_strength']:.3f}")
        if 'momentum_indicators' in trends:
            print(f"  • RSI: {trends['momentum_indicators'].get('rsi', 50):.1f}")
            print(f"  • Rate of change: {trends['momentum_indicators'].get('rate_of_change', 0):.1f}%")
    
    # Anomalies
    print(f"\n🚨 Anomalies Detected:")
    anomalies = analysis['anomalies']
    if anomalies:
        for anomaly in anomalies[:3]:
            print(f"  • {anomaly['date']}:")
            print(f"    - Type: {anomaly['type']}")
            print(f"    - Z-score: {anomaly['z_score']:.2f}")
            print(f"    - Severity: {anomaly['severity']}")
    else:
        print("  • No anomalies detected")
    
    # Volatility
    print(f"\n📊 Sentiment Volatility:")
    volatility = analysis['sentiment_volatility']
    if isinstance(volatility, dict):
        print(f"  • Overall: {volatility.get('overall', 0):.3f}")
        print(f"  • Daily: {volatility.get('daily', 0):.3f}")
        print(f"  • Hourly: {volatility.get('hourly', 0):.3f}")
    
    # Peak activity
    print(f"\n⏱️  Peak Activity:")
    peak = analysis['peak_activity']
    if peak.get('peak_hours'):
        print("  • Top activity hours:")
        for hour_data in peak['peak_hours'][:3]:
            print(f"    - {hour_data['hour_of_day']}:00 on {hour_data['day_of_week']}: "
                  f"{hour_data['post_count']} posts")
    
    # Periodicity
    print(f"\n🔄 Periodic Patterns:")
    periodicity = analysis['periodicity']
    if periodicity:
        if periodicity.get('most_positive_hour') is not None:
            print(f"  • Most positive hour: {periodicity['most_positive_hour']}:00")
        if periodicity.get('most_negative_hour') is not None:
            print(f"  • Most negative hour: {periodicity['most_negative_hour']}:00")
    
    return analysis


def demo_integrated_processing():
    """Demonstrate integrated processing pipeline."""
    print("\n\n=== Phase 4 Demo: Integrated Processing Pipeline ===\n")
    
    # Create larger dataset
    print("📊 Creating larger demo dataset...")
    posts = []
    for i in range(5):  # Create 5x the data
        batch = create_demo_posts()
        # Modify IDs to avoid exact duplicates
        for post in batch:
            post.id = f"{post.id}_{i}"
        posts.extend(batch)
    
    print(f"  • Created {len(posts)} posts")
    
    # Time the processing
    import time
    start_time = time.time()
    
    # Run full pipeline
    print("\n🔄 Running full processing pipeline...")
    
    # 1. Deduplication
    dedup = Deduplicator()
    unique_posts, dedup_stats = dedup.deduplicate(posts)
    print(f"  ✓ Deduplication: {len(posts)} → {len(unique_posts)} posts")
    
    # 2. Aggregation
    aggregator = DataAggregator()
    agg_stats = aggregator.aggregate(unique_posts)
    print(f"  ✓ Aggregation: {agg_stats['authors']['unique_authors']} unique authors")
    
    # 3. Time series
    analyzer = TimeSeriesAnalyzer()
    time_analysis = analyzer.analyze(unique_posts)
    print(f"  ✓ Time series: {len(time_analysis['daily_sentiment'])} days analyzed")
    
    processing_time = time.time() - start_time
    
    print(f"\n⏱️  Processing completed in {processing_time:.2f} seconds")
    print(f"  • Posts per second: {len(posts) / processing_time:.0f}")
    
    # Save results
    results = {
        'metadata': {
            'total_posts': len(posts),
            'unique_posts': len(unique_posts),
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        },
        'deduplication': dedup_stats,
        'aggregation': {
            'summary': {
                'total_posts': agg_stats['total_posts'],
                'sentiment_distribution': agg_stats['sentiment_distribution']['counts'],
                'unique_authors': agg_stats['authors']['unique_authors'],
                'negative_users_count': len(agg_stats['negative_users']),
                'viral_posts_count': len(agg_stats['viral_posts'])
            }
        },
        'time_series': {
            'trend': time_analysis['trends'].get('overall_trend'),
            'anomalies_count': len(time_analysis['anomalies']),
            'volatility': time_analysis['sentiment_volatility']
        }
    }
    
    output_file = Path('phase4_demo_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")


if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════╗
    ║     MakeSenseOfIt - Phase 4 Demo         ║
    ║        Data Processing Testing           ║
    ╚══════════════════════════════════════════╝
    """)
    
    # Setup logging
    setup_logging(verbose=True)
    
    try:
        # Create demo data
        print("🎲 Creating demo posts...")
        posts = create_demo_posts()
        print(f"  • Created {len(posts)} demo posts")
        
        # Run demos
        unique_posts = demo_deduplication(posts)
        stats = demo_aggregation(unique_posts)
        analysis = demo_time_series(unique_posts)
        
        # Integrated demo
        demo_integrated_processing()
        
        print("\n\n✅ Phase 4 demo completed successfully!")
        print("📝 All Phase 4 processing components are working correctly")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()