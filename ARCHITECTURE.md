# 📊 MakeSenseOfIt - Social Media Sentiment Analysis CLI

A standalone command-line tool for point-in-time sentiment analysis across social media platforms. Scrape, analyze, and visualize sentiment patterns for multiple queries over any time period - completely free.

```bash
python makesentsofit.py --queries "trump,donald trump,maga" --time 365 --output analysis_2024
```

## 🎯 What It Does

- **Multi-Query Analysis**: Analyze multiple related search terms in a single run
- **Time-Based Collection**: Gather all posts from the past N days
- **Sentiment Analysis**: State-of-the-art NLP with 86% accuracy
- **Deduplication**: Smart handling of overlapping results
- **Visualization**: Network graphs, sentiment timelines, and statistical charts
- **Zero Cost**: No API fees, runs entirely on free tools

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/makesentsofit.git
cd makesentsofit
pip install -r requirements.txt

# Run analysis
python makesentsofit.py --queries "bitcoin,btc,crypto" --time 30

# With all options
python makesentsofit.py \
  --queries "climate change,global warming,climate crisis" \
  --time 90 \
  --platforms twitter,reddit \
  --output climate_analysis \
  --format json,html \
  --visualize
```

## 📋 Output Example

```
📊 Analysis Complete: climate_analysis_2024-01-15
├── 📄 climate_analysis_2024-01-15.json (raw data)
├── 📄 climate_analysis_2024-01-15_summary.json (statistics)
├── 📊 climate_analysis_2024-01-15_report.html (interactive)
├── 🖼️ sentiment_timeline.png
├── 🖼️ user_network.png
├── 🖼️ word_cloud.png
└── 📄 negative_users.csv
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- 4GB RAM minimum (8GB recommended for large analyses)
- 2GB free disk space

### Step 1: Install Dependencies

```bash
# Core dependencies
pip install snscrape praw transformers nltk pandas numpy
pip install networkx matplotlib seaborn plotly wordcloud
pip install click rich tqdm python-dateutil

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon')"

# Optional: For browser-based scraping fallback
playwright install chromium
```

### Step 2: Quick Test

```bash
# Test with a small query
python makesentsofit.py --queries "test" --time 1 --limit 10
```

## 📖 Usage

### Basic Commands

```bash
# Single query, last 7 days
python makesentsofit.py --queries "elon musk" --time 7

# Multiple queries, last 30 days
python makesentsofit.py --queries "apple,aapl,tim cook" --time 30

# Specific platforms
python makesentsofit.py --queries "ukraine" --time 7 --platforms twitter

# With visualization
python makesentsofit.py --queries "ai,artificial intelligence" --time 14 --visualize

# Export formats
python makesentsofit.py --queries "tesla" --time 30 --format json,csv,html
```

### Advanced Options

```bash
python makesentsofit.py \
  --queries "query1,query2,query3" \     # Comma-separated queries
  --time 365 \                           # Days to look back
  --platforms twitter,reddit \           # Platforms to scrape
  --subreddits "news,worldnews" \       # Specific subreddits
  --min-engagement 10 \                  # Minimum likes/upvotes
  --languages en,es \                    # Language filter
  --exclude-retweets \                   # Skip retweets
  --sample-size 10000 \                  # Max posts per query
  --output my_analysis \                 # Output file prefix
  --format json,csv,html \               # Output formats
  --visualize \                          # Generate visualizations
  --verbose                              # Detailed logging
```

### Query Syntax

```bash
# Exact phrases
--queries '"exact phrase","another exact phrase"'

# Hashtags
--queries "#bitcoin,#btc,#cryptocurrency"

# Mixed
--queries "elon musk,@elonmusk,#tesla"

# Boolean (Twitter)
--queries "bitcoin AND price,ethereum OR eth"
```

## 📊 Visualizations

The tool generates several visualizations:

1. **Sentiment Timeline**: Daily sentiment trends for each query
2. **User Network Graph**: Connections between users discussing topics
3. **Word Cloud**: Most frequent terms in positive/negative posts
4. **Sentiment Distribution**: Pie charts and histograms
5. **Engagement Analysis**: Correlation between sentiment and engagement
6. **Geographic Heatmap**: Sentiment by location (if available)

## 🔧 Configuration

Create a `config.json` file for default settings:

```json
{
  "default_platforms": ["twitter", "reddit"],
  "default_time_window": 30,
  "rate_limits": {
    "twitter": 50,
    "reddit": 60
  },
  "sentiment_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
  "visualization_style": "dark",
  "output_directory": "./output",
  "cache_directory": "./cache"
}
```

---

# 📘 Phased Development Plan

## Overview

This document outlines a systematic approach to building MakeSenseOfIt through 7 distinct phases. Each phase has clear objectives, deliverables, and tests that must pass before proceeding to the next phase.

### Development Phases

1. **Core CLI and Structure** - Basic command-line interface
2. **Data Collection** - Scraping implementation  
3. **Sentiment Analysis** - NLP integration
4. **Data Processing** - Deduplication and aggregation
5. **Export and Storage** - Output generation
6. **Visualization** - Charts and graphs
7. **Optimization and Polish** - Performance and UX

---

## Phase 1: Core CLI and Structure
**Duration**: 2-3 days | **Complexity**: Low

### Objectives
- Establish project structure
- Implement basic CLI with all arguments
- Create configuration system
- Setup logging and error handling

### Requirements
- Python 3.8+
- click library for CLI
- Basic project structure

### Deliverables

#### 1.1 Project Structure
```
makesentsofit/
├── makesentsofit.py          # Main CLI entry point
├── requirements.txt          # Dependencies
├── config.json              # Default configuration
├── src/
│   ├── __init__.py
│   ├── cli.py               # CLI argument parsing
│   ├── config.py            # Configuration management
│   ├── logger.py            # Logging setup
│   └── utils.py             # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_cli.py
│   └── test_config.py
└── output/                  # Default output directory
```

#### 1.2 Core Implementation

```python
# makesentsofit.py
#!/usr/bin/env python3
"""
MakeSenseOfIt - Social Media Sentiment Analysis CLI
"""
import click
import sys
from datetime import datetime
from src.cli import create_cli
from src.config import Config
from src.logger import setup_logging

@click.command()
@click.option('--queries', '-q', required=True, help='Comma-separated search queries')
@click.option('--time', '-t', default=7, type=int, help='Days to look back')
@click.option('--platforms', '-p', default='twitter,reddit', help='Platforms to scrape')
@click.option('--output', '-o', help='Output file prefix')
@click.option('--format', '-f', default='json', help='Output format(s)')
@click.option('--visualize', '-v', is_flag=True, help='Generate visualizations')
@click.option('--verbose', is_flag=True, help='Verbose output')
@click.option('--config', type=click.Path(exists=True), help='Config file path')
def main(queries, time, platforms, output, format, visualize, verbose, config):
    """Analyze sentiment across social media platforms."""
    # Setup
    setup_logging(verbose)
    cfg = Config(config_file=config)
    
    # Parse arguments
    query_list = [q.strip() for q in queries.split(',')]
    platform_list = [p.strip() for p in platforms.split(',')]
    format_list = [f.strip() for f in format.split(',')]
    
    # Generate output prefix if not provided
    if not output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = f"analysis_{timestamp}"
    
    # Create analysis context
    context = {
        'queries': query_list,
        'time_window_days': time,
        'platforms': platform_list,
        'output_prefix': output,
        'formats': format_list,
        'visualize': visualize,
        'config': cfg
    }
    
    # Log configuration
    click.echo(f"🔍 Analyzing: {', '.join(query_list)}")
    click.echo(f"📅 Time window: {time} days")
    click.echo(f"🌐 Platforms: {', '.join(platform_list)}")
    
    # Phase 1: Just validate and exit
    click.echo("✅ Configuration validated. Ready for Phase 2.")
    return context

if __name__ == '__main__':
    main()
```

```python
# src/config.py
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class Config:
    """Application configuration."""
    # Default values
    default_platforms: List[str] = field(default_factory=lambda: ['twitter', 'reddit'])
    default_time_window: int = 30
    rate_limits: Dict[str, int] = field(default_factory=lambda: {
        'twitter': 50,
        'reddit': 60
    })
    sentiment_model: str = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    output_directory: str = './output'
    cache_directory: str = './cache'
    
    def __init__(self, config_file: Optional[str] = None):
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    setattr(self, key, value)
        
        # Ensure directories exist
        os.makedirs(self.output_directory, exist_ok=True)
        os.makedirs(self.cache_directory, exist_ok=True)
    
    def get_rate_limit(self, platform: str) -> int:
        """Get rate limit for a platform."""
        return self.rate_limits.get(platform, 60)
```

```python
# src/logger.py
import logging
import sys
from rich.logging import RichHandler

def setup_logging(verbose: bool = False):
    """Configure logging with rich output."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Remove existing handlers
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Setup rich handler
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    
    # Set levels for third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)
```

### Tests for Phase 1

```python
# tests/test_cli.py
import pytest
from click.testing import CliRunner
from makesentsofit import main

def test_cli_basic():
    """Test basic CLI functionality."""
    runner = CliRunner()
    result = runner.invoke(main, ['--queries', 'test', '--time', '1'])
    assert result.exit_code == 0
    assert 'Analyzing: test' in result.output

def test_cli_multiple_queries():
    """Test multiple queries parsing."""
    runner = CliRunner()
    result = runner.invoke(main, ['--queries', 'test1,test2,test3', '--time', '7'])
    assert result.exit_code == 0
    assert 'test1' in result.output
    assert 'test2' in result.output
    assert 'test3' in result.output

def test_cli_missing_queries():
    """Test that queries are required."""
    runner = CliRunner()
    result = runner.invoke(main, ['--time', '7'])
    assert result.exit_code != 0
    assert 'Error' in result.output

def test_config_loading():
    """Test configuration loading."""
    from src.config import Config
    config = Config()
    assert config.default_time_window == 30
    assert 'twitter' in config.default_platforms
```

### Exit Criteria for Phase 1
- [ ] CLI accepts all required arguments
- [ ] Configuration system works
- [ ] Logging is properly configured
- [ ] All Phase 1 tests pass
- [ ] Project structure is established

---

## Phase 2: Data Collection
**Duration**: 4-5 days | **Complexity**: High

### Objectives
- Implement multi-platform scraping
- Handle rate limiting
- Support time-based collection
- Error recovery and retries

### Requirements
- Phase 1 complete
- snscrape, praw libraries
- Understanding of platform rate limits

### Deliverables

#### 2.1 New Files
```
src/
├── scrapers/
│   ├── __init__.py
│   ├── base.py             # Base scraper class
│   ├── twitter.py          # Twitter scraper
│   ├── reddit.py           # Reddit scraper
│   └── rate_limiter.py     # Rate limiting
```

#### 2.2 Base Scraper Implementation

```python
# src/scrapers/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class Post:
    """Unified post structure across platforms."""
    id: str
    platform: str
    author: str
    author_id: Optional[str]
    content: str
    title: Optional[str]
    timestamp: datetime
    engagement: Dict[str, int]  # likes, shares, etc.
    url: str
    query: str  # Which search query found this
    metadata: Dict  # Platform-specific data
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'platform': self.platform,
            'author': self.author,
            'author_id': self.author_id,
            'content': self.content,
            'title': self.title,
            'timestamp': self.timestamp.isoformat(),
            'engagement': self.engagement,
            'url': self.url,
            'query': self.query,
            'metadata': self.metadata
        }

class BaseScraper(ABC):
    """Abstract base class for platform scrapers."""
    
    def __init__(self, rate_limiter):
        self.rate_limiter = rate_limiter
        self.posts_collected = 0
        
    @abstractmethod
    def scrape(self, query: str, start_date: datetime, end_date: datetime) -> List[Post]:
        """Scrape posts for a query within date range."""
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """Test if scraper can connect to platform."""
        pass
    
    def scrape_multiple(self, queries: List[str], start_date: datetime, end_date: datetime) -> List[Post]:
        """Scrape multiple queries."""
        all_posts = []
        
        for query in queries:
            logger.info(f"Scraping {self.__class__.__name__} for query: {query}")
            try:
                posts = self.scrape(query, start_date, end_date)
                all_posts.extend(posts)
                logger.info(f"Collected {len(posts)} posts for '{query}'")
            except Exception as e:
                logger.error(f"Error scraping '{query}': {e}")
                
        return all_posts
```

```python
# src/scrapers/twitter.py
import snscrape.modules.twitter as sntwitter
from datetime import datetime
from typing import List
import logging
from .base import BaseScraper, Post

logger = logging.getLogger(__name__)

class TwitterScraper(BaseScraper):
    """Twitter/X scraper using snscrape."""
    
    def validate_connection(self) -> bool:
        """Test connection to Twitter."""
        try:
            # Try a minimal search
            scraper = sntwitter.TwitterSearchScraper('test')
            next(scraper.get_items())
            return True
        except:
            return False
    
    def scrape(self, query: str, start_date: datetime, end_date: datetime) -> List[Post]:
        """Scrape Twitter for posts matching query."""
        posts = []
        
        # Build search query with date filters
        date_query = f"{query} since:{start_date.strftime('%Y-%m-%d')} until:{end_date.strftime('%Y-%m-%d')}"
        
        try:
            scraper = sntwitter.TwitterSearchScraper(date_query)
            
            for tweet in scraper.get_items():
                # Apply rate limiting
                self.rate_limiter.wait_if_needed()
                
                # Check date bounds (snscrape sometimes returns outside range)
                if tweet.date < start_date or tweet.date > end_date:
                    continue
                
                # Create unified post object
                post = Post(
                    id=str(tweet.id),
                    platform='twitter',
                    author=tweet.user.username,
                    author_id=str(tweet.user.id),
                    content=tweet.rawContent,
                    title=None,
                    timestamp=tweet.date,
                    engagement={
                        'likes': tweet.likeCount or 0,
                        'retweets': tweet.retweetCount or 0,
                        'replies': tweet.replyCount or 0
                    },
                    url=tweet.url,
                    query=query,
                    metadata={
                        'lang': tweet.lang,
                        'hashtags': [tag.text for tag in (tweet.hashtags or [])],
                        'mentions': [user.username for user in (tweet.mentionedUsers or [])],
                        'is_retweet': bool(tweet.retweetedTweet)
                    }
                )
                
                posts.append(post)
                self.posts_collected += 1
                
                # Optional: Progress indicator
                if self.posts_collected % 100 == 0:
                    logger.debug(f"Collected {self.posts_collected} posts so far...")
                    
        except Exception as e:
            logger.error(f"Error scraping Twitter: {e}")
            raise
            
        return posts
```

```python
# src/scrapers/reddit.py
import praw
from datetime import datetime
from typing import List
import logging
from .base import BaseScraper, Post

logger = logging.getLogger(__name__)

class RedditScraper(BaseScraper):
    """Reddit scraper using PRAW."""
    
    def __init__(self, rate_limiter, subreddits=['all']):
        super().__init__(rate_limiter)
        self.subreddits = subreddits
        
        # Initialize PRAW with read-only mode
        self.reddit = praw.Reddit(
            client_id='dummy',
            client_secret='dummy',
            user_agent='makesentsofit:v1.0'
        )
        self.reddit.read_only = True
    
    def validate_connection(self) -> bool:
        """Test connection to Reddit."""
        try:
            # Try to access a subreddit
            sub = self.reddit.subreddit('test')
            _ = sub.id
            return True
        except:
            return False
    
    def scrape(self, query: str, start_date: datetime, end_date: datetime) -> List[Post]:
        """Scrape Reddit for posts matching query."""
        posts = []
        
        for subreddit_name in self.subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Determine time filter
                days_diff = (end_date - start_date).days
                if days_diff <= 1:
                    time_filter = 'day'
                elif days_diff <= 7:
                    time_filter = 'week'
                elif days_diff <= 30:
                    time_filter = 'month'
                elif days_diff <= 365:
                    time_filter = 'year'
                else:
                    time_filter = 'all'
                
                # Search posts
                for post in subreddit.search(query, time_filter=time_filter, limit=None):
                    # Apply rate limiting
                    self.rate_limiter.wait_if_needed()
                    
                    # Check date bounds
                    post_time = datetime.fromtimestamp(post.created_utc)
                    if post_time < start_date or post_time > end_date:
                        continue
                    
                    # Create unified post object
                    post_obj = Post(
                        id=post.id,
                        platform='reddit',
                        author=str(post.author) if post.author else '[deleted]',
                        author_id=None,
                        content=post.selftext,
                        title=post.title,
                        timestamp=post_time,
                        engagement={
                            'score': post.score,
                            'upvote_ratio': post.upvote_ratio,
                            'num_comments': post.num_comments
                        },
                        url=f"https://reddit.com{post.permalink}",
                        query=query,
                        metadata={
                            'subreddit': post.subreddit.display_name,
                            'is_video': post.is_video,
                            'over_18': post.over_18,
                            'awards': len(post.all_awardings) if hasattr(post, 'all_awardings') else 0
                        }
                    )
                    
                    posts.append(post_obj)
                    self.posts_collected += 1
                    
            except Exception as e:
                logger.error(f"Error scraping r/{subreddit_name}: {e}")
                
        return posts
```

```python
# src/scrapers/rate_limiter.py
import time
from collections import deque
from threading import Lock

class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.call_times = deque(maxlen=calls_per_minute)
        self.lock = Lock()
        
    def wait_if_needed(self):
        """Wait if necessary to avoid rate limit."""
        with self.lock:
            now = time.time()
            
            # Remove old timestamps
            while self.call_times and now - self.call_times[0] > 60:
                self.call_times.popleft()
            
            # If we've made too many calls, wait
            if len(self.call_times) >= self.calls_per_minute:
                sleep_time = 60 - (now - self.call_times[0]) + 0.1
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    now = time.time()
            
            # Record this call
            self.call_times.append(now)
```

### Integration with Phase 1

Update `makesentsofit.py` to use scrapers:

```python
# Add to main() function after context creation:
from src.scrapers import create_scrapers

# Create scrapers
scrapers = create_scrapers(platform_list, cfg)

# Collect data
from datetime import datetime, timedelta
end_date = datetime.now()
start_date = end_date - timedelta(days=time)

all_posts = []
for platform, scraper in scrapers.items():
    if scraper.validate_connection():
        posts = scraper.scrape_multiple(query_list, start_date, end_date)
        all_posts.extend(posts)
        click.echo(f"✅ Collected {len(posts)} posts from {platform}")
    else:
        click.echo(f"❌ Could not connect to {platform}")

context['posts'] = all_posts
click.echo(f"📊 Total posts collected: {len(all_posts)}")
```

### Tests for Phase 2

```python
# tests/test_scrapers.py
import pytest
from datetime import datetime, timedelta
from src.scrapers.base import Post
from src.scrapers.rate_limiter import RateLimiter

def test_post_creation():
    """Test Post dataclass."""
    post = Post(
        id='123',
        platform='twitter',
        author='testuser',
        author_id='456',
        content='Test content',
        title=None,
        timestamp=datetime.now(),
        engagement={'likes': 10},
        url='https://example.com',
        query='test',
        metadata={}
    )
    
    assert post.id == '123'
    assert post.platform == 'twitter'
    dict_repr = post.to_dict()
    assert 'timestamp' in dict_repr
    assert isinstance(dict_repr['timestamp'], str)

def test_rate_limiter():
    """Test rate limiting."""
    limiter = RateLimiter(calls_per_minute=60)
    
    # Should not wait for first call
    start = time.time()
    limiter.wait_if_needed()
    assert time.time() - start < 0.1
    
    # Rapid calls should eventually wait
    for _ in range(65):
        limiter.wait_if_needed()
    
    # Should have taken at least 1 second
    assert time.time() - start >= 1.0

@pytest.mark.integration
def test_twitter_scraper():
    """Test Twitter scraper (requires internet)."""
    from src.scrapers.twitter import TwitterScraper
    from src.scrapers.rate_limiter import RateLimiter
    
    scraper = TwitterScraper(RateLimiter(50))
    
    # Test connection
    assert scraper.validate_connection()
    
    # Test small scrape
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    posts = scraper.scrape('python', start_date, end_date)
    
    assert len(posts) > 0
    assert all(p.platform == 'twitter' for p in posts)
```

### Exit Criteria for Phase 2
- [ ] Twitter scraper works with snscrape
- [ ] Reddit scraper works with PRAW
- [ ] Rate limiting prevents API bans
- [ ] Time-based filtering works correctly
- [ ] All Phase 2 tests pass
- [ ] Can collect 1000+ posts without errors

---

## Phase 3: Sentiment Analysis
**Duration**: 3-4 days | **Complexity**: Medium

### Objectives
- Integrate transformer models
- Implement VADER fallback
- Batch processing for efficiency
- Confidence scoring

### Requirements
- Phase 2 complete
- transformers, nltk libraries
- GPU optional but recommended

### Deliverables

#### 3.1 New Files
```
src/
├── sentiment/
│   ├── __init__.py
│   ├── analyzer.py         # Main sentiment analyzer
│   ├── models.py           # Model management
│   └── preprocessor.py     # Text preprocessing
```

#### 3.2 Sentiment Analysis Implementation

```python
# src/sentiment/analyzer.py
from typing import List, Dict, Tuple
import logging
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import torch
from .preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Multi-model sentiment analyzer with fallback."""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.preprocessor = TextPreprocessor()
        
        # Initialize transformer model
        try:
            device = 0 if torch.cuda.is_available() else -1
            self.transformer = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=device,
                truncation=True,
                max_length=512
            )
            self.transformer_available = True
            logger.info(f"Loaded transformer model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load transformer model: {e}")
            self.transformer_available = False
        
        # Initialize VADER as fallback
        self.vader = SentimentIntensityAnalyzer()
        
        # Cache for efficiency
        self.cache = {}
        
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment for a batch of texts."""
        results = []
        
        # Preprocess texts
        processed_texts = [self.preprocessor.clean(text) for text in texts]
        
        # Try transformer first
        if self.transformer_available:
            try:
                # Batch process with transformer
                batch_results = self.transformer(processed_texts)
                
                for text, result in zip(texts, batch_results):
                    results.append({
                        'label': result['label'],
                        'score': result['score'],
                        'method': 'transformer',
                        'original_text': text[:200]  # Store snippet
                    })
                    
                return results
                
            except Exception as e:
                logger.error(f"Transformer batch processing failed: {e}")
        
        # Fallback to VADER
        for text in texts:
            results.append(self._analyze_with_vader(text))
            
        return results
    
    def _analyze_with_vader(self, text: str) -> Dict:
        """Analyze using VADER sentiment analyzer."""
        scores = self.vader.polarity_scores(text)
        
        # Map VADER scores to labels
        compound = scores['compound']
        if compound >= 0.05:
            label = 'POSITIVE'
        elif compound <= -0.05:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'
        
        return {
            'label': label,
            'score': abs(compound),  # Confidence
            'method': 'vader',
            'compound': compound,
            'original_text': text[:200],
            'details': scores
        }
    
    def analyze_posts(self, posts: List['Post']) -> List['Post']:
        """Analyze sentiment for a list of posts."""
        # Extract texts
        texts = []
        for post in posts:
            # Combine title and content for Reddit
            if post.title:
                text = f"{post.title} {post.content}"
            else:
                text = post.content
            texts.append(text)
        
        # Batch analyze
        sentiments = self.analyze_batch(texts)
        
        # Attach sentiment to posts
        for post, sentiment in zip(posts, sentiments):
            post.sentiment = sentiment
            
        return posts
```

```python
# src/sentiment/preprocessor.py
import re
import emoji

class TextPreprocessor:
    """Preprocess text for sentiment analysis."""
    
    def __init__(self):
        # Compile regex patterns
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#(\w+)')
        self.whitespace_pattern = re.compile(r'\s+')
        
    def clean(self, text: str) -> str:
        """Clean text for sentiment analysis."""
        if not text:
            return ""
        
        # Convert emojis to text
        text = emoji.demojize(text, delimiters=(" ", " "))
        
        # Remove URLs
        text = self.url_pattern.sub('', text)
        
        # Keep mentions but remove @
        text = self.mention_pattern.sub(lambda m: m.group()[1:], text)
        
        # Keep hashtags but remove #
        text = self.hashtag_pattern.sub(r'\1', text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text)
        
        # Remove extra spaces
        text = text.strip()
        
        return text
    
    def extract_features(self, text: str) -> Dict:
        """Extract features for advanced analysis."""
        features = {
            'has_urls': bool(self.url_pattern.search(text)),
            'num_mentions': len(self.mention_pattern.findall(text)),
            'num_hashtags': len(self.hashtag_pattern.findall(text)),
            'num_emojis': len([c for c in text if c in emoji.EMOJI_DATA]),
            'text_length': len(text),
            'num_words': len(text.split()),
            'exclamation_marks': text.count('!'),
            'question_marks': text.count('?'),
            'all_caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1)
        }
        
        return features
```

### Integration with Phase 2

Update data collection to include sentiment:

```python
# Add to main() after data collection:
from src.sentiment.analyzer import SentimentAnalyzer

# Analyze sentiment
click.echo("🧠 Analyzing sentiment...")
analyzer = SentimentAnalyzer(cfg.sentiment_model)

# Process in batches for efficiency
batch_size = 100
for i in range(0, len(all_posts), batch_size):
    batch = all_posts[i:i+batch_size]
    analyzer.analyze_posts(batch)
    
    # Progress indicator
    progress = min(i + batch_size, len(all_posts))
    click.echo(f"  Processed {progress}/{len(all_posts)} posts...")

# Add sentiment statistics
sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
for post in all_posts:
    if hasattr(post, 'sentiment'):
        sentiment_counts[post.sentiment['label']] += 1

click.echo(f"✅ Sentiment analysis complete:")
click.echo(f"  😊 Positive: {sentiment_counts['POSITIVE']}")
click.echo(f"  😔 Negative: {sentiment_counts['NEGATIVE']}")
click.echo(f"  😐 Neutral: {sentiment_counts['NEUTRAL']}")
```

### Tests for Phase 3

```python
# tests/test_sentiment.py
import pytest
from src.sentiment.analyzer import SentimentAnalyzer
from src.sentiment.preprocessor import TextPreprocessor

def test_preprocessor():
    """Test text preprocessing."""
    preprocessor = TextPreprocessor()
    
    # Test URL removal
    text = "Check out https://example.com for more"
    cleaned = preprocessor.clean(text)
    assert "https://" not in cleaned
    
    # Test mention handling
    text = "@user1 @user2 hello"
    cleaned = preprocessor.clean(text)
    assert "user1" in cleaned
    assert "@" not in cleaned
    
    # Test hashtag handling
    text = "#Python #Programming is fun"
    cleaned = preprocessor.clean(text)
    assert "Python Programming" in cleaned
    assert "#" not in cleaned

def test_sentiment_analyzer():
    """Test sentiment analysis."""
    analyzer = SentimentAnalyzer()
    
    # Test single text
    texts = [
        "I love this! It's amazing!",
        "This is terrible and awful.",
        "It's okay, nothing special."
    ]
    
    results = analyzer.analyze_batch(texts)
    
    assert len(results) == 3
    assert results[0]['label'] == 'POSITIVE'
    assert results[1]['label'] == 'NEGATIVE'
    assert results[2]['label'] in ['NEUTRAL', 'NEGATIVE', 'POSITIVE']  # Model dependent
    
def test_vader_fallback():
    """Test VADER fallback."""
    analyzer = SentimentAnalyzer()
    
    # Force VADER
    result = analyzer._analyze_with_vader("This is fantastic!")
    
    assert result['method'] == 'vader'
    assert result['label'] == 'POSITIVE'
    assert 'compound' in result
```

### Exit Criteria for Phase 3
- [ ] Transformer model loads and works
- [ ] VADER fallback functions correctly
- [ ] Batch processing improves performance
- [ ] Sentiment attached to all posts
- [ ] All Phase 3 tests pass
- [ ] Can process 1000 posts in <60 seconds

---

## Phase 4: Data Processing
**Duration**: 2-3 days | **Complexity**: Medium

### Objectives
- Implement deduplication
- Aggregate statistics
- Time series analysis
- User behavior patterns

### Requirements
- Phase 3 complete
- pandas, numpy libraries
- Understanding of data structures

### Deliverables

#### 4.1 New Files
```
src/
├── processing/
│   ├── __init__.py
│   ├── deduplicator.py     # Remove duplicate posts
│   ├── aggregator.py       # Statistical aggregation
│   ├── analyzer.py         # Pattern analysis
│   └── time_series.py      # Time-based analysis
```

#### 4.2 Data Processing Implementation

```python
# src/processing/deduplicator.py
from typing import List, Set
import hashlib
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class Deduplicator:
    """Remove duplicate posts across queries."""
    
    def __init__(self, similarity_threshold: float = 0.9):
        self.similarity_threshold = similarity_threshold
        self.seen_ids = set()
        self.seen_hashes = set()
        
    def deduplicate(self, posts: List['Post']) -> Tuple[List['Post'], Dict]:
        """Remove duplicates and return stats."""
        unique_posts = []
        duplicates_by_query = defaultdict(int)
        cross_query_duplicates = 0
        
        for post in posts:
            # Check direct ID match
            post_id = f"{post.platform}:{post.id}"
            if post_id in self.seen_ids:
                duplicates_by_query[post.query] += 1
                continue
                
            # Check content hash (for cross-platform duplicates)
            content_hash = self._hash_content(post.content)
            if content_hash in self.seen_hashes:
                cross_query_duplicates += 1
                continue
                
            # Add to unique posts
            self.seen_ids.add(post_id)
            self.seen_hashes.add(content_hash)
            unique_posts.append(post)
        
        stats = {
            'total_posts': len(posts),
            'unique_posts': len(unique_posts),
            'duplicates_removed': len(posts) - len(unique_posts),
            'duplicates_by_query': dict(duplicates_by_query),
            'cross_query_duplicates': cross_query_duplicates
        }
        
        logger.info(f"Deduplication: {stats['total_posts']} → {stats['unique_posts']} posts")
        
        return unique_posts, stats
    
    def _hash_content(self, content: str) -> str:
        """Create hash of content for duplicate detection."""
        # Normalize text
        normalized = content.lower().strip()
        # Remove common variations
        normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
        normalized = ' '.join(normalized.split())  # Normalize whitespace
        
        # Create hash
        return hashlib.sha256(normalized.encode()).hexdigest()
```

```python
# src/processing/aggregator.py
import pandas as pd
from typing import List, Dict
from collections import defaultdict, Counter
from datetime import datetime, timedelta

class DataAggregator:
    """Aggregate statistics from posts."""
    
    def aggregate(self, posts: List['Post']) -> Dict:
        """Generate comprehensive statistics."""
        if not posts:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([{
            'id': p.id,
            'platform': p.platform,
            'author': p.author,
            'timestamp': p.timestamp,
            'query': p.query,
            'sentiment': p.sentiment['label'] if hasattr(p, 'sentiment') else 'UNKNOWN',
            'sentiment_score': p.sentiment['score'] if hasattr(p, 'sentiment') else 0,
            'likes': p.engagement.get('likes', p.engagement.get('score', 0)),
            'shares': p.engagement.get('retweets', p.engagement.get('num_comments', 0)),
            'content_length': len(p.content),
            'has_hashtags': bool(p.metadata.get('hashtags', [])),
            'has_mentions': bool(p.metadata.get('mentions', [])),
        } for p in posts])
        
        # Basic statistics
        stats = {
            'total_posts': len(df),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat(),
                'days': (df['timestamp'].max() - df['timestamp'].min()).days + 1
            },
            
            # Platform breakdown
            'by_platform': df['platform'].value_counts().to_dict(),
            
            # Query breakdown
            'by_query': df['query'].value_counts().to_dict(),
            
            # Sentiment distribution
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'sentiment_by_platform': df.groupby('platform')['sentiment'].value_counts().to_dict(),
            'sentiment_by_query': df.groupby('query')['sentiment'].value_counts().to_dict(),
            
            # Engagement metrics
            'engagement': {
                'total_likes': int(df['likes'].sum()),
                'total_shares': int(df['shares'].sum()),
                'avg_likes': float(df['likes'].mean()),
                'avg_shares': float(df['shares'].mean()),
                'most_liked': int(df['likes'].max()),
                'most_shared': int(df['shares'].max())
            },
            
            # Author statistics
            'authors': {
                'unique_authors': df['author'].nunique(),
                'most_active': df['author'].value_counts().head(10).to_dict(),
                'posts_per_author': float(df.groupby('author').size().mean())
            },
            
            # Content statistics
            'content': {
                'avg_length': float(df['content_length'].mean()),
                'with_hashtags': int(df['has_hashtags'].sum()),
                'with_mentions': int(df['has_mentions'].sum())
            },
            
            # Time patterns
            'temporal': self._analyze_temporal_patterns(df)
        }
        
        # Identify negative users
        stats['negative_users'] = self._identify_negative_users(df)
        
        return stats
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in posts."""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['date'] = df['timestamp'].dt.date
        
        return {
            'posts_by_hour': df['hour'].value_counts().sort_index().to_dict(),
            'posts_by_day_of_week': df['day_of_week'].value_counts().to_dict(),
            'posts_by_date': df['date'].value_counts().sort_index().to_dict(),
            'peak_hour': int(df['hour'].mode()[0]) if not df.empty else None,
            'peak_day': df['day_of_week'].mode()[0] if not df.empty else None
        }
    
    def _identify_negative_users(self, df: pd.DataFrame, min_posts: int = 3) -> List[Dict]:
        """Identify consistently negative users."""
        # Group by author
        author_sentiment = df.groupby('author').agg({
            'sentiment': lambda x: x.value_counts().to_dict(),
            'id': 'count',
            'sentiment_score': 'mean',
            'platform': 'first'
        }).rename(columns={'id': 'post_count'})
        
        # Filter authors with minimum posts
        author_sentiment = author_sentiment[author_sentiment['post_count'] >= min_posts]
        
        # Calculate negativity ratio
        negative_users = []
        for author, row in author_sentiment.iterrows():
            sentiments = row['sentiment']
            negative_count = sentiments.get('NEGATIVE', 0)
            total_count = sum(sentiments.values())
            
            if total_count > 0:
                negative_ratio = negative_count / total_count
                
                if negative_ratio > 0.6:  # 60% negative threshold
                    negative_users.append({
                        'author': author,
                        'platform': row['platform'],
                        'post_count': row['post_count'],
                        'negative_ratio': negative_ratio,
                        'negative_posts': negative_count,
                        'avg_sentiment_score': row['sentiment_score'],
                        'sentiment_breakdown': sentiments
                    })
        
        # Sort by negativity
        negative_users.sort(key=lambda x: x['negative_ratio'], reverse=True)
        
        return negative_users[:50]  # Top 50
```

```python
# src/processing/time_series.py
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta

class TimeSeriesAnalyzer:
    """Analyze time series patterns in sentiment data."""
    
    def analyze(self, posts: List['Post']) -> Dict:
        """Generate time series analysis."""
        if not posts:
            return {}
        
        # Create time series DataFrame
        df = pd.DataFrame([{
            'timestamp': p.timestamp,
            'sentiment': p.sentiment['label'] if hasattr(p, 'sentiment') else 'UNKNOWN',
            'sentiment_score': p.sentiment['score'] if hasattr(p, 'sentiment') else 0,
            'platform': p.platform,
            'query': p.query,
            'engagement': p.engagement.get('likes', 0) + p.engagement.get('retweets', 0)
        } for p in posts])
        
        # Set timestamp as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Daily aggregation
        daily_sentiment = self._aggregate_daily_sentiment(df)
        
        # Trend analysis
        trends = self._analyze_trends(daily_sentiment)
        
        # Anomaly detection
        anomalies = self._detect_anomalies(daily_sentiment)
        
        return {
            'daily_sentiment': daily_sentiment,
            'trends': trends,
            'anomalies': anomalies,
            'sentiment_volatility': self._calculate_volatility(df),
            'peak_activity': self._find_peak_periods(df)
        }
    
    def _aggregate_daily_sentiment(self, df: pd.DataFrame) -> Dict:
        """Aggregate sentiment by day."""
        # Resample to daily frequency
        daily = df.resample('D').agg({
            'sentiment': lambda x: x.value_counts().to_dict(),
            'sentiment_score': ['mean', 'std'],
            'engagement': 'sum'
        })
        
        # Calculate daily sentiment scores
        result = {}
        for date, row in daily.iterrows():
            sentiments = row[('sentiment', '<lambda>')]
            total = sum(sentiments.values())
            
            if total > 0:
                result[date.strftime('%Y-%m-%d')] = {
                    'positive': sentiments.get('POSITIVE', 0),
                    'negative': sentiments.get('NEGATIVE', 0),
                    'neutral': sentiments.get('NEUTRAL', 0),
                    'total': total,
                    'sentiment_ratio': (sentiments.get('POSITIVE', 0) - sentiments.get('NEGATIVE', 0)) / total,
                    'avg_score': row[('sentiment_score', 'mean')],
                    'score_std': row[('sentiment_score', 'std')],
                    'total_engagement': int(row[('engagement', 'sum')])
                }
        
        return result
    
    def _analyze_trends(self, daily_sentiment: Dict) -> Dict:
        """Analyze sentiment trends over time."""
        if len(daily_sentiment) < 3:
            return {}
        
        # Extract sentiment ratios
        dates = sorted(daily_sentiment.keys())
        ratios = [daily_sentiment[d]['sentiment_ratio'] for d in dates]
        
        # Calculate moving averages
        ma_3 = pd.Series(ratios).rolling(3).mean().tolist()
        ma_7 = pd.Series(ratios).rolling(7).mean().tolist()
        
        # Determine trend direction
        recent_trend = 'stable'
        if len(ratios) >= 3:
            recent_change = ratios[-1] - ratios[-3]
            if recent_change > 0.1:
                recent_trend = 'improving'
            elif recent_change < -0.1:
                recent_trend = 'worsening'
        
        return {
            'overall_trend': recent_trend,
            'sentiment_ratios': dict(zip(dates, ratios)),
            'ma_3_day': dict(zip(dates, ma_3)),
            'ma_7_day': dict(zip(dates, ma_7)),
            'trend_strength': abs(ratios[-1] - ratios[0]) if len(ratios) > 1 else 0
        }
    
    def _detect_anomalies(self, daily_sentiment: Dict) -> List[Dict]:
        """Detect anomalous sentiment days."""
        if len(daily_sentiment) < 7:
            return []
        
        anomalies = []
        
        # Calculate baseline statistics
        all_ratios = [d['sentiment_ratio'] for d in daily_sentiment.values()]
        mean_ratio = np.mean(all_ratios)
        std_ratio = np.std(all_ratios)
        
        # Find anomalies (2 standard deviations)
        for date, data in daily_sentiment.items():
            z_score = (data['sentiment_ratio'] - mean_ratio) / std_ratio if std_ratio > 0 else 0
            
            if abs(z_score) > 2:
                anomalies.append({
                    'date': date,
                    'sentiment_ratio': data['sentiment_ratio'],
                    'z_score': z_score,
                    'type': 'positive_spike' if z_score > 0 else 'negative_spike',
                    'total_posts': data['total']
                })
        
        return anomalies
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate sentiment volatility."""
        # Daily sentiment changes
        daily_sentiment = df.resample('D')['sentiment_score'].mean()
        
        if len(daily_sentiment) < 2:
            return 0.0
        
        # Calculate daily returns (changes)
        returns = daily_sentiment.pct_change().dropna()
        
        # Return standard deviation (volatility)
        return float(returns.std()) if len(returns) > 0 else 0.0
    
    def _find_peak_periods(self, df: pd.DataFrame) -> Dict:
        """Find periods of peak activity."""
        # Hourly aggregation
        hourly = df.resample('H').size()
        
        # Find top 10 peak hours
        peak_hours = hourly.nlargest(10)
        
        return {
            'peak_hours': [
                {
                    'timestamp': ts.isoformat(),
                    'post_count': int(count)
                }
                for ts, count in peak_hours.items()
            ],
            'avg_posts_per_hour': float(hourly.mean()),
            'max_posts_per_hour': int(hourly.max()) if len(hourly) > 0 else 0
        }
```

### Integration with Previous Phases

Update main function to include processing:

```python
# Add after sentiment analysis:
from src.processing import Deduplicator, DataAggregator, TimeSeriesAnalyzer

# Deduplicate posts
click.echo("🔍 Deduplicating posts...")
deduplicator = Deduplicator()
unique_posts, dedup_stats = deduplicator.deduplicate(all_posts)
click.echo(f"  Removed {dedup_stats['duplicates_removed']} duplicates")

# Aggregate statistics
click.echo("📊 Aggregating statistics...")
aggregator = DataAggregator()
statistics = aggregator.aggregate(unique_posts)

# Time series analysis
time_analyzer = TimeSeriesAnalyzer()
time_series = time_analyzer.analyze(unique_posts)

# Update context
context['posts'] = unique_posts
context['statistics'] = statistics
context['time_series'] = time_series
context['deduplication'] = dedup_stats

# Display summary
click.echo(f"\n📈 Analysis Summary:")
click.echo(f"  Total unique posts: {len(unique_posts)}")
click.echo(f"  Date range: {statistics['date_range']['days']} days")
click.echo(f"  Unique authors: {statistics['authors']['unique_authors']}")
click.echo(f"  Average sentiment: {time_series.get('trends', {}).get('overall_trend', 'unknown')}")

if statistics.get('negative_users'):
    click.echo(f"\n⚠️  Found {len(statistics['negative_users'])} consistently negative users")
```

### Tests for Phase 4

```python
# tests/test_processing.py
import pytest
from datetime import datetime, timedelta
from src.scrapers.base import Post
from src.processing import Deduplicator, DataAggregator

def create_test_post(id, content, sentiment_label='NEUTRAL'):
    """Helper to create test posts."""
    return Post(
        id=str(id),
        platform='twitter',
        author='testuser',
        author_id='123',
        content=content,
        title=None,
        timestamp=datetime.now() - timedelta(hours=id),
        engagement={'likes': 10, 'retweets': 5},
        url='https://example.com',
        query='test',
        metadata={},
        sentiment={'label': sentiment_label, 'score': 0.8}
    )

def test_deduplicator():
    """Test deduplication."""
    posts = [
        create_test_post(1, "Hello world"),
        create_test_post(1, "Hello world"),  # Duplicate ID
        create_test_post(2, "Hello world"),  # Duplicate content
        create_test_post(3, "Different content")
    ]
    
    dedup = Deduplicator()
    unique, stats = dedup.deduplicate(posts)
    
    assert len(unique) == 2  # Only 2 unique posts
    assert stats['duplicates_removed'] == 2

def test_aggregator():
    """Test data aggregation."""
    posts = [
        create_test_post(1, "Positive post", "POSITIVE"),
        create_test_post(2, "Negative post", "NEGATIVE"),
        create_test_post(3, "Neutral post", "NEUTRAL"),
        create_test_post(4, "Another negative", "NEGATIVE")
    ]
    
    aggregator = DataAggregator()
    stats = aggregator.aggregate(posts)
    
    assert stats['total_posts'] == 4
    assert stats['sentiment_distribution']['NEGATIVE'] == 2
    assert stats['sentiment_distribution']['POSITIVE'] == 1
    assert stats['authors']['unique_authors'] == 1
```

### Exit Criteria for Phase 4
- [ ] Deduplication removes duplicates correctly
- [ ] Statistics aggregation provides insights
- [ ] Time series analysis works
- [ ] Negative user identification works
- [ ] All Phase 4 tests pass
- [ ] Processing 10,000 posts takes <30 seconds

---

## Phase 5: Export and Storage
**Duration**: 2 days | **Complexity**: Low

### Objectives
- Export to multiple formats (JSON, CSV, HTML)
- Generate comprehensive reports
- Create shareable outputs
- Archive raw data

### Requirements
- Phase 4 complete
- pandas, jinja2 libraries
- Understanding of file formats

### Deliverables

#### 5.1 New Files
```
src/
├── export/
│   ├── __init__.py
│   ├── formatter.py        # Format data for export
│   ├── writers.py          # Write to different formats
│   └── templates/          # HTML templates
│       ├── report.html
│       └── styles.css
```

#### 5.2 Export Implementation

```python
# src/export/formatter.py
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

class DataFormatter:
    """Format data for different export types."""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    def format_for_json(self, context: Dict) -> Dict:
        """Format data for JSON export."""
        return {
            'metadata': {
                'generated_at': self.timestamp,
                'version': '1.0',
                'queries': context['queries'],
                'time_window_days': context['time_window_days'],
                'platforms': context['platforms']
            },
            'statistics': context.get('statistics', {}),
            'time_series': context.get('time_series', {}),
            'deduplication': context.get('deduplication', {}),
            'posts': [p.to_dict() for p in context.get('posts', [])]
        }
    
    def format_for_csv(self, context: Dict) -> Dict[str, pd.DataFrame]:
        """Format data for CSV export (multiple files)."""
        dataframes = {}
        
        # Posts DataFrame
        posts_data = []
        for post in context.get('posts', []):
            posts_data.append({
                'id': post.id,
                'platform': post.platform,
                'author': post.author,
                'timestamp': post.timestamp,
                'query': post.query,
                'content': post.content[:500],  # Truncate long content
                'sentiment': post.sentiment.get('label') if hasattr(post, 'sentiment') else '',
                'sentiment_score': post.sentiment.get('score') if hasattr(post, 'sentiment') else 0,
                'likes': post.engagement.get('likes', 0),
                'shares': post.engagement.get('retweets', post.engagement.get('num_comments', 0)),
                'url': post.url
            })
        dataframes['posts'] = pd.DataFrame(posts_data)
        
        # Statistics DataFrame
        stats = context.get('statistics', {})
        if stats:
            stats_data = {
                'metric': [],
                'value': []
            }
            
            # Flatten statistics
            for key, value in stats.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if not isinstance(subvalue, (dict, list)):
                            stats_data['metric'].append(f"{key}_{subkey}")
                            stats_data['value'].append(subvalue)
                elif not isinstance(value, (dict, list)):
                    stats_data['metric'].append(key)
                    stats_data['value'].append(value)
                    
            dataframes['statistics'] = pd.DataFrame(stats_data)
        
        # Negative users DataFrame
        negative_users = stats.get('negative_users', [])
        if negative_users:
            dataframes['negative_users'] = pd.DataFrame(negative_users)
        
        # Time series DataFrame
        time_series = context.get('time_series', {})
        if time_series.get('daily_sentiment'):
            ts_data = []
            for date, data in time_series['daily_sentiment'].items():
                ts_data.append({
                    'date': date,
                    **data
                })
            dataframes['time_series'] = pd.DataFrame(ts_data)
        
        return dataframes
    
    def format_for_html(self, context: Dict) -> Dict:
        """Format data for HTML report."""
        stats = context.get('statistics', {})
        time_series = context.get('time_series', {})
        
        # Prepare data for template
        return {
            'title': f"Sentiment Analysis Report - {self.timestamp}",
            'queries': context['queries'],
            'time_window': context['time_window_days'],
            'platforms': context['platforms'],
            'total_posts': stats.get('total_posts', 0),
            'unique_authors': stats.get('authors', {}).get('unique_authors', 0),
            'date_range': stats.get('date_range', {}),
            'sentiment_distribution': stats.get('sentiment_distribution', {}),
            'top_negative_users': stats.get('negative_users', [])[:10],
            'engagement_stats': stats.get('engagement', {}),
            'time_series_data': time_series.get('daily_sentiment', {}),
            'trends': time_series.get('trends', {}),
            'anomalies': time_series.get('anomalies', []),
            'generated_at': self.timestamp
        }
```

```python
# src/export/writers.py
import json
import os
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from jinja2 import Environment, FileSystemLoader

class ExportWriter:
    """Write formatted data to files."""
    
    def __init__(self, output_dir: str = './output'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup Jinja2 for HTML templates
        template_dir = Path(__file__).parent / 'templates'
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir))
    
    def write_json(self, data: Dict, filename_prefix: str) -> Path:
        """Write JSON file."""
        filepath = self.output_dir / f"{filename_prefix}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
        return filepath
    
    def write_csv(self, dataframes: Dict[str, pd.DataFrame], filename_prefix: str) -> List[Path]:
        """Write multiple CSV files."""
        filepaths = []
        
        for name, df in dataframes.items():
            filepath = self.output_dir / f"{filename_prefix}_{name}.csv"
            df.to_csv(filepath, index=False, encoding='utf-8')
            filepaths.append(filepath)
            
        return filepaths
    
    def write_html(self, data: Dict, filename_prefix: str) -> Path:
        """Write HTML report."""
        template = self.jinja_env.get_template('report.html')
        html_content = template.render(**data)
        
        filepath = self.output_dir / f"{filename_prefix}_report.html"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return filepath
    
    def write_summary(self, context: Dict, filename_prefix: str) -> Path:
        """Write a summary JSON with just statistics."""
        summary = {
            'metadata': {
                'queries': context['queries'],
                'time_window_days': context['time_window_days'],
                'platforms': context['platforms']
            },
            'statistics': context.get('statistics', {}),
            'time_series_summary': {
                'trend': context.get('time_series', {}).get('trends', {}).get('overall_trend'),
                'volatility': context.get('time_series', {}).get('sentiment_volatility'),
                'anomalies_count': len(context.get('time_series', {}).get('anomalies', []))
            },
            'top_negative_users': context.get('statistics', {}).get('negative_users', [])[:10]
        }
        
        filepath = self.output_dir / f"{filename_prefix}_summary.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        return filepath
```

HTML Template Example:
```html
<!-- src/export/templates/report.html -->
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 10px; }
        .metric { display: inline-block; margin: 10px; padding: 15px; background: #e0e0e0; border-radius: 5px; }
        .chart { margin: 20px 0; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        .negative { color: #d32f2f; }
        .positive { color: #388e3c; }
        .neutral { color: #757575; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Generated: {{ generated_at }}</p>
        <p>Queries: {{ queries|join(', ') }}</p>
        <p>Time Window: {{ time_window }} days</p>
        <p>Platforms: {{ platforms|join(', ') }}</p>
    </div>
    
    <h2>Summary Statistics</h2>
    <div class="metrics">
        <div class="metric">
            <h3>Total Posts</h3>
            <p>{{ total_posts|number_format }}</p>
        </div>
        <div class="metric">
            <h3>Unique Authors</h3>
            <p>{{ unique_authors|number_format }}</p>
        </div>
        <div class="metric">
            <h3>Date Range</h3>
            <p>{{ date_range.days }} days</p>
        </div>
    </div>
    
    <h2>Sentiment Distribution</h2>
    <table>
        <tr>
            <th>Sentiment</th>
            <th>Count</th>
            <th>Percentage</th>
        </tr>
        {% for sentiment, count in sentiment_distribution.items() %}
        <tr>
            <td class="{{ sentiment|lower }}">{{ sentiment }}</td>
            <td>{{ count }}</td>
            <td>{{ (count / total_posts * 100)|round(1) }}%</td>
        </tr>
        {% endfor %}
    </table>
    
    <h2>Top Negative Users</h2>
    <table>
        <tr>
            <th>Author</th>
            <th>Platform</th>
            <th>Posts</th>
            <th>Negative Ratio</th>
        </tr>
        {% for user in top_negative_users %}
        <tr>
            <td>{{ user.author }}</td>
            <td>{{ user.platform }}</td>
            <td>{{ user.post_count }}</td>
            <td class="negative">{{ (user.negative_ratio * 100)|round(1) }}%</td>
        </tr>
        {% endfor %}
    </table>
    
    <!-- Add charts placeholder -->
    <h2>Sentiment Timeline</h2>
    <div id="timeline-chart" class="chart">
        <p>Chart will be generated in Phase 6</p>
    </div>
</body>
</html>
```

### Integration

Update main function:

```python
# Add after processing:
from src.export import DataFormatter, ExportWriter

# Format and export data
click.echo("💾 Exporting results...")
formatter = DataFormatter()
writer = ExportWriter(cfg.output_directory)

# Export based on requested formats
exported_files = []

if 'json' in context['formats']:
    json_data = formatter.format_for_json(context)
    filepath = writer.write_json(json_data, context['output_prefix'])
    exported_files.append(filepath)
    click.echo(f"  ✓ JSON: {filepath.name}")

if 'csv' in context['formats']:
    csv_data = formatter.format_for_csv(context)
    filepaths = writer.write_csv(csv_data, context['output_prefix'])
    exported_files.extend(filepaths)
    for fp in filepaths:
        click.echo(f"  ✓ CSV: {fp.name}")

if 'html' in context['formats']:
    html_data = formatter.format_for_html(context)
    filepath = writer.write_html(html_data, context['output_prefix'])
    exported_files.append(filepath)
    click.echo(f"  ✓ HTML: {filepath.name}")

# Always write summary
summary_path = writer.write_summary(context, context['output_prefix'])
exported_files.append(summary_path)
click.echo(f"  ✓ Summary: {summary_path.name}")

click.echo(f"\n✅ Analysis complete! Files saved to: {writer.output_dir}")
```

### Tests for Phase 5

```python
# tests/test_export.py
import pytest
import json
from pathlib import Path
from src.export import DataFormatter, ExportWriter

def test_json_export(tmp_path):
    """Test JSON export."""
    context = {
        'queries': ['test'],
        'time_window_days': 7,
        'platforms': ['twitter'],
        'posts': [],
        'statistics': {'total_posts': 100}
    }
    
    formatter = DataFormatter()
    writer = ExportWriter(tmp_path)
    
    json_data = formatter.format_for_json(context)
    filepath = writer.write_json(json_data, 'test')
    
    assert filepath.exists()
    
    # Verify content
    with open(filepath) as f:
        loaded = json.load(f)
    
    assert loaded['metadata']['queries'] == ['test']
    assert loaded['statistics']['total_posts'] == 100

def test_csv_export(tmp_path):
    """Test CSV export."""
    import pandas as pd
    
    context = {
        'posts': [],
        'statistics': {
            'negative_users': [
                {'author': 'user1', 'negative_ratio': 0.8}
            ]
        }
    }
    
    formatter = DataFormatter()
    writer = ExportWriter(tmp_path)
    
    csv_data = formatter.format_for_csv(context)
    filepaths = writer.write_csv(csv_data, 'test')
    
    assert len(filepaths) > 0
    
    # Verify negative users CSV
    neg_users_file = [f for f in filepaths if 'negative_users' in f.name][0]
    df = pd.read_csv(neg_users_file)
    
    assert len(df) == 1
    assert df.iloc[0]['author'] == 'user1'
```

### Exit Criteria for Phase 5
- [ ] JSON export includes all data
- [ ] CSV export creates multiple files
- [ ] HTML report is readable
- [ ] Summary file is concise
- [ ] All Phase 5 tests pass
- [ ] Exports work for 10MB+ datasets

---

## Phase 6: Visualization
**Duration**: 3-4 days | **Complexity**: Medium

### Objectives
- Generate static visualizations
- Create interactive charts
- Build network graphs
- Generate word clouds

### Requirements
- Phase 5 complete
- matplotlib, seaborn, plotly, networkx libraries
- Basic understanding of data visualization

### Deliverables

#### 6.1 New Files
```
src/
├── visualization/
│   ├── __init__.py
│   ├── charts.py           # Time series and distributions
│   ├── network.py          # Network graphs
│   ├── wordcloud.py        # Word cloud generation
│   └── interactive.py      # Plotly interactive charts
```

#### 6.2 Visualization Implementation

```python
# src/visualization/charts.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from typing import Dict, List
import numpy as np

class ChartGenerator:
    """Generate static charts and graphs."""
    
    def __init__(self, style: str = 'seaborn'):
        plt.style.use(style)
        sns.set_palette("husl")
        
    def sentiment_timeline(self, time_series_data: Dict, output_path: str):
        """Create sentiment timeline chart."""
        if not time_series_data:
            return
            
        # Prepare data
        dates = []
        positive = []
        negative = []
        neutral = []
        
        for date_str, data in sorted(time_series_data.items()):
            dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
            total = data['total']
            positive.append(data['positive'] / total * 100 if total > 0 else 0)
            negative.append(data['negative'] / total * 100 if total > 0 else 0)
            neutral.append(data['neutral'] / total * 100 if total > 0 else 0)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Stacked area chart
        ax1.fill_between(dates, 0, positive, label='Positive', alpha=0.7, color='green')
        ax1.fill_between(dates, positive, [p+n for p,n in zip(positive, negative)], 
                        label='Negative', alpha=0.7, color='red')
        ax1.fill_between(dates, [p+n for p,n in zip(positive, negative)], 100, 
                        label='Neutral', alpha=0.7, color='gray')
        
        ax1.set_ylabel('Sentiment Distribution (%)')
        ax1.set_ylim(0, 100)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Volume chart
        volumes = [data['total'] for _, data in sorted(time_series_data.items())]
        ax2.bar(dates, volumes, alpha=0.5, width=0.8)
        ax2.set_ylabel('Post Volume')
        ax2.set_xlabel('Date')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def sentiment_distribution_pie(self, sentiment_dist: Dict, output_path: str):
        """Create sentiment distribution pie chart."""
        if not sentiment_dist:
            return
            
        # Prepare data
        labels = list(sentiment_dist.keys())
        sizes = list(sentiment_dist.values())
        colors = {'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'gray'}
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels, 
            colors=[colors.get(l, 'blue') for l in labels],
            autopct='%1.1f%%',
            startangle=90
        )
        
        # Enhance text
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_weight('bold')
        
        ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold')
        
        # Save
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def engagement_heatmap(self, posts: List['Post'], output_path: str):
        """Create engagement heatmap by hour and day."""
        if not posts:
            return
            
        # Create DataFrame
        data = []
        for post in posts:
            data.append({
                'hour': post.timestamp.hour,
                'day': post.timestamp.strftime('%A'),
                'engagement': post.engagement.get('likes', 0) + post.engagement.get('retweets', 0)
            })
        
        df = pd.DataFrame(data)
        
        # Pivot for heatmap
        pivot = df.pivot_table(
            values='engagement', 
            index='day', 
            columns='hour', 
            aggfunc='mean'
        )
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = pivot.reindex([d for d in day_order if d in pivot.index])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 6))
        
        sns.heatmap(
            pivot, 
            cmap='YlOrRd', 
            annot=True, 
            fmt='.0f',
            cbar_kws={'label': 'Average Engagement'},
            ax=ax
        )
        
        ax.set_title('Engagement Heatmap by Day and Hour', fontsize=16)
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Day of Week')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
```

```python
# src/visualization/network.py
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict
from collections import defaultdict

class NetworkGraphGenerator:
    """Generate network graphs of user interactions."""
    
    def create_user_network(self, posts: List['Post'], output_path: str, 
                           min_connections: int = 2):
        """Create network graph of user interactions."""
        if not posts:
            return
            
        # Build graph
        G = nx.Graph()
        
        # Track connections (mentions, replies)
        connections = defaultdict(lambda: defaultdict(int))
        
        for post in posts:
            author = post.author
            
            # Add mentions as edges
            for mention in post.metadata.get('mentions', []):
                connections[author][mention] += 1
        
        # Add edges with minimum connection threshold
        for author, mentions in connections.items():
            for mention, count in mentions.items():
                if count >= min_connections:
                    G.add_edge(author, mention, weight=count)
        
        if len(G.nodes()) == 0:
            return
        
        # Calculate node properties
        degree_centrality = nx.degree_centrality(G)
        node_sizes = [300 + 1000 * degree_centrality.get(node, 0) for node in G.nodes()]
        
        # Sentiment coloring
        node_colors = []
        author_sentiments = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
        
        for post in posts:
            if hasattr(post, 'sentiment'):
                label = post.sentiment.get('label', 'NEUTRAL').lower()
                author_sentiments[post.author][label] += 1
        
        for node in G.nodes():
            sentiments = author_sentiments[node]
            total = sum(sentiments.values())
            
            if total == 0:
                node_colors.append('gray')
            elif sentiments['negative'] / total > 0.6:
                node_colors.append('red')
            elif sentiments['positive'] / total > 0.6:
                node_colors.append('green')
            else:
                node_colors.append('yellow')
        
        # Create layout
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw network
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.7,
            ax=ax
        )
        
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            alpha=0.3,
            ax=ax
        )
        
        # Add labels for high-degree nodes
        labels = {node: node for node in G.nodes() 
                 if degree_centrality[node] > 0.1}
        
        nx.draw_networkx_labels(
            G, pos,
            labels,
            font_size=10,
            ax=ax
        )
        
        ax.set_title('User Interaction Network', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Positive',
                      markerfacecolor='g', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Negative',
                      markerfacecolor='r', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Mixed/Neutral',
                      markerfacecolor='y', markersize=10)
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
```

```python
# src/visualization/wordcloud.py
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import List, Dict
import re

class WordCloudGenerator:
    """Generate word clouds from posts."""
    
    def __init__(self):
        # Common words to exclude
        self.stopwords = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about',
            'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'rt', 'via', 'amp', 'https', 'http', 'com'
        ])
    
    def create_wordcloud(self, posts: List['Post'], output_path: str, 
                        sentiment_filter: str = None):
        """Create word cloud from posts."""
        if not posts:
            return
            
        # Filter by sentiment if specified
        if sentiment_filter:
            posts = [p for p in posts 
                    if hasattr(p, 'sentiment') and 
                    p.sentiment.get('label') == sentiment_filter]
        
        if not posts:
            return
        
        # Combine all text
        all_text = ' '.join([
            f"{p.title} {p.content}" if p.title else p.content 
            for p in posts
        ])
        
        # Clean text
        all_text = re.sub(r'https?://\S+', '', all_text)  # Remove URLs
        all_text = re.sub(r'@\w+', '', all_text)  # Remove mentions
        all_text = re.sub(r'#(\w+)', r'\1', all_text)  # Keep hashtag text
        all_text = re.sub(r'[^\w\s]', ' ', all_text)  # Remove punctuation
        all_text = all_text.lower()
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=self.stopwords,
            max_words=100,
            relative_scaling=0.5,
            colormap='viridis' if sentiment_filter == 'POSITIVE' else 'Reds'
        ).generate(all_text)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        title = f"Word Cloud"
        if sentiment_filter:
            title += f" - {sentiment_filter} Sentiment"
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
```

```python
# src/visualization/interactive.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List

class InteractiveChartGenerator:
    """Generate interactive Plotly charts."""
    
    def create_interactive_timeline(self, time_series_data: Dict, output_path: str):
        """Create interactive sentiment timeline."""
        if not time_series_data:
            return
            
        # Prepare data
        df_data = []
        for date, data in sorted(time_series_data.items()):
            df_data.append({
                'date': date,
                'positive': data['positive'],
                'negative': data['negative'],
                'neutral': data['neutral'],
                'total': data['total'],
                'sentiment_ratio': data['sentiment_ratio']
            })
        
        df = pd.DataFrame(df_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Sentiment Distribution', 'Post Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Sentiment traces
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['positive'],
                name='Positive', fill='tonexty',
                line=dict(color='green', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['negative'],
                name='Negative', fill='tonexty',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['date'], y=df['neutral'],
                name='Neutral', fill='tonexty',
                line=dict(color='gray', width=2)
            ),
            row=1, col=1
        )
        
        # Volume bars
        fig.add_trace(
            go.Bar(
                x=df['date'], y=df['total'],
                name='Volume', marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Posts", row=2, col=1)
        
        fig.update_layout(
            title="Interactive Sentiment Timeline",
            hovermode='x unified',
            showlegend=True,
            height=600
        )
        
        # Save as HTML
        fig.write_html(output_path)
    
    def create_3d_sentiment_scatter(self, posts: List['Post'], output_path: str):
        """Create 3D scatter plot of posts."""
        if not posts or len(posts) < 10:
            return
            
        # Prepare data
        data = []
        for post in posts[:1000]:  # Limit for performance
            if hasattr(post, 'sentiment'):
                data.append({
                    'timestamp': post.timestamp,
                    'engagement': post.engagement.get('likes', 0) + 
                                post.engagement.get('retweets', 0),
                    'sentiment_score': post.sentiment.get('score', 0),
                    'sentiment_label': post.sentiment.get('label', 'UNKNOWN'),
                    'author': post.author,
                    'content_preview': post.content[:100] + '...'
                })
        
        df = pd.DataFrame(data)
        
        # Create 3D scatter
        fig = px.scatter_3d(
            df,
            x='timestamp',
            y='engagement',
            z='sentiment_score',
            color='sentiment_label',
            color_discrete_map={
                'POSITIVE': 'green',
                'NEGATIVE': 'red',
                'NEUTRAL': 'gray'
            },
            hover_data=['author', 'content_preview'],
            title='3D Sentiment Analysis'
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Time',
                yaxis_title='Engagement',
                zaxis_title='Sentiment Score'
            ),
            height=700
        )
        
        # Save
        fig.write_html(output_path)
```

### Integration

Update main function:

```python
# Add after export:
if context['visualize']:
    from src.visualization import (
        ChartGenerator, NetworkGraphGenerator, 
        WordCloudGenerator, InteractiveChartGenerator
    )
    
    click.echo("📊 Generating visualizations...")
    
    # Create generators
    charts = ChartGenerator()
    network = NetworkGraphGenerator()
    wordcloud = WordCloudGenerator()
    interactive = InteractiveChartGenerator()
    
    # Generate visualizations
    output_prefix = context['output_prefix']
    
    # Time series chart
    if context.get('time_series', {}).get('daily_sentiment'):
        charts.sentiment_timeline(
            context['time_series']['daily_sentiment'],
            f"{output_prefix}_timeline.png"
        )
        click.echo("  ✓ Sentiment timeline")
    
    # Sentiment distribution
    if context.get('statistics', {}).get('sentiment_distribution'):
        charts.sentiment_distribution_pie(
            context['statistics']['sentiment_distribution'],
            f"{output_prefix}_sentiment_pie.png"
        )
        click.echo("  ✓ Sentiment distribution")
    
    # Network graph
    network.create_user_network(
        context['posts'],
        f"{output_prefix}_network.png"
    )
    click.echo("  ✓ User network graph")
    
    # Word clouds
    wordcloud.create_wordcloud(
        context['posts'],
        f"{output_prefix}_wordcloud_all.png"
    )
    wordcloud.create_wordcloud(
        context['posts'],
        f"{output_prefix}_wordcloud_negative.png",
        sentiment_filter='NEGATIVE'
    )
    click.echo("  ✓ Word clouds")
    
    # Interactive charts
    if context.get('time_series', {}).get('daily_sentiment'):
        interactive.create_interactive_timeline(
            context['time_series']['daily_sentiment'],
            f"{output_prefix}_interactive_timeline.html"
        )
        interactive.create_3d_sentiment_scatter(
            context['posts'],
            f"{output_prefix}_3d_scatter.html"
        )
        click.echo("  ✓ Interactive visualizations")
```

### Tests for Phase 6

```python
# tests/test_visualization.py
import pytest
from pathlib import Path
from src.visualization import ChartGenerator, WordCloudGenerator

def test_chart_generation(tmp_path):
    """Test chart generation."""
    charts = ChartGenerator()
    
    # Test data
    time_series_data = {
        '2024-01-01': {
            'positive': 10,
            'negative': 5,
            'neutral': 5,
            'total': 20,
            'sentiment_ratio': 0.25
        },
        '2024-01-02': {
            'positive': 15,
            'negative': 3,
            'neutral': 7,
            'total': 25,
            'sentiment_ratio': 0.48
        }
    }
    
    output_path = tmp_path / "timeline.png"
    charts.sentiment_timeline(time_series_data, str(output_path))
    
    assert output_path.exists()
    assert output_path.stat().st_size > 0

def test_wordcloud_generation(tmp_path):
    """Test word cloud generation."""
    from src.scrapers.base import Post
    from datetime import datetime
    
    # Create test posts
    posts = [
        Post(
            id='1',
            platform='twitter',
            author='test',
            author_id='123',
            content='Python programming is amazing and fun',
            title=None,
            timestamp=datetime.now(),
            engagement={},
            url='',
            query='test',
            metadata={},
            sentiment={'label': 'POSITIVE', 'score': 0.9}
        )
    ]
    
    wc = WordCloudGenerator()
    output_path = tmp_path / "wordcloud.png"
    
    wc.create_wordcloud(posts, str(output_path))
    
    assert output_path.exists()
```

### Exit Criteria for Phase 6
- [ ] Timeline charts show sentiment trends
- [ ] Network graphs reveal user connections  
- [ ] Word clouds highlight key terms
- [ ] Interactive charts work in browser
- [ ] All Phase 6 tests pass
- [ ] Visualizations generated in <60 seconds

---

## Phase 7: Optimization and Polish
**Duration**: 2-3 days | **Complexity**: Low

### Objectives
- Performance optimization
- Error handling improvements
- User experience enhancements
- Documentation completion

### Requirements
- All previous phases complete
- Performance profiling tools
- User feedback

### Deliverables

#### 7.1 Performance Optimizations

```python
# src/utils/performance.py
import time
import functools
import logging
from contextlib import contextmanager
import psutil
import gc

logger = logging.getLogger(__name__)

def timeit(func):
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.debug(f"{func.__name__} took {duration:.2f}s")
        return result
    return wrapper

@contextmanager
def memory_tracker(operation_name: str):
    """Context manager to track memory usage."""
    gc.collect()
    process = psutil.Process()
    
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    yield
    
    gc.collect()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    logger.debug(f"{operation_name} memory: {mem_before:.1f}MB → {mem_after:.1f}MB "
                f"(+{mem_after - mem_before:.1f}MB)")

class ProgressTracker:
    """Enhanced progress tracking with ETA."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        
    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment
        
        if self.current % 100 == 0 or self.current == self.total:
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0
            
            logger.info(f"{self.description}: {self.current}/{self.total} "
                       f"({self.current/self.total*100:.1f}%) "
                       f"ETA: {eta:.0f}s")
```

#### 7.2 Error Handling Improvements

```python
# src/utils/errors.py
import sys
import traceback
from typing import Optional
import click

class ScraperError(Exception):
    """Base exception for scraper errors."""
    pass

class RateLimitError(ScraperError):
    """Rate limit exceeded."""
    pass

class AuthenticationError(ScraperError):
    """Authentication failed."""
    pass

class NetworkError(ScraperError):
    """Network connection error."""
    pass

def handle_errors(func):
    """Decorator for graceful error handling."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            click.echo("\n\n⚠️  Analysis interrupted by user")
            sys.exit(1)
        except ScraperError as e:
            click.echo(f"\n❌ Scraper error: {e}", err=True)
            sys.exit(2)
        except Exception as e:
            click.echo(f"\n💥 Unexpected error: {e}", err=True)
            
            # Save error details
            with open('error_log.txt', 'w') as f:
                f.write(f"Error: {e}\n\n")
                f.write(traceback.format_exc())
            
            click.echo("Error details saved to error_log.txt", err=True)
            sys.exit(3)
    
    return wrapper

class RetryHandler:
    """Handle retries with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    def retry(self, func, *args, **kwargs):
        """Execute function with retries."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. "
                                 f"Retrying in {delay}s...")
                    time.sleep(delay)
        
        raise last_error
```

#### 7.3 Configuration Validation

```python
# src/utils/validation.py
from typing import List, Dict
import click

class ConfigValidator:
    """Validate configuration and inputs."""
    
    @staticmethod
    def validate_queries(queries: List[str]) -> List[str]:
        """Validate and clean queries."""
        cleaned = []
        
        for query in queries:
            query = query.strip()
            
            if not query:
                continue
                
            if len(query) < 2:
                click.echo(f"⚠️  Skipping too short query: '{query}'", err=True)
                continue
                
            if len(query) > 500:
                click.echo(f"⚠️  Truncating long query: '{query[:50]}...'", err=True)
                query = query[:500]
                
            cleaned.append(query)
        
        if not cleaned:
            raise ValueError("No valid queries provided")
            
        return cleaned
    
    @staticmethod
    def validate_time_window(days: int) -> int:
        """Validate time window."""
        if days < 1:
            raise ValueError("Time window must be at least 1 day")
            
        if days > 365:
            click.echo(f"⚠️  Large time window ({days} days) may take a long time", err=True)
            
        return days
    
    @staticmethod
    def validate_platforms(platforms: List[str]) -> List[str]:
        """Validate platform selection."""
        valid_platforms = {'twitter', 'reddit'}
        validated = []
        
        for platform in platforms:
            platform = platform.lower().strip()
            
            if platform in valid_platforms:
                validated.append(platform)
            else:
                click.echo(f"⚠️  Unknown platform: '{platform}'", err=True)
        
        if not validated:
            raise ValueError("No valid platforms selected")
            
        return validated
```

#### 7.4 Final Main Function

```python
# Updated makesentsofit.py with all optimizations
#!/usr/bin/env python3
"""
MakeSenseOfIt - Social Media Sentiment Analysis CLI
Complete implementation with all phases integrated.
"""
import click
import sys
from datetime import datetime, timedelta
from pathlib import Path

from src.cli import create_cli
from src.config import Config
from src.logger import setup_logging
from src.scrapers import create_scrapers
from src.sentiment.analyzer import SentimentAnalyzer
from src.processing import Deduplicator, DataAggregator, TimeSeriesAnalyzer
from src.export import DataFormatter, ExportWriter
from src.visualization import (
    ChartGenerator, NetworkGraphGenerator, 
    WordCloudGenerator, InteractiveChartGenerator
)
from src.utils.validation import ConfigValidator
from src.utils.errors import handle_errors
from src.utils.performance import memory_tracker, ProgressTracker

@click.command()
@click.option('--queries', '-q', required=True, help='Comma-separated search queries')
@click.option('--time', '-t', default=7, type=int, help='Days to look back')
@click.option('--platforms', '-p', default='twitter,reddit', help='Platforms to scrape')
@click.option('--output', '-o', help='Output file prefix')
@click.option('--format', '-f', default='json', help='Output format(s)')
@click.option('--visualize', '-v', is_flag=True, help='Generate visualizations')
@click.option('--verbose', is_flag=True, help='Verbose output')
@click.option('--config', type=click.Path(exists=True), help='Config file path')
@click.option('--limit', type=int, help='Limit posts per query (for testing)')
@click.option('--no-cache', is_flag=True, help='Disable caching')
@click.version_option(version='1.0.0')
@handle_errors
def main(queries, time, platforms, output, format, visualize, verbose, config, limit, no_cache):
    """
    Analyze sentiment across social media platforms.
    
    Examples:
        
        # Basic usage
        makesentsofit --queries "bitcoin,btc" --time 7
        
        # Multiple platforms with visualization
        makesentsofit -q "climate change" -t 30 -p twitter,reddit -v
        
        # Custom output with multiple formats
        makesentsofit -q "ai,artificial intelligence" -t 14 -f json,csv,html -o ai_analysis
    """
    # Setup
    setup_logging(verbose)
    cfg = Config(config_file=config)
    
    # Display banner
    click.echo("""
    ╔═══════════════════════════════════════╗
    ║      MakeSenseOfIt v1.0.0            ║
    ║   Social Media Sentiment Analysis     ║
    ╚═══════════════════════════════════════╝
    """)
    
    # Validate inputs
    validator = ConfigValidator()
    query_list = validator.validate_queries([q.strip() for q in queries.split(',')])
    platform_list = validator.validate_platforms([p.strip() for p in platforms.split(',')])
    time_window = validator.validate_time_window(time)
    format_list = [f.strip() for f in format.split(',')]
    
    # Generate output prefix if not provided
    if not output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output = f"analysis_{timestamp}"
    
    # Create analysis context
    context = {
        'queries': query_list,
        'time_window_days': time_window,
        'platforms': platform_list,
        'output_prefix': output,
        'formats': format_list,
        'visualize': visualize,
        'config': cfg,
        'limit': limit
    }
    
    # Display configuration
    click.echo(f"🔍 Queries: {', '.join(query_list)}")
    click.echo(f"📅 Time window: {time_window} days")
    click.echo(f"🌐 Platforms: {', '.join(platform_list)}")
    click.echo(f"📁 Output: {output}")
    click.echo("")
    
    # Phase 2: Data Collection
    with memory_tracker("Data Collection"):
        click.echo("📡 Collecting data...")
        scrapers = create_scrapers(platform_list, cfg)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=time_window)
        
        all_posts = []
        for platform, scraper in scrapers.items():
            if scraper.validate_connection():
                posts = scraper.scrape_multiple(query_list, start_date, end_date)
                
                # Apply limit if specified
                if limit:
                    posts = posts[:limit]
                    
                all_posts.extend(posts)
                click.echo(f"  ✓ {platform}: {len(posts)} posts")
            else:
                click.echo(f"  ✗ {platform}: Connection failed", err=True)
        
        if not all_posts:
            click.echo("\n❌ No posts collected. Check your queries and connection.", err=True)
            sys.exit(1)
    
    # Phase 3: Sentiment Analysis  
    with memory_tracker("Sentiment Analysis"):
        click.echo(f"\n🧠 Analyzing sentiment for {len(all_posts)} posts...")
        analyzer = SentimentAnalyzer(cfg.sentiment_model)
        
        # Process in batches with progress tracking
        batch_size = 100
        progress = ProgressTracker(len(all_posts), "Sentiment analysis")
        
        for i in range(0, len(all_posts), batch_size):
            batch = all_posts[i:i+batch_size]
            analyzer.analyze_posts(batch)
            progress.update(len(batch))
    
    # Phase 4: Data Processing
    with memory_tracker("Data Processing"):
        click.echo("\n📊 Processing data...")
        
        # Deduplication
        deduplicator = Deduplicator()
        unique_posts, dedup_stats = deduplicator.deduplicate(all_posts)
        click.echo(f"  ✓ Deduplication: {len(all_posts)} → {len(unique_posts)} posts")
        
        # Aggregation
        aggregator = DataAggregator()
        statistics = aggregator.aggregate(unique_posts)
        
        # Time series analysis
        time_analyzer = TimeSeriesAnalyzer()
        time_series = time_analyzer.analyze(unique_posts)
        
        # Update context
        context['posts'] = unique_posts
        context['statistics'] = statistics
        context['time_series'] = time_series
        context['deduplication'] = dedup_stats
    
    # Display results summary
    click.echo("\n📈 Analysis Summary:")
    click.echo(f"  • Total unique posts: {len(unique_posts):,}")
    click.echo(f"  • Date range: {statistics['date_range']['days']} days")
    click.echo(f"  • Unique authors: {statistics['authors']['unique_authors']:,}")
    
    # Sentiment breakdown
    sent_dist = statistics.get('sentiment_distribution', {})
    total_sent = sum(sent_dist.values())
    if total_sent > 0:
        click.echo("\n  Sentiment Distribution:")
        for sentiment, count in sent_dist.items():
            percentage = (count / total_sent) * 100
            click.echo(f"    • {sentiment}: {count:,} ({percentage:.1f}%)")
    
    # Negative users
    negative_users = statistics.get('negative_users', [])
    if negative_users:
        click.echo(f"\n  ⚠️  Found {len(negative_users)} consistently negative users")
        for user in negative_users[:3]:
            click.echo(f"    • {user['author']}: {user['negative_ratio']:.1%} negative")
    
    # Phase 5: Export
    with memory_tracker("Export"):
        click.echo("\n💾 Exporting results...")
        formatter = DataFormatter()
        writer = ExportWriter(cfg.output_directory)
        
        exported_files = []
        
        if 'json' in format_list:
            json_data = formatter.format_for_json(context)
            filepath = writer.write_json(json_data, context['output_prefix'])
            exported_files.append(filepath)
            click.echo(f"  ✓ JSON: {filepath.name}")
        
        if 'csv' in format_list:
            csv_data = formatter.format_for_csv(context)
            filepaths = writer.write_csv(csv_data, context['output_prefix'])
            exported_files.extend(filepaths)
            for fp in filepaths:
                click.echo(f"  ✓ CSV: {fp.name}")
        
        if 'html' in format_list:
            html_data = formatter.format_for_html(context)
            filepath = writer.write_html(html_data, context['output_prefix'])
            exported_files.append(filepath)
            click.echo(f"  ✓ HTML: {filepath.name}")
        
        # Always write summary
        summary_path = writer.write_summary(context, context['output_prefix'])
        exported_files.append(summary_path)
        click.echo(f"  ✓ Summary: {summary_path.name}")
    
    # Phase 6: Visualization
    if visualize:
        with memory_tracker("Visualization"):
            click.echo("\n📊 Generating visualizations...")
            
            # Create output directory for images
            viz_dir = Path(cfg.output_directory) / context['output_prefix']
            viz_dir.mkdir(exist_ok=True)
            
            # Initialize generators
            charts = ChartGenerator()
            network = NetworkGraphGenerator()
            wordcloud = WordCloudGenerator()
            interactive = InteractiveChartGenerator()
            
            # Generate visualizations with error handling
            try:
                # Timeline
                if time_series.get('daily_sentiment'):
                    charts.sentiment_timeline(
                        time_series['daily_sentiment'],
                        str(viz_dir / 'sentiment_timeline.png')
                    )
                    click.echo("  ✓ Sentiment timeline")
                
                # Pie chart
                if sent_dist:
                    charts.sentiment_distribution_pie(
                        sent_dist,
                        str(viz_dir / 'sentiment_distribution.png')
                    )
                    click.echo("  ✓ Sentiment distribution")
                
                # Network graph
                network.create_user_network(
                    unique_posts,
                    str(viz_dir / 'user_network.png')
                )
                click.echo("  ✓ User network graph")
                
                # Word clouds
                wordcloud.create_wordcloud(
                    unique_posts,
                    str(viz_dir / 'wordcloud_all.png')
                )
                wordcloud.create_wordcloud(
                    unique_posts,
                    str(viz_dir / 'wordcloud_negative.png'),
                    sentiment_filter='NEGATIVE'
                )
                click.echo("  ✓ Word clouds")
                
                # Interactive visualizations
                if time_series.get('daily_sentiment'):
                    interactive.create_interactive_timeline(
                        time_series['daily_sentiment'],
                        str(viz_dir / 'interactive_timeline.html')
                    )
                    interactive.create_3d_sentiment_scatter(
                        unique_posts,
                        str(viz_dir / '3d_sentiment_scatter.html')
                    )
                    click.echo("  ✓ Interactive visualizations")
                    
            except Exception as e:
                click.echo(f"  ⚠️  Visualization error: {e}", err=True)
    
    # Final summary
    click.echo(f"\n✅ Analysis complete!")
    click.echo(f"📁 Results saved to: {Path(cfg.output_directory).absolute()}")
    
    # Open report if HTML was generated
    if 'html' in format_list:
        html_path = Path(cfg.output_directory) / f"{context['output_prefix']}_report.html"
        if html_path.exists():
            click.echo(f"\n🌐 View report: file://{html_path.absolute()}")
            
            if click.confirm("Open report in browser?"):
                import webbrowser
                webbrowser.open(f"file://{html_path.absolute()}")

if __name__ == '__main__':
    main()
```

### Tests for Phase 7

```python
# tests/test_integration.py
import pytest
from click.testing import CliRunner
from makesentsofit import main

def test_full_integration():
    """Test complete analysis pipeline."""
    runner = CliRunner()
    
    with runner.isolated_filesystem():
        # Run analysis
        result = runner.invoke(main, [
            '--queries', 'python,programming',
            '--time', '1',
            '--platforms', 'twitter',
            '--format', 'json,csv',
            '--limit', '10'
        ])
        
        assert result.exit_code == 0
        assert 'Analysis complete!' in result.output
        
        # Check files were created
        import os
        files = os.listdir('output')
        assert any(f.endswith('.json') for f in files)
        assert any(f.endswith('.csv') for f in files)

def test_error_handling():
    """Test error handling."""
    runner = CliRunner()
    
    # Test invalid query
    result = runner.invoke(main, [
        '--queries', '',
        '--time', '7'
    ])
    
    assert result.exit_code != 0

def test_performance_reasonable():
    """Test that performance is reasonable."""
    import time
    runner = CliRunner()
    
    start = time.time()
    
    with runner.isolated_filesystem():
        result = runner.invoke(main, [
            '--queries', 'test',
            '--time', '1',
            '--limit', '100'
        ])
    
    duration = time.time() - start
    
    # Should complete within reasonable time
    assert duration < 300  # 5 minutes max
    assert result.exit_code == 0
```

### Exit Criteria for Phase 7
- [ ] Performance tracking implemented
- [ ] Comprehensive error handling
- [ ] Input validation works
- [ ] Memory usage optimized
- [ ] All integration tests pass
- [ ] Documentation complete
- [ ] Ready for production use

---

## 🎉 Project Complete!

### Final Checklist
- [ ] All 7 phases implemented
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Performance acceptable
- [ ] Error handling robust
- [ ] User experience polished

### Next Steps
1. Deploy to production
2. Monitor usage and gather feedback
3. Plan v2.0 features
4. Consider API version
5. Add more platforms (LinkedIn, TikTok, etc.)

---

**Congratulations!** You now have a complete, production-ready social media sentiment analysis tool that:
- Costs $0 in API fees
- Handles multiple queries and platforms
- Provides comprehensive analysis
- Generates beautiful visualizations
- Scales to millions of posts

The phased approach ensures each component is properly tested before moving on, making development systematic and manageable.