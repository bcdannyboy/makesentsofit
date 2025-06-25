"""
Time series analysis module.
Analyzes temporal patterns and trends in sentiment data.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class TimeSeriesAnalyzer:
    """
    Analyze time series patterns in sentiment data.
    
    Provides:
    - Daily sentiment aggregation
    - Trend analysis
    - Anomaly detection
    - Volatility measurement
    - Peak activity periods
    """
    
    def __init__(self, anomaly_threshold: float = 2.0,
                 min_data_points: int = 3):
        """
        Initialize time series analyzer.
        
        Args:
            anomaly_threshold: Z-score threshold for anomaly detection
            min_data_points: Minimum data points required for analysis
        """
        self.anomaly_threshold = anomaly_threshold
        self.min_data_points = min_data_points
    
    def analyze(self, posts: List['Post']) -> Dict[str, Any]:
        """
        Generate time series analysis from posts.
        
        Args:
            posts: List of posts to analyze
            
        Returns:
            Dictionary containing time series analysis
        """
        if not posts:
            logger.warning("No posts for time series analysis")
            return self._empty_analysis()
        
        logger.info(f"Starting time series analysis for {len(posts)} posts")
        
        # Create time series DataFrame
        df = self._create_time_series_df(posts)
        
        # Generate analyses
        analysis = {
            'daily_sentiment': self._aggregate_daily_sentiment(df),
            'hourly_sentiment': self._aggregate_hourly_sentiment(df),
            'trends': self._analyze_trends(df),
            'anomalies': self._detect_anomalies(df),
            'sentiment_volatility': self._calculate_volatility(df),
            'peak_activity': self._find_peak_periods(df),
            'momentum': self._calculate_momentum(df),
            'periodicity': self._analyze_periodicity(df),
            'forecast_indicators': self._calculate_forecast_indicators(df)
        }
        
        logger.info("Time series analysis complete")
        return analysis
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure."""
        return {
            'daily_sentiment': {},
            'hourly_sentiment': {},
            'trends': {},
            'anomalies': [],
            'sentiment_volatility': 0.0,
            'peak_activity': {},
            'momentum': {},
            'periodicity': {},
            'forecast_indicators': {}
        }
    
    def _create_time_series_df(self, posts: List['Post']) -> pd.DataFrame:
        """Create DataFrame for time series analysis."""
        data = []
        
        for post in posts:
            sentiment_label = 'UNKNOWN'
            sentiment_score = 0.0
            
            if hasattr(post, 'sentiment') and post.sentiment:
                sentiment_label = post.sentiment.get('label', 'UNKNOWN')
                sentiment_score = post.sentiment.get('score', 0.0)
            
            # Calculate sentiment numeric value
            sentiment_value = self._sentiment_to_numeric(sentiment_label)
            
            data.append({
                'timestamp': post.timestamp,
                'sentiment': sentiment_label,
                'sentiment_score': sentiment_score,
                'sentiment_value': sentiment_value,
                'platform': post.platform,
                'query': post.query,
                'engagement': sum(post.engagement.values()),
                'author': post.author
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def _sentiment_to_numeric(self, sentiment: str) -> float:
        """Convert sentiment label to numeric value."""
        mapping = {
            'POSITIVE': 1.0,
            'NEUTRAL': 0.0,
            'NEGATIVE': -1.0,
            'UNKNOWN': 0.0
        }
        return mapping.get(sentiment, 0.0)
    
    def _aggregate_daily_sentiment(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Aggregate sentiment by day."""
        if df.empty:
            return {}
        
        # Resample to daily frequency
        daily = df.resample('D').agg({
            'sentiment': lambda x: x.value_counts().to_dict() if len(x) > 0 else {},
            'sentiment_value': ['mean', 'std', 'count'],
            'engagement': ['sum', 'mean', 'max'],
            'author': 'nunique'
        })
        
        result = {}
        
        for date, row in daily.iterrows():
            sentiments = row[('sentiment', '<lambda>')]
            if not sentiments:
                continue
                
            total = sum(sentiments.values())
            
            # Calculate metrics
            positive_count = sentiments.get('POSITIVE', 0)
            negative_count = sentiments.get('NEGATIVE', 0)
            neutral_count = sentiments.get('NEUTRAL', 0)
            
            sentiment_ratio = (positive_count - negative_count) / total if total > 0 else 0
            positive_ratio = positive_count / total if total > 0 else 0
            
            result[date.strftime('%Y-%m-%d')] = {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count,
                'total': total,
                'sentiment_ratio': float(sentiment_ratio),
                'positive_ratio': float(positive_ratio),
                'avg_sentiment': float(row[('sentiment_value', 'mean')]),
                'sentiment_std': float(row[('sentiment_value', 'std')]) if not pd.isna(row[('sentiment_value', 'std')]) else 0.0,
                'total_engagement': int(row[('engagement', 'sum')]),
                'avg_engagement': float(row[('engagement', 'mean')]),
                'max_engagement': int(row[('engagement', 'max')]),
                'unique_authors': int(row[('author', 'nunique')]),
                'posts_per_author': float(total / row[('author', 'nunique')]) if row[('author', 'nunique')] > 0 else 0
            }
        
        return result
    
    def _aggregate_hourly_sentiment(self, df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """Aggregate sentiment by hour of day."""
        if df.empty:
            return {}
        
        # Group by hour
        df['hour'] = df.index.hour
        hourly = df.groupby('hour').agg({
            'sentiment_value': ['mean', 'count'],
            'engagement': 'mean'
        })
        
        result = {}
        
        for hour, row in hourly.iterrows():
            result[int(hour)] = {
                'avg_sentiment': float(row[('sentiment_value', 'mean')]),
                'post_count': int(row[('sentiment_value', 'count')]),
                'avg_engagement': float(row[('engagement', 'mean')])
            }
        
        return result
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sentiment trends over time."""
        if len(df) < self.min_data_points:
            return self._empty_trends()
        
        # Daily aggregation for trend analysis
        daily_sentiment = df.resample('D')['sentiment_value'].mean()
        
        if len(daily_sentiment) < 2:
            return self._empty_trends()
        
        # Calculate moving averages
        ma_3 = daily_sentiment.rolling(3, min_periods=1).mean()
        ma_7 = daily_sentiment.rolling(7, min_periods=1).mean()
        
        # Determine trend direction
        recent_trend = self._calculate_trend_direction(daily_sentiment)
        
        # Calculate trend strength
        if len(daily_sentiment) > 1:
            trend_strength = abs(daily_sentiment.iloc[-1] - daily_sentiment.iloc[0])
        else:
            trend_strength = 0.0
        
        # Momentum indicators
        momentum = self._calculate_momentum_indicators(daily_sentiment)
        
        return {
            'overall_trend': recent_trend,
            'trend_strength': float(trend_strength),
            'sentiment_values': {date.strftime('%Y-%m-%d'): float(value) 
                               for date, value in daily_sentiment.items()},
            'ma_3_day': {date.strftime('%Y-%m-%d'): float(value) 
                        for date, value in ma_3.items()},
            'ma_7_day': {date.strftime('%Y-%m-%d'): float(value) 
                        for date, value in ma_7.items()},
            'momentum_indicators': momentum,
            'turning_points': self._identify_turning_points(daily_sentiment)
        }
    
    def _empty_trends(self) -> Dict[str, Any]:
        """Return empty trends structure."""
        return {
            'overall_trend': 'insufficient_data',
            'trend_strength': 0.0,
            'sentiment_values': {},
            'ma_3_day': {},
            'ma_7_day': {},
            'momentum_indicators': {},
            'turning_points': []
        }
    
    def _calculate_trend_direction(self, series: pd.Series) -> str:
        """Calculate trend direction from time series."""
        if len(series) < 3:
            return 'insufficient_data'
        
        # Use linear regression for trend
        x = np.arange(len(series))
        y = series.values
        
        # Remove NaN values
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return 'insufficient_data'
        
        x = x[mask]
        y = y[mask]
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Recent change
        if len(series) >= 3:
            recent_change = series.iloc[-1] - series.iloc[-3]
        else:
            recent_change = 0
        
        # Determine trend
        if abs(slope) < 0.05 and abs(recent_change) < 0.1:
            return 'stable'
        elif slope > 0.05 or recent_change > 0.1:
            return 'improving'
        else:
            return 'worsening'
    
    def _calculate_momentum_indicators(self, series: pd.Series) -> Dict[str, float]:
        """Calculate momentum indicators."""
        if len(series) < 2:
            return {'rsi': 50.0, 'rate_of_change': 0.0}
        
        # Rate of Change (ROC)
        if len(series) >= 7:
            roc = (series.iloc[-1] - series.iloc[-7]) / (series.iloc[-7] + 0.001) * 100
        else:
            roc = 0.0
        
        # Relative Strength Index (RSI) - simplified version
        if len(series) >= 14:
            gains = series.diff().clip(lower=0)
            losses = -series.diff().clip(upper=0)
            
            avg_gain = gains.rolling(14).mean().iloc[-1]
            avg_loss = losses.rolling(14).mean().iloc[-1]
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50.0
        
        return {
            'rsi': float(rsi),
            'rate_of_change': float(roc)
        }
    
    def _identify_turning_points(self, series: pd.Series) -> List[Dict[str, Any]]:
        """Identify turning points in sentiment trend."""
        if len(series) < 3:
            return []
        
        turning_points = []
        
        for i in range(1, len(series) - 1):
            prev_val = series.iloc[i - 1]
            curr_val = series.iloc[i]
            next_val = series.iloc[i + 1]
            
            # Skip if any values are NaN
            if pd.isna(prev_val) or pd.isna(curr_val) or pd.isna(next_val):
                continue
            
            # Local maximum
            if curr_val > prev_val and curr_val > next_val:
                turning_points.append({
                    'date': series.index[i].strftime('%Y-%m-%d'),
                    'type': 'peak',
                    'value': float(curr_val)
                })
            # Local minimum
            elif curr_val < prev_val and curr_val < next_val:
                turning_points.append({
                    'date': series.index[i].strftime('%Y-%m-%d'),
                    'type': 'trough',
                    'value': float(curr_val)
                })
        
        return turning_points
    
    def _detect_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect anomalous sentiment days."""
        if len(df) < self.min_data_points * 2:
            return []
        
        # Daily aggregation
        daily_stats = df.resample('D').agg({
            'sentiment_value': ['mean', 'count'],
            'engagement': 'sum'
        })
        
        # Calculate baseline statistics
        sentiment_mean = daily_stats[('sentiment_value', 'mean')].mean()
        sentiment_std = daily_stats[('sentiment_value', 'mean')].std()
        
        if sentiment_std == 0:
            return []
        
        anomalies = []
        
        for date, row in daily_stats.iterrows():
            sentiment_val = row[('sentiment_value', 'mean')]
            post_count = row[('sentiment_value', 'count')]
            
            if pd.isna(sentiment_val):
                continue
            
            # Calculate z-score
            z_score = (sentiment_val - sentiment_mean) / sentiment_std
            
            if abs(z_score) > self.anomaly_threshold:
                anomaly_type = 'positive_spike' if z_score > 0 else 'negative_spike'
                
                anomalies.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'sentiment_value': float(sentiment_val),
                    'z_score': float(z_score),
                    'type': anomaly_type,
                    'post_count': int(post_count),
                    'total_engagement': int(row[('engagement', 'sum')]),
                    'severity': 'high' if abs(z_score) > 3 else 'medium'
                })
        
        # Sort by severity (z-score)
        anomalies.sort(key=lambda x: abs(x['z_score']), reverse=True)
        
        return anomalies
    
    def _calculate_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate sentiment volatility metrics."""
        if len(df) < 2:
            return {'overall': 0.0, 'daily': 0.0, 'hourly': 0.0}
        
        # Overall volatility (standard deviation of sentiment values)
        overall_volatility = df['sentiment_value'].std()
        
        # Daily volatility
        daily_sentiment = df.resample('D')['sentiment_value'].mean()
        daily_returns = daily_sentiment.pct_change(fill_method=None).dropna()
        daily_volatility = daily_returns.std() if len(daily_returns) > 0 else 0.0
        
        # Hourly volatility
        hourly_sentiment = df.resample('H')['sentiment_value'].mean()
        hourly_returns = hourly_sentiment.pct_change(fill_method=None).dropna()
        hourly_volatility = hourly_returns.std() if len(hourly_returns) > 0 else 0.0
        
        return {
            'overall': float(overall_volatility),
            'daily': float(daily_volatility),
            'hourly': float(hourly_volatility),
            'coefficient_of_variation': float(overall_volatility / abs(df['sentiment_value'].mean())) 
                                       if df['sentiment_value'].mean() != 0 else 0.0
        }
    
    def _find_peak_periods(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Find periods of peak activity."""
        if df.empty:
            return {}
        
        # Hourly aggregation
        hourly = df.resample('H').size()
        
        if len(hourly) == 0:
            return {}
        
        # Find top 10 peak hours
        peak_hours = hourly.nlargest(10)
        
        # Daily aggregation
        daily = df.resample('D').size()
        peak_day = daily.idxmax() if len(daily) > 0 else None
        
        return {
            'peak_hours': [
                {
                    'timestamp': ts.isoformat(),
                    'post_count': int(count),
                    'hour_of_day': ts.hour,
                    'day_of_week': ts.strftime('%A')
                }
                for ts, count in peak_hours.items()
            ],
            'peak_day': {
                'date': peak_day.strftime('%Y-%m-%d') if peak_day else None,
                'post_count': int(daily.max()) if len(daily) > 0 else 0,
                'day_of_week': peak_day.strftime('%A') if peak_day else None
            },
            'avg_posts_per_hour': float(hourly.mean()) if len(hourly) > 0 else 0.0,
            'max_posts_per_hour': int(hourly.max()) if len(hourly) > 0 else 0,
            'activity_concentration': float(hourly.nlargest(24).sum() / hourly.sum()) 
                                     if hourly.sum() > 0 else 0.0
        }
    
    def _calculate_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate sentiment momentum indicators."""
        if len(df) < 2:
            return {}
        
        # Daily sentiment
        daily_sentiment = df.resample('D')['sentiment_value'].mean()
        
        if len(daily_sentiment) < 2:
            return {}
        
        # Simple momentum (current vs N days ago)
        momentum_periods = [1, 3, 7]
        momentum_values = {}
        
        for period in momentum_periods:
            if len(daily_sentiment) > period:
                current = daily_sentiment.iloc[-1]
                past = daily_sentiment.iloc[-period-1]
                momentum = current - past
                momentum_values[f'{period}_day'] = float(momentum)
        
        # Acceleration (change in momentum)
        if len(daily_sentiment) >= 3:
            recent_momentum = daily_sentiment.iloc[-1] - daily_sentiment.iloc[-2]
            prev_momentum = daily_sentiment.iloc[-2] - daily_sentiment.iloc[-3]
            acceleration = recent_momentum - prev_momentum
        else:
            acceleration = 0.0
        
        return {
            'momentum': momentum_values,
            'acceleration': float(acceleration),
            'direction': 'positive' if sum(momentum_values.values()) > 0 else 'negative'
        }
    
    def _analyze_periodicity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze periodic patterns in sentiment."""
        if len(df) < 24:  # Need at least 24 hours of data
            return {}
        
        # Hourly patterns
        hourly_avg = df.groupby(df.index.hour)['sentiment_value'].mean()
        
        # Day of week patterns
        df['day_of_week'] = df.index.dayofweek
        daily_avg = df.groupby('day_of_week')['sentiment_value'].mean()
        
        # Calculate pattern strength (variance)
        hourly_variance = hourly_avg.var() if len(hourly_avg) > 0 else 0.0
        daily_variance = daily_avg.var() if len(daily_avg) > 0 else 0.0
        
        return {
            'hourly_pattern': {int(hour): float(val) for hour, val in hourly_avg.items()},
            'day_of_week_pattern': {int(day): float(val) for day, val in daily_avg.items()},
            'hourly_pattern_strength': float(hourly_variance),
            'weekly_pattern_strength': float(daily_variance),
            'most_positive_hour': int(hourly_avg.idxmax()) if len(hourly_avg) > 0 else None,
            'most_negative_hour': int(hourly_avg.idxmin()) if len(hourly_avg) > 0 else None,
            'most_positive_day': int(daily_avg.idxmax()) if len(daily_avg) > 0 else None,
            'most_negative_day': int(daily_avg.idxmin()) if len(daily_avg) > 0 else None
        }
    
    def _calculate_forecast_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate indicators useful for forecasting."""
        if len(df) < 7:
            return {}
        
        # Recent sentiment distribution
        # Use loc instead of deprecated last method
        three_days_ago = df.index.max() - pd.Timedelta(days=3)
        recent_df = df.loc[df.index >= three_days_ago]
        recent_sentiment = recent_df['sentiment'].value_counts(normalize=True).to_dict()
        
        # Trend consistency
        daily_sentiment = df.resample('D')['sentiment_value'].mean()
        if len(daily_sentiment) >= 3:
            recent_changes = daily_sentiment.diff().iloc[-3:]
            trend_consistency = (recent_changes > 0).sum() / 3  # Proportion of positive changes
        else:
            trend_consistency = 0.5
        
        # Volatility trend
        if len(df) >= 14:
            first_half = df.iloc[:len(df)//2]['sentiment_value'].std()
            second_half = df.iloc[len(df)//2:]['sentiment_value'].std()
            volatility_trend = 'increasing' if second_half > first_half else 'decreasing'
        else:
            volatility_trend = 'unknown'
        
        return {
            'recent_sentiment_distribution': recent_sentiment,
            'trend_consistency': float(trend_consistency),
            'volatility_trend': volatility_trend,
            'last_sentiment_value': float(df['sentiment_value'].iloc[-1]),
            'last_update': df.index[-1].isoformat()
        }