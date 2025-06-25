"""
Rate limiting implementation for API calls.
Uses token bucket algorithm to ensure we don't exceed rate limits.
"""
import time
import logging
from collections import deque
from threading import RLock
from typing import Optional

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Token bucket rate limiter to prevent exceeding API limits.
    
    This implementation uses a sliding window approach to track
    API calls and ensure we don't exceed the specified rate limit.
    """
    
    def __init__(self, calls_per_minute: int = 60, 
                 burst_size: Optional[int] = None):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_minute: Maximum calls allowed per minute
            burst_size: Maximum burst size (defaults to calls_per_minute)
        """
        self.calls_per_minute = calls_per_minute
        self.burst_size = burst_size or calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        
        # Use deque to store timestamps of recent calls
        self.call_times = deque(maxlen=self.burst_size)
        
        # Thread safety - using reentrant lock to allow nested acquisition
        self.lock = RLock()
        
        # Statistics
        self.total_calls = 0
        self.total_wait_time = 0.0
        self.max_wait_time = 0.0
        
        logger.debug(f"RateLimiter initialized: {calls_per_minute} calls/min, "
                    f"burst size: {self.burst_size}")
    
    def wait_if_needed(self) -> float:
        """
        Wait if necessary to avoid exceeding rate limit.
        
        Returns:
            Time waited in seconds
        """
        wait_time = 0.0
        
        with self.lock:
            now = time.time()
            
            # Remove timestamps older than 1 minute
            self._cleanup_old_timestamps(now)
            
            # Check if we need to wait
            if len(self.call_times) >= self.calls_per_minute:
                # Calculate how long to wait
                oldest_call = self.call_times[0]
                wait_time = 60.0 - (now - oldest_call) + 0.1  # Add small buffer
                
                if wait_time > 0:
                    logger.debug(f"Rate limit reached. Waiting {wait_time:.2f}s")
                    time.sleep(wait_time)
                    now = time.time()
                    
                    # Update statistics
                    self.total_wait_time += wait_time
                    self.max_wait_time = max(self.max_wait_time, wait_time)
            
            # Check burst limit
            elif len(self.call_times) >= self.burst_size:
                # Need to wait at least min_interval since last call
                last_call = self.call_times[-1]
                min_wait = self.min_interval - (now - last_call)
                
                if min_wait > 0:
                    logger.debug(f"Burst limit reached. Waiting {min_wait:.2f}s")
                    time.sleep(min_wait)
                    wait_time = min_wait
                    now = time.time()
                    
                    # Update statistics
                    self.total_wait_time += wait_time
                    self.max_wait_time = max(self.max_wait_time, wait_time)
            
            # Record this call
            self.call_times.append(now)
            self.total_calls += 1
        
        return wait_time
    
    def _cleanup_old_timestamps(self, current_time: float):
        """
        Remove timestamps older than 1 minute.
        
        Args:
            current_time: Current timestamp
        """
        cutoff_time = current_time - 60.0
        
        # Remove old timestamps
        while self.call_times and self.call_times[0] < cutoff_time:
            self.call_times.popleft()
    
    def reset(self):
        """Reset rate limiter state."""
        with self.lock:
            self.call_times.clear()
            self.total_calls = 0
            self.total_wait_time = 0.0
            self.max_wait_time = 0.0
        
        logger.debug("Rate limiter reset")
    
    def get_current_rate(self) -> float:
        """
        Get current call rate (calls per minute).
        
        Returns:
            Current rate of calls per minute
        """
        with self.lock:
            now = time.time()
            self._cleanup_old_timestamps(now)
            
            if not self.call_times:
                return 0.0
            
            # Calculate rate over the time window
            time_window = now - self.call_times[0]
            if time_window > 0:
                return len(self.call_times) * 60.0 / time_window
            
            return float(len(self.call_times))
    
    def get_remaining_calls(self) -> int:
        """
        Get number of calls remaining before rate limit.
        
        Returns:
            Number of calls that can be made without waiting
        """
        with self.lock:
            now = time.time()
            self._cleanup_old_timestamps(now)
            
            return max(0, self.calls_per_minute - len(self.call_times))
    
    def get_stats(self) -> dict:
        """
        Get rate limiter statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        with self.lock:
            return {
                'total_calls': self.total_calls,
                'current_rate': self.get_current_rate(),
                'remaining_calls': self.get_remaining_calls(),
                'total_wait_time': self.total_wait_time,
                'max_wait_time': self.max_wait_time,
                'average_wait_time': self.total_wait_time / self.total_calls if self.total_calls > 0 else 0.0
            }
    
    def __str__(self) -> str:
        """String representation of rate limiter."""
        stats = self.get_stats()
        return (f"RateLimiter({self.calls_per_minute}/min, "
                f"current: {stats['current_rate']:.1f}/min, "
                f"remaining: {stats['remaining_calls']})")
    
    def __repr__(self) -> str:
        """Detailed representation of rate limiter."""
        return f"RateLimiter(calls_per_minute={self.calls_per_minute}, burst_size={self.burst_size})"


class AdaptiveRateLimiter(RateLimiter):
    """
    Adaptive rate limiter that adjusts based on API responses.
    
    This advanced rate limiter can slow down when receiving
    rate limit errors and speed up when API is responding well.
    """
    
    def __init__(self, calls_per_minute: int = 60, 
                 min_rate: int = 10,
                 max_rate: Optional[int] = None):
        """
        Initialize adaptive rate limiter.
        
        Args:
            calls_per_minute: Initial calls per minute
            min_rate: Minimum calls per minute
            max_rate: Maximum calls per minute
        """
        super().__init__(calls_per_minute)
        
        self.initial_rate = calls_per_minute
        self.min_rate = min_rate
        self.max_rate = max_rate or calls_per_minute * 2
        
        # Adaptation parameters
        self.success_count = 0
        self.error_count = 0
        self.last_adjustment = time.time()
        self.adjustment_interval = 60.0  # Adjust every minute
    
    def record_success(self):
        """Record a successful API call."""
        with self.lock:
            self.success_count += 1
            self._maybe_adjust_rate()
    
    def record_error(self, is_rate_limit: bool = False):
        """
        Record an API error.
        
        Args:
            is_rate_limit: Whether the error was a rate limit error
        """
        with self.lock:
            self.error_count += 1
            
            if is_rate_limit:
                # Immediately reduce rate
                self._reduce_rate(factor=0.5)
            
            self._maybe_adjust_rate()
    
    def _maybe_adjust_rate(self):
        """Check if rate should be adjusted based on performance."""
        now = time.time()
        
        if now - self.last_adjustment < self.adjustment_interval:
            return
        
        # Calculate success ratio
        total = self.success_count + self.error_count
        if total < 10:  # Need enough data
            return
        
        success_ratio = self.success_count / total
        
        if success_ratio > 0.95 and self.error_count == 0:
            # Excellent performance, increase rate
            self._increase_rate(factor=1.1)
        elif success_ratio < 0.8 or self.error_count > 5:
            # Poor performance, decrease rate
            self._reduce_rate(factor=0.9)
        
        # Reset counters
        self.success_count = 0
        self.error_count = 0
        self.last_adjustment = now
    
    def _increase_rate(self, factor: float = 1.1):
        """Increase rate limit by factor."""
        new_rate = min(int(self.calls_per_minute * factor), self.max_rate)
        
        if new_rate > self.calls_per_minute:
            logger.info(f"Increasing rate limit: {self.calls_per_minute} → {new_rate}")
            self.calls_per_minute = new_rate
            self.min_interval = 60.0 / new_rate
    
    def _reduce_rate(self, factor: float = 0.9):
        """Reduce rate limit by factor."""
        new_rate = max(int(self.calls_per_minute * factor), self.min_rate)
        
        if new_rate < self.calls_per_minute:
            logger.warning(f"Reducing rate limit: {self.calls_per_minute} → {new_rate}")
            self.calls_per_minute = new_rate
            self.min_interval = 60.0 / new_rate