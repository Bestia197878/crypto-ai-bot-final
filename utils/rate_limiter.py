"""
Rate Limiter for Exchange API Calls
Prevents API rate limit violations
"""
import time
from collections import defaultdict
from functools import wraps
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading


@dataclass
class RateLimitConfig:
    """Rate limit configuration for different exchanges"""
    # Requests per minute
    requests_per_minute: int = 60
    # Requests per second
    requests_per_second: int = 10
    # Burst allowance
    burst_limit: int = 20


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self.tokens = self.config.burst_limit
        self.last_update = time.time()
        self.lock = threading.Lock()
        
    def _add_tokens(self):
        """Add tokens based on time passed"""
        now = time.time()
        time_passed = now - self.last_update
        # Add tokens at requests_per_second rate
        tokens_to_add = time_passed * self.config.requests_per_second
        self.tokens = min(self.config.burst_limit, self.tokens + tokens_to_add)
        self.last_update = now
    
    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """Acquire permission to make API call"""
        with self.lock:
            self._add_tokens()
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            
            if not blocking:
                return False
            
            # Calculate wait time
            wait_time = (1 - self.tokens) / self.config.requests_per_second
            
        if timeout and wait_time > timeout:
            return False
            
        time.sleep(wait_time)
        return self.acquire(blocking=False)


class ExchangeRateLimiter:
    """Rate limiter manager for multiple exchanges"""
    
    # Exchange-specific rate limits
    EXCHANGE_LIMITS = {
        'binance': RateLimitConfig(
            requests_per_minute=1200,
            requests_per_second=20,
            burst_limit=50
        ),
        'bybit': RateLimitConfig(
            requests_per_minute=120,
            requests_per_second=2,
            burst_limit=10
        ),
        'kucoin': RateLimitConfig(
            requests_per_minute=2000,
            requests_per_second=33,
            burst_limit=100
        )
    }
    
    def __init__(self):
        self.limiters: Dict[str, RateLimiter] = {}
        self.request_history: Dict[str, List[datetime]] = defaultdict(list)
        self._lock = threading.Lock()
        
    def get_limiter(self, exchange: str) -> RateLimiter:
        """Get or create rate limiter for exchange"""
        exchange = exchange.lower()
        
        if exchange not in self.limiters:
            config = self.EXCHANGE_LIMITS.get(exchange, RateLimitConfig())
            self.limiters[exchange] = RateLimiter(config)
            
        return self.limiters[exchange]
    
    def wait_for_permission(self, exchange: str, timeout: float = 30.0) -> bool:
        """Wait for permission to make API call"""
        limiter = self.get_limiter(exchange)
        return limiter.acquire(blocking=True, timeout=timeout)
    
    def record_request(self, exchange: str, endpoint: str = None):
        """Record API request for monitoring"""
        with self._lock:
            key = f"{exchange}:{endpoint}" if endpoint else exchange
            self.request_history[key].append(datetime.now())
            
            # Clean old history (> 1 hour)
            cutoff = datetime.now() - timedelta(hours=1)
            self.request_history[key] = [
                t for t in self.request_history[key] if t > cutoff
            ]
    
    def get_request_count(self, exchange: str, minutes: int = 1) -> int:
        """Get request count in last N minutes"""
        with self._lock:
            cutoff = datetime.now() - timedelta(minutes=minutes)
            return len([t for t in self.request_history[exchange] if t > cutoff])
    
    def is_rate_limited(self, exchange: str) -> bool:
        """Check if exchange is currently rate limited"""
        limiter = self.get_limiter(exchange)
        return limiter.tokens < 1


# Global rate limiter instance
rate_limiter = ExchangeRateLimiter()


def rate_limited(exchange: str = None, timeout: float = 30.0):
    """Decorator to rate limit API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Determine exchange from args or use default
            exch = exchange
            if not exch and args:
                # Try to get exchange from self.exchange or similar
                if hasattr(args[0], 'exchange'):
                    exch = args[0].exchange
                elif hasattr(args[0], 'name'):
                    exch = args[0].name
            
            if exch:
                # Wait for rate limit permission
                if not rate_limiter.wait_for_permission(exch, timeout):
                    raise RateLimitExceeded(f"Rate limit exceeded for {exch}")
                
                # Record the request
                rate_limiter.record_request(exch, func.__name__)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded"""
    pass


# Convenience function
def check_rate_limit(exchange: str) -> bool:
    """Quick check if API call is allowed"""
    limiter = rate_limiter.get_limiter(exchange)
    return limiter.acquire(blocking=False)
