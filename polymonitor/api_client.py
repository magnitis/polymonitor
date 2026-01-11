"""
Polymarket Gamma API Client
============================

Production-ready client for interacting with Polymarket's Gamma API.

Features:
- Automatic pagination to fetch all results
- Exponential backoff for rate limiting
- Disk-based caching to minimize API calls
- Session management for connection pooling
- Comprehensive error handling

API Documentation: https://docs.polymarket.com/developers/gamma-markets-api/fetch-markets-guide

Endpoints Used:
- GET /events - Fetch events with markets
- GET /markets - Fetch individual markets
- GET /tags - Fetch available tags for filtering
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TypeVar

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from diskcache import Cache
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from polymonitor.config import Config, get_config
from polymonitor.models import Event, Market, Tag

# Type variable for generic caching
T = TypeVar("T")

# Set up logging
logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when API rate limit is exceeded."""
    pass


class APIError(Exception):
    """Base exception for API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class PolymarketClient:
    """
    Client for Polymarket's Gamma API.
    
    This client provides methods to fetch events, markets, and tags
    from Polymarket with built-in:
    - Automatic pagination
    - Rate limit handling with exponential backoff
    - Response caching
    - Session management
    
    Example:
        >>> client = PolymarketClient()
        >>> events = client.get_all_events(active=True)
        >>> for event in events:
        ...     print(f"{event.title}: {len(event.markets)} markets")
    
    API Reference:
        https://docs.polymarket.com/developers/gamma-markets-api/fetch-markets-guide
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the Polymarket API client.
        
        Args:
            config: Configuration object. If None, loads from config file.
        """
        self.config = config or get_config()
        self._session: Optional[requests.Session] = None
        self._cache: Optional[Cache] = None
        self._last_request_time: float = 0
        self._min_request_interval: float = 60.0 / self.config.api.rate_limit.requests_per_minute
    
    @property
    def session(self) -> requests.Session:
        """
        Get or create the requests session with retry logic.
        
        The session is configured with:
        - Connection pooling for efficiency
        - Automatic retries for transient errors
        - Proper headers for the API
        """
        if self._session is None:
            self._session = requests.Session()
            
            # Configure retries for transient errors
            retry_strategy = Retry(
                total=self.config.api.max_retries,
                backoff_factor=self.config.api.rate_limit.initial_backoff,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=["GET", "POST"],
            )
            
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=10,
                pool_maxsize=10,
            )
            
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)
            
            # Set default headers
            self._session.headers.update({
                "Accept": "application/json",
                "User-Agent": "PolymonitorClient/0.1.0",
            })
        
        return self._session
    
    @property
    def cache(self) -> Cache:
        """Get or create the disk cache."""
        if self._cache is None and self.config.cache.enabled:
            cache_dir = Path(self.config.cache.directory)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache = Cache(str(cache_dir))
        return self._cache
    
    def _get_cache_key(self, endpoint: str, params: dict[str, Any]) -> str:
        """Generate a cache key from endpoint and parameters."""
        param_str = json.dumps(params, sort_keys=True, default=str)
        key_str = f"{endpoint}:{param_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached(self, key: str, ttl_seconds: int) -> Optional[Any]:
        """
        Get a value from cache if it exists and isn't expired.
        
        Args:
            key: Cache key
            ttl_seconds: Time-to-live in seconds
        
        Returns:
            Cached value or None if not found/expired
        """
        if not self.config.cache.enabled or self.cache is None:
            return None
        
        return self.cache.get(key, default=None)
    
    def _set_cached(self, key: str, value: Any, ttl_seconds: int) -> None:
        """
        Store a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds
        """
        if self.config.cache.enabled and self.cache is not None:
            self.cache.set(key, value, expire=ttl_seconds)
    
    def _rate_limit(self) -> None:
        """
        Enforce rate limiting between requests.
        
        Ensures we don't exceed the configured requests per minute.
        """
        now = time.time()
        elapsed = now - self._last_request_time
        
        if elapsed < self._min_request_interval:
            sleep_time = self._min_request_interval - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _request(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        cache_ttl: Optional[int] = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Make a request to the Gamma API with rate limiting and caching.
        
        Args:
            endpoint: API endpoint (e.g., "/events")
            params: Query parameters
            cache_ttl: Cache TTL in seconds (None to use default)
        
        Returns:
            API response as dict or list
        
        Raises:
            RateLimitError: If rate limited (triggers retry)
            APIError: For other API errors
        """
        params = params or {}
        url = f"{self.config.api.base_url}{endpoint}"
        
        # Check cache first
        if cache_ttl is not None and cache_ttl > 0:
            cache_key = self._get_cache_key(endpoint, params)
            cached = self._get_cached(cache_key, cache_ttl)
            if cached is not None:
                logger.debug(f"Cache hit for {endpoint}")
                return cached
        
        # Apply rate limiting
        self._rate_limit()
        
        logger.debug(f"GET {url} with params {params}")
        
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.config.api.timeout,
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited. Retry after {retry_after}s")
                raise RateLimitError(f"Rate limited. Retry after {retry_after}s")
            
            # Handle other errors
            if response.status_code >= 400:
                error_msg = f"API error {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise APIError(
                    error_msg,
                    status_code=response.status_code,
                    response=response.json() if response.text else None,
                )
            
            data = response.json()
            
            # Cache successful response
            if cache_ttl is not None and cache_ttl > 0:
                self._set_cached(cache_key, data, cache_ttl)
            
            return data
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for {endpoint}")
            raise APIError(f"Request timeout for {endpoint}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {endpoint}: {e}")
            raise APIError(f"Request failed: {e}")
    
    def get_tags(self) -> list[Tag]:
        """
        Fetch all available tags from the API.
        
        Tags are used to categorize events and markets. This is useful
        for filtering to find climate-related markets.
        
        Returns:
            List of Tag objects
        
        Example:
            >>> tags = client.get_tags()
            >>> climate_tags = [t for t in tags if "climate" in t.label.lower()]
        """
        data = self._request(
            "/tags",
            cache_ttl=self.config.cache.ttl.tags,
        )
        
        if isinstance(data, list):
            return [Tag(**tag) for tag in data]
        return []
    
    def get_events(
        self,
        limit: int = 100,
        offset: int = 0,
        active: Optional[bool] = None,
        closed: Optional[bool] = None,
        tag_id: Optional[str] = None,
        slug: Optional[str] = None,
        order: str = "volume",
        ascending: bool = False,
    ) -> list[Event]:
        """
        Fetch events from the Gamma API.
        
        This fetches a single page of events. For all events, use
        get_all_events() which handles pagination.
        
        Args:
            limit: Maximum number of events to return (max 100)
            offset: Offset for pagination
            active: Filter by active status
            closed: Filter by closed status
            tag_id: Filter by tag ID
            slug: Filter by event slug
            order: Sort order field (volume, liquidity, createdAt)
            ascending: Sort direction
        
        Returns:
            List of Event objects
        
        API Reference:
            GET https://gamma-api.polymarket.com/events
            
            Query Parameters:
            - limit: Number of results (default 100)
            - offset: Pagination offset
            - active: Filter active events
            - closed: Filter closed events
            - tag: Filter by tag ID
            - slug: Filter by event slug
            - order: Sort field
            - ascending: Sort direction
        """
        params: dict[str, Any] = {
            "limit": min(limit, 100),  # API max is 100
            "offset": offset,
        }
        
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()
        if tag_id:
            params["tag"] = tag_id
        if slug:
            params["slug"] = slug
        if order:
            params["order"] = order
        if ascending:
            params["ascending"] = "true"
        
        data = self._request(
            "/events",
            params=params,
            cache_ttl=self.config.cache.ttl.events,
        )
        
        if isinstance(data, list):
            return [Event(**event) for event in data]
        return []
    
    def get_all_events(
        self,
        active: Optional[bool] = None,
        closed: Optional[bool] = None,
        tag_id: Optional[str] = None,
        order: str = "volume",
        ascending: bool = False,
        max_pages: int = 100,
    ) -> list[Event]:
        """
        Fetch all events with automatic pagination.
        
        This method handles pagination automatically, fetching all
        available events up to max_pages.
        
        Args:
            active: Filter by active status
            closed: Filter by closed status
            tag_id: Filter by tag ID
            order: Sort order field
            ascending: Sort direction
            max_pages: Maximum number of pages to fetch (safety limit)
        
        Returns:
            List of all Event objects
        
        Example:
            >>> # Get all active events
            >>> events = client.get_all_events(active=True)
            >>> print(f"Found {len(events)} active events")
        """
        all_events: list[Event] = []
        offset = 0
        page_size = self.config.api.page_size
        
        for page in range(max_pages):
            logger.info(f"Fetching events page {page + 1} (offset={offset})")
            
            events = self.get_events(
                limit=page_size,
                offset=offset,
                active=active,
                closed=closed,
                tag_id=tag_id,
                order=order,
                ascending=ascending,
            )
            
            if not events:
                logger.info(f"No more events found after {len(all_events)} total")
                break
            
            all_events.extend(events)
            
            # Check if we got less than a full page (last page)
            if len(events) < page_size:
                logger.info(f"Last page reached with {len(all_events)} total events")
                break
            
            offset += page_size
        
        return all_events
    
    def iter_all_events(
        self,
        active: Optional[bool] = None,
        closed: Optional[bool] = None,
        tag_id: Optional[str] = None,
        order: str = "volume",
        ascending: bool = False,
    ) -> Iterator[Event]:
        """
        Iterate over all events with lazy pagination.
        
        This is memory-efficient for processing large numbers of events
        as it yields events one at a time.
        
        Args:
            active: Filter by active status
            closed: Filter by closed status
            tag_id: Filter by tag ID
            order: Sort order field
            ascending: Sort direction
        
        Yields:
            Event objects one at a time
        
        Example:
            >>> for event in client.iter_all_events(active=True):
            ...     process_event(event)
        """
        offset = 0
        page_size = self.config.api.page_size
        
        while True:
            events = self.get_events(
                limit=page_size,
                offset=offset,
                active=active,
                closed=closed,
                tag_id=tag_id,
                order=order,
                ascending=ascending,
            )
            
            if not events:
                break
            
            for event in events:
                yield event
            
            if len(events) < page_size:
                break
            
            offset += page_size
    
    def get_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        active: Optional[bool] = None,
        closed: Optional[bool] = None,
        order: str = "volume",
        ascending: bool = False,
    ) -> list[Market]:
        """
        Fetch markets directly from the /markets endpoint.
        
        Args:
            limit: Maximum number of markets to return
            offset: Offset for pagination
            active: Filter by active status
            closed: Filter by closed status
            order: Sort order field
            ascending: Sort direction
        
        Returns:
            List of Market objects
        """
        params: dict[str, Any] = {
            "limit": min(limit, 100),
            "offset": offset,
        }
        
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()
        if order:
            params["order"] = order
        if ascending:
            params["ascending"] = "true"
        
        data = self._request(
            "/markets",
            params=params,
            cache_ttl=self.config.cache.ttl.markets,
        )
        
        if isinstance(data, list):
            return [Market(**market) for market in data]
        return []
    
    def get_all_markets(
        self,
        active: Optional[bool] = None,
        closed: Optional[bool] = None,
        order: str = "volume",
        ascending: bool = False,
        max_pages: int = 100,
    ) -> list[Market]:
        """
        Fetch all markets with automatic pagination.
        
        Args:
            active: Filter by active status
            closed: Filter by closed status
            order: Sort order field
            ascending: Sort direction
            max_pages: Maximum pages to fetch
        
        Returns:
            List of all Market objects
        """
        all_markets: list[Market] = []
        offset = 0
        page_size = self.config.api.page_size
        
        for page in range(max_pages):
            logger.info(f"Fetching markets page {page + 1} (offset={offset})")
            
            markets = self.get_markets(
                limit=page_size,
                offset=offset,
                active=active,
                closed=closed,
                order=order,
                ascending=ascending,
            )
            
            if not markets:
                break
            
            all_markets.extend(markets)
            
            if len(markets) < page_size:
                break
            
            offset += page_size
        
        return all_markets
    
    def search_events(self, query: str) -> list[Event]:
        """
        Search events by text query.
        
        Args:
            query: Search query string
        
        Returns:
            List of matching Event objects
        
        Note:
            This searches in event titles and descriptions.
        """
        params = {"_q": query, "limit": 100}
        
        data = self._request(
            "/events",
            params=params,
            cache_ttl=60,  # Short cache for search results
        )
        
        if isinstance(data, list):
            return [Event(**event) for event in data]
        return []
    
    def get_climate_events(
        self,
        keywords: Optional[list[str]] = None,
        min_liquidity: float = 0,
        active_only: bool = True,
    ) -> list[Event]:
        """
        Fetch climate-related events using keyword filtering.
        
        This is a convenience method that fetches events and filters
        them based on climate-related keywords in titles and descriptions.
        
        Args:
            keywords: List of climate keywords to search for.
                     If None, uses default climate keywords from config.
            min_liquidity: Minimum liquidity threshold
            active_only: Only return active events
        
        Returns:
            List of climate-related Event objects
        
        Example:
            >>> climate_events = client.get_climate_events()
            >>> print(f"Found {len(climate_events)} climate events")
        """
        if keywords is None:
            keywords = self.config.filters.climate_keywords
        
        # Normalize keywords for matching
        keywords_lower = [kw.lower() for kw in keywords]
        
        all_events = self.get_all_events(
            active=True if active_only else None,
            closed=False if active_only else None,
        )
        
        climate_events: list[Event] = []
        
        for event in all_events:
            # Check title and description for climate keywords
            text_to_check = f"{event.title} {event.description}".lower()
            
            # Also check tag labels
            tag_text = " ".join(event.tag_labels).lower()
            combined_text = f"{text_to_check} {tag_text}"
            
            is_climate = any(kw in combined_text for kw in keywords_lower)
            
            # Check liquidity threshold
            meets_liquidity = event.liquidity >= min_liquidity
            
            if is_climate and meets_liquidity:
                climate_events.append(event)
        
        logger.info(f"Found {len(climate_events)} climate-related events")
        return climate_events
    
    def get_climate_markets(
        self,
        keywords: Optional[list[str]] = None,
        min_liquidity: float = 0,
        active_only: bool = True,
    ) -> list[Market]:
        """
        Fetch all climate-related markets.
        
        Extracts markets from climate events and also searches for
        individual markets that match climate keywords.
        
        Args:
            keywords: Climate keywords to filter by
            min_liquidity: Minimum liquidity threshold
            active_only: Only return active markets
        
        Returns:
            List of climate-related Market objects
        """
        climate_events = self.get_climate_events(
            keywords=keywords,
            min_liquidity=min_liquidity,
            active_only=active_only,
        )
        
        # Extract all markets from climate events
        all_markets: list[Market] = []
        seen_ids: set[str] = set()
        
        for event in climate_events:
            for market in event.markets:
                if market.id not in seen_ids:
                    # Apply liquidity filter at market level too
                    if market.liquidity >= min_liquidity:
                        all_markets.append(market)
                        seen_ids.add(market.id)
        
        logger.info(f"Found {len(all_markets)} climate-related markets")
        return all_markets
    
    def close(self) -> None:
        """Close the client and cleanup resources."""
        if self._session:
            self._session.close()
            self._session = None
        
        if self._cache:
            self._cache.close()
            self._cache = None
    
    def __enter__(self) -> "PolymarketClient":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        if self._cache:
            self._cache.clear()
            logger.info("Cache cleared")
