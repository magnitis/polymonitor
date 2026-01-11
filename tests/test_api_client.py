"""
Tests for Polymarket API client.
"""

import pytest
import responses
import json
from pathlib import Path

from polymonitor.api_client import PolymarketClient, RateLimitError, APIError
from polymonitor.config import Config


# Sample API responses for testing
SAMPLE_EVENTS = [
    {
        "id": "event1",
        "title": "Climate 2026",
        "description": "Climate predictions for 2026",
        "active": True,
        "closed": False,
        "volume": 100000,
        "liquidity": 50000,
        "markets": [
            {
                "id": "market1",
                "question": "Will 2026 be the hottest year on record?",
                "outcomePrices": [0.65, 0.35],
                "clobTokenIds": ["token1", "token2"],
                "volume": 50000,
                "liquidity": 25000,
            }
        ],
        "tags": [
            {"id": "1", "label": "Climate"},
        ],
    }
]

SAMPLE_TAGS = [
    {"id": "1", "label": "Climate", "slug": "climate"},
    {"id": "2", "label": "Weather", "slug": "weather"},
    {"id": "3", "label": "Politics", "slug": "politics"},
]


class TestPolymarketClient:
    """Tests for PolymarketClient."""
    
    @pytest.fixture
    def client(self):
        """Create a client with caching disabled for testing."""
        config = Config()
        config.cache.enabled = False
        return PolymarketClient(config)
    
    @responses.activate
    def test_get_tags(self, client):
        """Test fetching tags."""
        responses.add(
            responses.GET,
            "https://gamma-api.polymarket.com/tags",
            json=SAMPLE_TAGS,
            status=200,
        )
        
        tags = client.get_tags()
        
        assert len(tags) == 3
        assert tags[0].label == "Climate"
        assert tags[1].slug == "weather"
    
    @responses.activate
    def test_get_events(self, client):
        """Test fetching events."""
        responses.add(
            responses.GET,
            "https://gamma-api.polymarket.com/events",
            json=SAMPLE_EVENTS,
            status=200,
        )
        
        events = client.get_events(limit=10)
        
        assert len(events) == 1
        assert events[0].title == "Climate 2026"
        assert len(events[0].markets) == 1
    
    @responses.activate
    def test_get_events_with_filters(self, client):
        """Test fetching events with filters."""
        responses.add(
            responses.GET,
            "https://gamma-api.polymarket.com/events",
            json=SAMPLE_EVENTS,
            status=200,
        )
        
        events = client.get_events(
            active=True,
            closed=False,
            order="volume",
        )
        
        assert len(events) == 1
        # Check that request was made with correct params
        assert "active=true" in responses.calls[0].request.url
    
    @responses.activate
    def test_get_all_events_pagination(self, client):
        """Test automatic pagination."""
        # First page
        responses.add(
            responses.GET,
            "https://gamma-api.polymarket.com/events",
            json=SAMPLE_EVENTS * 100,  # Full page
            status=200,
        )
        
        # Second page (empty - end of results)
        responses.add(
            responses.GET,
            "https://gamma-api.polymarket.com/events",
            json=[],
            status=200,
        )
        
        events = client.get_all_events()
        
        assert len(events) == 100
        assert len(responses.calls) == 2
    
    @responses.activate
    def test_rate_limit_handling(self, client):
        """Test rate limit retry logic."""
        # First request rate limited
        responses.add(
            responses.GET,
            "https://gamma-api.polymarket.com/events",
            json={"error": "rate limited"},
            status=429,
            headers={"Retry-After": "1"},
        )
        
        # Second request succeeds
        responses.add(
            responses.GET,
            "https://gamma-api.polymarket.com/events",
            json=SAMPLE_EVENTS,
            status=200,
        )
        
        events = client.get_events()
        
        assert len(events) == 1
        assert len(responses.calls) == 2
    
    @responses.activate
    def test_api_error_handling(self, client):
        """Test API error handling."""
        responses.add(
            responses.GET,
            "https://gamma-api.polymarket.com/events",
            json={"error": "internal server error"},
            status=500,
        )
        
        with pytest.raises(APIError) as exc_info:
            client.get_events()
        
        assert exc_info.value.status_code == 500
    
    @responses.activate
    def test_get_climate_events(self, client):
        """Test fetching climate-related events."""
        responses.add(
            responses.GET,
            "https://gamma-api.polymarket.com/events",
            json=SAMPLE_EVENTS + [
                {
                    "id": "event2",
                    "title": "Sports 2026",
                    "description": "Sports predictions",
                    "active": True,
                    "markets": [],
                    "tags": [],
                }
            ],
            status=200,
        )
        
        # Empty second page
        responses.add(
            responses.GET,
            "https://gamma-api.polymarket.com/events",
            json=[],
            status=200,
        )
        
        events = client.get_climate_events()
        
        # Should only return climate event
        assert len(events) == 1
        assert "Climate" in events[0].title
    
    @responses.activate
    def test_climate_keyword_filtering(self, client):
        """Test that climate keywords filter correctly."""
        events_data = [
            {
                "id": "1",
                "title": "Hurricane Season 2026",
                "description": "Atlantic hurricane predictions",
                "active": True,
                "markets": [],
                "tags": [],
            },
            {
                "id": "2",
                "title": "Temperature Records",
                "description": "Will it be the hottest year?",
                "active": True,
                "markets": [],
                "tags": [],
            },
            {
                "id": "3",
                "title": "Stock Market 2026",
                "description": "Financial predictions",
                "active": True,
                "markets": [],
                "tags": [],
            },
        ]
        
        responses.add(
            responses.GET,
            "https://gamma-api.polymarket.com/events",
            json=events_data,
            status=200,
        )
        
        responses.add(
            responses.GET,
            "https://gamma-api.polymarket.com/events",
            json=[],
            status=200,
        )
        
        events = client.get_climate_events()
        
        # Should return hurricane and temperature events, not stock
        assert len(events) == 2
        titles = [e.title for e in events]
        assert "Hurricane Season 2026" in titles
        assert "Temperature Records" in titles
        assert "Stock Market 2026" not in titles
    
    def test_context_manager(self):
        """Test context manager usage."""
        with PolymarketClient() as client:
            assert client.session is not None
        
        # Session should be closed after context
        assert client._session is None
    
    @responses.activate
    def test_search_events(self, client):
        """Test event search."""
        responses.add(
            responses.GET,
            "https://gamma-api.polymarket.com/events",
            json=SAMPLE_EVENTS,
            status=200,
        )
        
        events = client.search_events("climate")
        
        assert len(events) == 1
        # Check search query was included
        assert "_q=climate" in responses.calls[0].request.url


class TestCaching:
    """Tests for caching functionality."""
    
    @pytest.fixture
    def cached_client(self, tmp_path):
        """Create a client with caching enabled."""
        config = Config()
        config.cache.enabled = True
        config.cache.directory = str(tmp_path / "cache")
        return PolymarketClient(config)
    
    @responses.activate
    def test_cache_hit(self, cached_client):
        """Test that cached responses are returned."""
        responses.add(
            responses.GET,
            "https://gamma-api.polymarket.com/tags",
            json=SAMPLE_TAGS,
            status=200,
        )
        
        # First call - should hit API
        tags1 = cached_client.get_tags()
        
        # Second call - should use cache
        tags2 = cached_client.get_tags()
        
        # Only one API call should be made
        assert len(responses.calls) == 1
        assert tags1 == tags2
    
    def test_clear_cache(self, cached_client):
        """Test cache clearing."""
        # Ensure cache exists
        _ = cached_client.cache
        
        cached_client.clear_cache()
        
        # Cache should be empty (no error means success)


class TestIterators:
    """Tests for iterator methods."""
    
    @pytest.fixture
    def client(self):
        config = Config()
        config.cache.enabled = False
        return PolymarketClient(config)
    
    @responses.activate
    def test_iter_all_events(self, client):
        """Test lazy iteration over events."""
        # First page
        responses.add(
            responses.GET,
            "https://gamma-api.polymarket.com/events",
            json=SAMPLE_EVENTS * 50,
            status=200,
        )
        
        # Second page (partial - last page)
        responses.add(
            responses.GET,
            "https://gamma-api.polymarket.com/events",
            json=SAMPLE_EVENTS * 10,
            status=200,
        )
        
        count = 0
        for event in client.iter_all_events():
            count += 1
            if count >= 5:
                break
        
        # Should have made only 1 request (first page)
        assert len(responses.calls) == 1
