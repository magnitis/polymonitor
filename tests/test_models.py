"""
Tests for Pydantic data models.
"""

import pytest
from datetime import datetime

from polymonitor.models import (
    Market,
    Event,
    Outcome,
    Opportunity,
    ProbabilityEstimate,
    Tag,
    MarketStatus,
    ConvictionLevel,
    MarketType,
)


class TestMarket:
    """Tests for the Market model."""
    
    def test_market_creation_minimal(self):
        """Test creating a market with minimal data."""
        market = Market(question="Will it rain tomorrow?")
        
        assert market.question == "Will it rain tomorrow?"
        assert market.yes_price == 0.5  # Default when no prices
        assert market.status == MarketStatus.ACTIVE
    
    def test_market_creation_full(self):
        """Test creating a market with full data."""
        market = Market(
            id="123",
            conditionId="abc123",
            question="Will 2026 be the hottest year on record?",
            description="Based on NASA GISS data",
            outcomes=["Yes", "No"],
            outcomePrices=[0.65, 0.35],
            clobTokenIds=["token1", "token2"],
            active=True,
            closed=False,
            volume=50000,
            liquidity=10000,
        )
        
        assert market.id == "123"
        assert market.condition_id == "abc123"
        assert market.yes_price == 0.65
        assert market.no_price == 0.35
        assert market.status == MarketStatus.ACTIVE
        assert market.volume == 50000
    
    def test_market_outcome_prices_parsing(self):
        """Test parsing outcome prices from various formats."""
        # From list of floats
        market1 = Market(
            question="Test",
            outcomePrices=[0.7, 0.3],
        )
        assert market1.outcome_prices == [0.7, 0.3]
        
        # From JSON string
        market2 = Market(
            question="Test",
            outcomePrices='[0.8, 0.2]',
        )
        assert market2.outcome_prices == [0.8, 0.2]
    
    def test_market_status_resolved(self):
        """Test market status when resolved."""
        market = Market(
            question="Test",
            resolution="Yes",
        )
        
        assert market.status == MarketStatus.RESOLVED
    
    def test_market_status_closed(self):
        """Test market status when closed."""
        market = Market(
            question="Test",
            closed=True,
        )
        
        assert market.status == MarketStatus.CLOSED
    
    def test_get_outcome(self):
        """Test getting outcome objects."""
        market = Market(
            question="Test",
            outcomes=["Yes", "No"],
            outcomePrices=[0.6, 0.4],
        )
        
        yes_outcome = market.get_outcome(0)
        assert yes_outcome.name == "Yes"
        assert yes_outcome.price == 0.6
        assert yes_outcome.implied_probability == 0.6


class TestEvent:
    """Tests for the Event model."""
    
    def test_event_creation(self):
        """Test creating an event."""
        event = Event(
            id="event1",
            title="Climate 2026",
            description="Climate predictions for 2026",
        )
        
        assert event.id == "event1"
        assert event.title == "Climate 2026"
        assert event.active == True
    
    def test_event_with_markets(self):
        """Test event with nested markets."""
        event = Event(
            id="event1",
            title="Climate 2026",
            markets=[
                {"question": "Will 2026 be hottest?", "outcomePrices": [0.7, 0.3]},
                {"question": "Will sea ice hit record low?", "outcomePrices": [0.5, 0.5]},
            ],
        )
        
        assert len(event.markets) == 2
        assert event.markets[0].question == "Will 2026 be hottest?"
        assert event.markets[0].yes_price == 0.7
    
    def test_event_with_tags(self):
        """Test event with tags."""
        event = Event(
            id="event1",
            title="Climate 2026",
            tags=[
                {"id": "1", "label": "Climate"},
                {"id": "2", "label": "Weather"},
            ],
        )
        
        assert len(event.tags) == 2
        assert event.tag_labels == ["Climate", "Weather"]


class TestProbabilityEstimate:
    """Tests for ProbabilityEstimate model."""
    
    def test_estimate_creation(self):
        """Test creating a probability estimate."""
        estimate = ProbabilityEstimate(
            probability=0.75,
            conviction=ConvictionLevel.HIGH,
            model_name="temperature_model",
            reasoning="Based on historical trends",
            data_sources=["NASA GISS"],
        )
        
        assert estimate.probability == 0.75
        assert estimate.conviction == ConvictionLevel.HIGH
        assert estimate.model_name == "temperature_model"
    
    def test_estimate_with_factors(self):
        """Test estimate with factor breakdown."""
        estimate = ProbabilityEstimate(
            probability=0.8,
            conviction=ConvictionLevel.MEDIUM,
            model_name="test",
            factors={
                "trend": 0.3,
                "historical": 0.5,
            },
        )
        
        assert estimate.factors["trend"] == 0.3
        assert estimate.factors["historical"] == 0.5


class TestOpportunity:
    """Tests for Opportunity model."""
    
    def test_opportunity_creation(self):
        """Test creating an opportunity."""
        market = Market(
            question="Will 2026 be hottest?",
            outcomePrices=[0.5, 0.5],
            liquidity=10000,
        )
        
        estimate = ProbabilityEstimate(
            probability=0.7,
            conviction=ConvictionLevel.HIGH,
            model_name="test",
        )
        
        opp = Opportunity(
            market=market,
            estimate=estimate,
            market_probability=0.5,
            side="yes",
            edge=0.2,
        )
        
        assert opp.edge == 0.2
        assert opp.edge_percentage == 20.0
        assert opp.side == "yes"
    
    def test_opportunity_expected_value(self):
        """Test expected value calculation."""
        market = Market(
            question="Test",
            outcomePrices=[0.4, 0.6],
        )
        
        estimate = ProbabilityEstimate(
            probability=0.6,  # We think YES is 60%
            conviction=ConvictionLevel.MEDIUM,
            model_name="test",
        )
        
        opp = Opportunity(
            market=market,
            estimate=estimate,
            market_probability=0.4,  # Market says YES is 40%
            side="yes",
            edge=0.2,
        )
        
        # EV should be positive since we think YES is underpriced
        assert opp.expected_value > 0
    
    def test_opportunity_kelly_calculation(self):
        """Test Kelly Criterion bet sizing."""
        market = Market(
            question="Test",
            outcomePrices=[0.4, 0.6],
        )
        
        estimate = ProbabilityEstimate(
            probability=0.6,
            conviction=ConvictionLevel.HIGH,
            model_name="test",
        )
        
        opp = Opportunity(
            market=market,
            estimate=estimate,
            market_probability=0.4,
            side="yes",
            edge=0.2,
        )
        
        # Calculate Kelly for $1000 bankroll, full Kelly
        bet = opp.calculate_kelly(bankroll=1000, fraction=1.0)
        assert bet > 0
        assert bet <= 1000


class TestTag:
    """Tests for Tag model."""
    
    def test_tag_creation(self):
        """Test creating a tag."""
        tag = Tag(id="1", label="Climate", slug="climate")
        
        assert tag.id == "1"
        assert tag.label == "Climate"
        assert tag.slug == "climate"
