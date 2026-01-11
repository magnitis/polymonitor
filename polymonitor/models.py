"""
Data Models for Polymarket Climate Monitor
==========================================

Pydantic models for type-safe handling of Polymarket data structures.
These models are based on the Gamma API response formats.

The Gamma API returns data in the following hierarchy:
- Events contain multiple Markets
- Markets have Outcomes with prices
- Tags categorize Events/Markets
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, computed_field, field_validator


class MarketStatus(str, Enum):
    """Status of a market on Polymarket."""
    ACTIVE = "active"
    CLOSED = "closed"
    RESOLVED = "resolved"
    ARCHIVED = "archived"


class ConvictionLevel(str, Enum):
    """Conviction level for a probability estimate."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class MarketType(str, Enum):
    """Type of climate market for specialized probability models."""
    TEMPERATURE = "temperature"
    HURRICANE = "hurricane"
    EMISSIONS = "emissions"
    SEA_ICE = "sea_ice"
    WILDFIRE = "wildfire"
    PRECIPITATION = "precipitation"
    GENERAL_CLIMATE = "general_climate"
    UNKNOWN = "unknown"


class Tag(BaseModel):
    """
    Tag model representing a category/tag from Polymarket.
    
    Tags are used to categorize events and markets. Climate-related
    markets may have tags like "Weather", "Climate", "Environment", etc.
    """
    id: str = Field(..., description="Unique identifier for the tag")
    label: str = Field(..., description="Display label for the tag")
    slug: str = Field(default="", description="URL-friendly slug")
    
    class Config:
        extra = "allow"  # Allow additional fields from API


class Outcome(BaseModel):
    """
    Outcome model representing a possible outcome in a market.
    
    Each market typically has 2 outcomes (Yes/No) with associated
    prices that represent the market's implied probability.
    """
    id: str = Field(default="", description="Unique outcome identifier")
    name: str = Field(..., description="Outcome name (e.g., 'Yes', 'No')")
    price: float = Field(..., ge=0, le=1, description="Current price (0-1)")
    
    @computed_field
    @property
    def implied_probability(self) -> float:
        """The implied probability based on the price."""
        return self.price
    
    class Config:
        extra = "allow"


class Market(BaseModel):
    """
    Market model representing a prediction market on Polymarket.
    
    A market is a specific question with defined outcomes. For climate
    markets, this could be "Will 2026 be the hottest year on record?"
    
    Key fields:
    - condition_id: Unique identifier for the market condition
    - question: The market question being predicted
    - outcomes: List of possible outcomes with prices
    - volume: Total trading volume in USD
    - liquidity: Current liquidity in USD
    """
    
    # Core identifiers
    id: str = Field(default="", description="Market ID")
    condition_id: str = Field(default="", alias="conditionId", description="Condition ID for trading")
    question: str = Field(..., description="The market question")
    description: str = Field(default="", description="Detailed market description")
    
    # Market metadata
    slug: str = Field(default="", description="URL slug for the market")
    image: str = Field(default="", description="Market image URL")
    icon: str = Field(default="", description="Market icon URL")
    
    # Outcomes and prices
    outcomes: list[str] = Field(default_factory=list, description="Outcome names")
    outcome_prices: list[float] = Field(
        default_factory=list,
        alias="outcomePrices",
        description="Current prices for each outcome"
    )
    
    @field_validator("outcomes", mode="before")
    @classmethod
    def parse_outcomes(cls, v: Any) -> list[str]:
        """Parse outcomes which may come as a JSON string."""
        if isinstance(v, str):
            import json
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                return [v] if v else []
        if isinstance(v, list):
            return [str(o) for o in v]
        return []
    
    # Token identifiers (for trading)
    clob_token_ids: list[str] = Field(
        default_factory=list,
        alias="clobTokenIds",
        description="CLOB token IDs for trading"
    )
    
    # Market status
    active: bool = Field(default=True, description="Whether market is active")
    closed: bool = Field(default=False, description="Whether market is closed")
    archived: bool = Field(default=False, description="Whether market is archived")
    
    # Timestamps
    end_date: Optional[datetime] = Field(
        default=None,
        alias="endDate",
        description="When the market closes"
    )
    created_at: Optional[datetime] = Field(
        default=None,
        alias="createdAt",
        description="When the market was created"
    )
    
    # Trading metrics
    volume: float = Field(default=0, description="Total volume traded (USD)")
    volume_24h: float = Field(default=0, alias="volume24hr", description="24h volume (USD)")
    liquidity: float = Field(default=0, description="Current liquidity (USD)")
    
    # Resolution
    resolution: Optional[str] = Field(default=None, description="Resolution outcome if resolved")
    resolution_source: Optional[str] = Field(
        default=None,
        alias="resolutionSource",
        description="Source for resolution"
    )
    
    @field_validator("outcome_prices", mode="before")
    @classmethod
    def parse_outcome_prices(cls, v: Any) -> list[float]:
        """Parse outcome prices which may come as strings."""
        if isinstance(v, str):
            # Handle JSON string format
            import json
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                return []
        if isinstance(v, list):
            return [float(p) if isinstance(p, (int, float, str)) else 0.0 for p in v]
        return []
    
    @field_validator("clob_token_ids", mode="before")
    @classmethod
    def parse_clob_token_ids(cls, v: Any) -> list[str]:
        """Parse CLOB token IDs which may come as strings."""
        if isinstance(v, str):
            import json
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                return []
        if isinstance(v, list):
            return [str(t) for t in v]
        return []
    
    @computed_field
    @property
    def status(self) -> MarketStatus:
        """Determine the market status."""
        if self.archived:
            return MarketStatus.ARCHIVED
        if self.resolution:
            return MarketStatus.RESOLVED
        if self.closed:
            return MarketStatus.CLOSED
        return MarketStatus.ACTIVE
    
    @computed_field
    @property
    def yes_price(self) -> float:
        """Price for the 'Yes' outcome (first outcome)."""
        return self.outcome_prices[0] if self.outcome_prices else 0.5
    
    @computed_field
    @property
    def no_price(self) -> float:
        """Price for the 'No' outcome (second outcome)."""
        return self.outcome_prices[1] if len(self.outcome_prices) > 1 else 1 - self.yes_price
    
    def get_outcome(self, index: int = 0) -> Outcome:
        """Get an Outcome object for the specified index."""
        name = self.outcomes[index] if index < len(self.outcomes) else f"Outcome {index}"
        price = self.outcome_prices[index] if index < len(self.outcome_prices) else 0.5
        return Outcome(name=name, price=price)
    
    class Config:
        extra = "allow"
        populate_by_name = True


class Event(BaseModel):
    """
    Event model representing an event containing multiple markets.
    
    An event groups related markets together. For example, an event
    about "2026 Climate Records" might contain markets for:
    - Hottest year on record
    - Arctic sea ice minimum
    - Number of Category 5 hurricanes
    
    The Gamma API returns events which contain markets.
    """
    
    # Core identifiers
    id: str = Field(..., description="Event ID")
    slug: str = Field(default="", description="URL slug")
    title: str = Field(..., description="Event title")
    description: str = Field(default="", description="Event description")
    
    # Event metadata
    image: str = Field(default="", description="Event image URL")
    icon: str = Field(default="", description="Event icon URL")
    
    # Status
    active: bool = Field(default=True, description="Whether event is active")
    closed: bool = Field(default=False, description="Whether event is closed")
    archived: bool = Field(default=False, description="Whether event is archived")
    
    # Timestamps
    end_date: Optional[datetime] = Field(default=None, alias="endDate")
    created_at: Optional[datetime] = Field(default=None, alias="createdAt")
    
    # Metrics
    volume: float = Field(default=0, description="Total volume across all markets")
    liquidity: float = Field(default=0, description="Total liquidity")
    
    # Tags for categorization
    tags: list[Tag] = Field(default_factory=list, description="Event tags")
    
    # Markets within this event
    markets: list[Market] = Field(default_factory=list, description="Markets in this event")
    
    @field_validator("tags", mode="before")
    @classmethod
    def parse_tags(cls, v: Any) -> list[Tag]:
        """Parse tags which may come in various formats."""
        if not v:
            return []
        if isinstance(v, list):
            result = []
            for tag in v:
                if isinstance(tag, dict):
                    result.append(Tag(**tag))
                elif isinstance(tag, Tag):
                    result.append(tag)
            return result
        return []
    
    @field_validator("markets", mode="before")
    @classmethod
    def parse_markets(cls, v: Any) -> list[Market]:
        """Parse markets which may come in various formats."""
        if not v:
            return []
        if isinstance(v, list):
            result = []
            for market in v:
                if isinstance(market, dict):
                    result.append(Market(**market))
                elif isinstance(market, Market):
                    result.append(market)
            return result
        return []
    
    @computed_field
    @property
    def tag_labels(self) -> list[str]:
        """Get list of tag labels for easy filtering."""
        return [tag.label for tag in self.tags]
    
    class Config:
        extra = "allow"
        populate_by_name = True


class ProbabilityEstimate(BaseModel):
    """
    Model for a probability estimate from our models.
    
    Contains not just the probability but also metadata about
    confidence, reasoning, and data sources used.
    """
    
    probability: float = Field(..., ge=0, le=1, description="Estimated probability (0-1)")
    conviction: ConvictionLevel = Field(
        default=ConvictionLevel.LOW,
        description="Confidence in the estimate"
    )
    
    # Model metadata
    model_name: str = Field(..., description="Name of the model used")
    model_version: str = Field(default="1.0", description="Model version")
    
    # Reasoning and sources
    reasoning: str = Field(default="", description="Explanation of the estimate")
    data_sources: list[str] = Field(
        default_factory=list,
        description="Data sources used for estimate"
    )
    
    # Factors considered
    factors: dict[str, float] = Field(
        default_factory=dict,
        description="Individual factor contributions"
    )
    
    # Timestamps
    estimated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When estimate was made"
    )
    
    # Uncertainty
    confidence_interval: tuple[float, float] = Field(
        default=(0.0, 1.0),
        description="95% confidence interval"
    )
    
    class Config:
        extra = "allow"


class Opportunity(BaseModel):
    """
    Model representing a betting opportunity.
    
    An opportunity is identified when our probability estimate
    differs significantly from the market price.
    
    Key metrics:
    - edge: Difference between fair probability and market probability
    - expected_value: Expected value per dollar bet
    - kelly_fraction: Optimal bet size using Kelly Criterion
    """
    
    # Market reference
    market: Market = Field(..., description="The market with the opportunity")
    event_title: str = Field(default="", description="Parent event title")
    
    # Our estimate
    estimate: ProbabilityEstimate = Field(..., description="Our probability estimate")
    
    # Market probability
    market_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Current market probability"
    )
    
    # Opportunity metrics
    side: str = Field(..., description="Which side to bet ('yes' or 'no')")
    edge: float = Field(..., description="Edge = fair_prob - market_prob")
    
    # Bet sizing
    kelly_fraction: float = Field(
        default=0.0,
        ge=0,
        description="Kelly Criterion optimal bet fraction"
    )
    recommended_bet: float = Field(
        default=0.0,
        ge=0,
        description="Recommended bet size in USD"
    )
    
    # Ranking score
    score: float = Field(default=0.0, description="Composite ranking score")
    
    # Timestamps
    detected_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When opportunity was detected"
    )
    
    @computed_field
    @property
    def expected_value(self) -> float:
        """
        Calculate expected value per dollar bet.
        
        EV = (prob_win * payout) - (prob_lose * stake)
        For a $1 bet at the market price.
        """
        if self.side == "yes":
            prob_win = self.estimate.probability
            payout = 1 / self.market_probability if self.market_probability > 0 else 0
        else:
            prob_win = 1 - self.estimate.probability
            payout = 1 / (1 - self.market_probability) if self.market_probability < 1 else 0
        
        return (prob_win * (payout - 1)) - ((1 - prob_win) * 1)
    
    @computed_field
    @property
    def edge_percentage(self) -> float:
        """Edge as a percentage (e.g., 0.15 -> 15%)."""
        return abs(self.edge) * 100
    
    @computed_field
    @property
    def conviction_label(self) -> str:
        """Human-readable conviction level."""
        return self.estimate.conviction.value.replace("_", " ").title()
    
    def calculate_kelly(self, bankroll: float, fraction: float = 0.25) -> float:
        """
        Calculate Kelly Criterion bet size.
        
        Kelly formula: f* = (bp - q) / b
        where:
        - b = odds received on the bet (decimal - 1)
        - p = probability of winning
        - q = probability of losing (1 - p)
        
        Args:
            bankroll: Total bankroll in USD
            fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
        
        Returns:
            Recommended bet size in USD
        """
        if self.side == "yes":
            p = self.estimate.probability
            b = (1 / self.market_probability) - 1 if self.market_probability > 0 else 0
        else:
            p = 1 - self.estimate.probability
            b = (1 / (1 - self.market_probability)) - 1 if self.market_probability < 1 else 0
        
        q = 1 - p
        
        if b <= 0:
            return 0.0
        
        kelly = (b * p - q) / b
        kelly = max(0, kelly)  # Don't bet negative amounts
        
        # Apply fractional Kelly and calculate bet size
        bet_size = bankroll * kelly * fraction
        return round(bet_size, 2)
    
    class Config:
        extra = "allow"


class PerformanceRecord(BaseModel):
    """
    Model for tracking performance of identified opportunities.
    
    Used for backtesting and evaluating the probability model.
    """
    
    # Opportunity reference
    opportunity_id: str = Field(..., description="Unique opportunity ID")
    market_id: str = Field(..., description="Market ID")
    market_question: str = Field(..., description="Market question")
    
    # Predictions
    our_probability: float = Field(..., description="Our estimated probability")
    market_probability_at_detection: float = Field(
        ...,
        description="Market probability when detected"
    )
    
    # Outcome
    actual_outcome: Optional[str] = Field(
        default=None,
        description="Actual resolution outcome"
    )
    we_were_correct: Optional[bool] = Field(
        default=None,
        description="Whether our prediction was correct"
    )
    
    # Bet details (if taken)
    bet_taken: bool = Field(default=False, description="Whether bet was placed")
    bet_amount: float = Field(default=0, description="Amount bet")
    bet_side: str = Field(default="", description="Side bet on")
    profit_loss: float = Field(default=0, description="P&L from this bet")
    
    # Timestamps
    detected_at: datetime = Field(..., description="When opportunity was detected")
    resolved_at: Optional[datetime] = Field(
        default=None,
        description="When market resolved"
    )
    
    class Config:
        extra = "allow"


# Type aliases for convenience
Markets = list[Market]
Events = list[Event]
Opportunities = list[Opportunity]
