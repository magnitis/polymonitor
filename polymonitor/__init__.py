"""
Polymarket Climate Monitor
==========================

A production-ready Python package for monitoring Polymarket's climate science
markets and identifying betting opportunities.

Features:
- Real-time market monitoring via Gamma API
- Sophisticated probability estimation for climate markets
- Kelly Criterion-based bet sizing
- Automated opportunity detection
- Performance tracking and backtesting

Usage:
    from polymonitor import PolymarketClient, ClimateProbabilityModel, OpportunityMonitor
    
    client = PolymarketClient()
    model = ClimateProbabilityModel()
    monitor = OpportunityMonitor(client, model)
    
    opportunities = monitor.scan_once()
"""

__version__ = "0.1.0"
__author__ = "Polymonitor Team"

from polymonitor.models import (
    Market,
    Event,
    Outcome,
    Opportunity,
    Tag,
)
from polymonitor.config import Config, load_config
from polymonitor.api_client import PolymarketClient
from polymonitor.probability_model import (
    BaseProbabilityModel,
    ClimateProbabilityModel,
    TemperatureModel,
    HurricaneModel,
    EmissionsModel,
)
from polymonitor.monitor import OpportunityMonitor
from polymonitor.trading import TradingEngine, KellyCriterion

__all__ = [
    # Version
    "__version__",
    # Models
    "Market",
    "Event",
    "Outcome",
    "Opportunity",
    "Tag",
    # Config
    "Config",
    "load_config",
    # API
    "PolymarketClient",
    # Probability
    "BaseProbabilityModel",
    "ClimateProbabilityModel",
    "TemperatureModel",
    "HurricaneModel",
    "EmissionsModel",
    # Monitoring
    "OpportunityMonitor",
    # Trading
    "TradingEngine",
    "KellyCriterion",
]
