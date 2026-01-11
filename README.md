# 🌍 Polymarket Climate Monitor

A production-ready Python tool for monitoring Polymarket's climate science prediction markets and identifying betting opportunities.

## Features

- **Real-time Market Monitoring**: Fetch and track climate-related prediction markets via Polymarket's Gamma API
- **Sophisticated Probability Models**: Built-in models for temperature, hurricane, emissions, and sea ice markets
- **Opportunity Detection**: Automatically identify mispriced markets with configurable edge thresholds
- **Kelly Criterion Bet Sizing**: Optimal bet sizing with fractional Kelly for safety
- **Trading Integration**: (Optional) Execute trades via py-clob-client (disabled by default)
- **Web Dashboard**: Beautiful Streamlit dashboard for visualizing opportunities
- **Extensible Architecture**: Easy to add custom probability models and data sources

## Quick Start

### Installation

```bash
# Clone the repository
cd polymonitor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. Copy the example environment file:
```bash
cp env.example .env
```

2. (Optional) Customize `config.yaml` for your preferences:
```yaml
opportunities:
  min_edge: 0.15  # Minimum 15% edge to flag opportunity
  
filters:
  min_liquidity: 1000  # Minimum $1000 liquidity
```

### Basic Usage

```bash
# Scan for opportunities once
python main.py scan

# Continuous monitoring (every 5 minutes)
python main.py monitor --interval 300

# Launch web dashboard
python main.py dashboard

# Export opportunities to CSV
python main.py export --output my_opportunities.csv
```

## Architecture

```
polymonitor/
├── main.py              # CLI entry point
├── dashboard.py         # Streamlit web dashboard
├── config.yaml          # Configuration file
├── polymonitor/
│   ├── __init__.py
│   ├── api_client.py    # Polymarket Gamma API client
│   ├── config.py        # Configuration management
│   ├── models.py        # Pydantic data models
│   ├── probability_model.py  # Probability estimation models
│   ├── monitor.py       # Opportunity detection & monitoring
│   └── trading.py       # Trading engine (disabled by default)
├── tests/               # Unit tests
└── data/                # Logs and opportunity history
```

## API Integration

This tool uses the [Polymarket Gamma API](https://docs.polymarket.com/developers/gamma-markets-api/fetch-markets-guide):

```python
from polymonitor import PolymarketClient

# Create client
client = PolymarketClient()

# Fetch climate-related events
events = client.get_climate_events()

# Get all markets with pagination
all_markets = client.get_all_markets(active=True)

# Search for specific markets
results = client.search_events("hurricane 2026")
```

### Rate Limiting

The client automatically handles rate limiting with exponential backoff:

```python
# Configured in config.yaml
api:
  rate_limit:
    requests_per_minute: 60
    initial_backoff: 1.0
    max_backoff: 60.0
```

### Caching

Responses are cached to minimize API calls:

```python
# Clear cache when needed
client.clear_cache()
```

## Probability Models

### Built-in Models

| Model | Markets | Methodology |
|-------|---------|-------------|
| `TemperatureModel` | Hottest year, temperature records | Historical percentile analysis, trend projection |
| `HurricaneModel` | Storm counts, Category 5 | NOAA historical data, seasonal forecasts |
| `EmissionsModel` | CO2 levels, emission targets | Trend analysis, policy assessment |
| `SeaIceModel` | Arctic ice extent | Historical minimum analysis |
| `WildfireModel` | Fire statistics | Climate-adjusted trends |

### Using the Models

```python
from polymonitor import ClimateProbabilityModel, PolymarketClient

# Create model
model = ClimateProbabilityModel()

# Get markets
client = PolymarketClient()
markets = client.get_climate_markets()

# Estimate probability for each market
for market in markets:
    estimate = model.estimate(market)
    print(f"{market.question[:50]}...")
    print(f"  Market: {market.yes_price:.1%}")
    print(f"  Fair:   {estimate.probability:.1%}")
    print(f"  Edge:   {(estimate.probability - market.yes_price)*100:+.1f}%")
```

### Creating Custom Models

Extend the base model to add your own logic:

```python
from polymonitor.probability_model import BaseProbabilityModel, MarketType
from polymonitor.models import Market, ProbabilityEstimate, ConvictionLevel

class MyCustomModel(BaseProbabilityModel):
    name = "my_custom_model"
    version = "1.0"
    
    def estimate(self, market: Market) -> ProbabilityEstimate:
        # Your custom probability estimation logic
        my_probability = self._analyze_market(market)
        
        return self._create_estimate(
            probability=my_probability,
            conviction=ConvictionLevel.MEDIUM,
            reasoning="Based on my custom analysis...",
            data_sources=["My Data Source"],
            factors={"key_factor": 0.7},
        )
    
    def _analyze_market(self, market: Market) -> float:
        # Add your analysis here
        # - Query external APIs
        # - Use ML models
        # - Apply domain expertise
        return 0.65

# Register with composite model
from polymonitor import ClimateProbabilityModel

composite = ClimateProbabilityModel()
composite.register_model(MarketType.TEMPERATURE, MyCustomModel())
```

## Opportunity Detection

```python
from polymonitor import PolymarketClient, ClimateProbabilityModel
from polymonitor.monitor import OpportunityMonitor

# Initialize
client = PolymarketClient()
model = ClimateProbabilityModel()
monitor = OpportunityMonitor(client, model)

# Scan for opportunities
opportunities = monitor.scan_once(min_edge=0.15)

# Display results
for opp in opportunities[:5]:
    print(f"🎯 {opp.side.upper()}: {opp.market.question[:50]}...")
    print(f"   Edge: {opp.edge_percentage:.1f}%")
    print(f"   Kelly: {opp.kelly_fraction:.1%}")
    print(f"   Score: {opp.score:.2f}")
```

### Opportunity Ranking

Opportunities are ranked by a composite score:

```
Score = (edge_weight × normalized_edge) 
      + (liquidity_weight × log_normalized_liquidity)
      + (conviction_weight × conviction_score)
```

Configure weights in `config.yaml`:

```yaml
opportunities:
  ranking:
    edge_weight: 0.5
    liquidity_weight: 0.3
    conviction_weight: 0.2
```

## Trading (Disabled by Default)

⚠️ **WARNING**: Trading is disabled by default. Only enable with funds you can afford to lose.

### Enable Trading

Trading requires **both**:
1. The `--enable-trading` flag on CLI commands
2. Valid credentials in `.env`

#### Step 1: Set credentials in `.env`:
```bash
POLYMARKET_PRIVATE_KEY=your_private_key
POLYMARKET_FUNDER_ADDRESS=your_funder_address
```

#### Step 2: Use the `--enable-trading` flag:
```bash
# Execute trades interactively
python main.py --enable-trading trade

# Auto-trade all opportunities above 25% edge
python main.py --enable-trading trade --auto --min-edge 0.25

# Monitor with auto-trading enabled
python main.py --enable-trading monitor --auto-trade
```

Alternatively, set `trading.enabled: true` in `config.yaml` to enable via config.

### Kelly Criterion Bet Sizing

The tool uses fractional Kelly for safer bet sizing:

```python
from polymonitor.trading import KellyCriterion

kelly = KellyCriterion(fraction=0.25)  # Quarter Kelly

bet_size = kelly.calculate(
    bankroll=1000,
    true_prob=0.7,      # Our estimate
    market_price=0.5,   # Market price
    side="yes"
)
print(f"Recommended bet: ${bet_size:.2f}")
```

### Risk Management

```yaml
trading:
  risk:
    max_bet_per_market: 100    # Max $100 per market
    max_total_exposure: 1000   # Max $1000 total
    max_open_positions: 10     # Max 10 positions
    min_auto_trade_edge: 0.20  # 20% edge for auto-trade
```

### Simulated Trading

Test strategies without real money:

```python
from polymonitor.trading import SimulatedTradingEngine

engine = SimulatedTradingEngine(initial_bankroll=10000)

# Execute simulated trades
result = engine.execute_opportunity(opportunity, bankroll=10000)

# Check performance
print(f"P&L: ${engine.pnl:.2f}")
print(f"ROI: {engine.roi:.1f}%")
```

## Notifications

### Discord

```yaml
notifications:
  enabled: true
  discord:
    enabled: true
    min_edge_notify: 0.25
```

Set webhook URL in `.env`:
```bash
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

### Custom Notifications

```python
from polymonitor.monitor import OpportunityMonitor

def my_notifier(opportunity):
    # Send to Slack, email, SMS, etc.
    print(f"New opportunity: {opportunity.edge_percentage:.1f}% edge!")

monitor = OpportunityMonitor(client, model)
monitor.add_notification_callback(my_notifier)
```

## Web Dashboard

Launch the Streamlit dashboard:

```bash
python main.py dashboard
```

Or directly:

```bash
streamlit run dashboard.py
```

Features:
- Real-time opportunity display
- Analytics and charts
- Historical performance tracking
- CSV export

## CLI Reference

```bash
# General options
python main.py --help
python main.py --config config.local.yaml scan
python main.py --log-level DEBUG scan
python main.py --enable-trading trade  # Enable live trading

# Scan command
python main.py scan
python main.py scan --min-edge 0.10
python main.py scan --min-liquidity 5000

# Monitor command
python main.py monitor
python main.py monitor --interval 600  # 10 minutes
python main.py --enable-trading monitor --auto-trade  # With auto-trading

# Trade command (requires --enable-trading)
python main.py --enable-trading trade
python main.py --enable-trading trade --auto  # Auto-trade all opportunities
python main.py --enable-trading trade --min-edge 0.25  # Higher threshold
python main.py --enable-trading trade --yes  # Skip confirmations (dangerous!)

# Export command
python main.py export
python main.py export --output my_data.csv

# Dashboard command
python main.py dashboard
python main.py dashboard --port 8080

# Utility commands
python main.py performance
python main.py clear-cache
python main.py backtest --start 2024-01-01 --bankroll 10000
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=polymonitor --cov-report=html

# Run specific test file
pytest tests/test_probability_model.py

# Run specific test
pytest tests/test_models.py::TestMarket::test_market_creation_full
```

## Project Structure

```
polymonitor/
├── main.py                 # CLI entry point
├── dashboard.py            # Streamlit dashboard
├── config.yaml             # Default configuration
├── env.example             # Environment template
├── requirements.txt        # Dependencies
│
├── polymonitor/
│   ├── __init__.py         # Package exports
│   ├── api_client.py       # Gamma API client
│   ├── config.py           # Configuration loading
│   ├── models.py           # Pydantic models
│   ├── probability_model.py # Probability estimation
│   ├── monitor.py          # Opportunity detection
│   └── trading.py          # Trading engine
│
├── tests/
│   ├── test_api_client.py
│   ├── test_models.py
│   ├── test_probability_model.py
│   └── test_trading.py
│
├── data/                   # Generated data
│   ├── opportunities.json  # Logged opportunities
│   └── performance.json    # Performance tracking
│
├── logs/                   # Log files
└── .cache/                 # API cache
```

## Configuration Reference

### config.yaml

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `api.base_url` | - | `https://gamma-api.polymarket.com` | Gamma API URL |
| `api.timeout` | - | `30` | Request timeout (seconds) |
| `api.rate_limit.requests_per_minute` | - | `60` | Rate limit |
| `cache.enabled` | - | `true` | Enable response caching |
| `cache.ttl.markets` | - | `300` | Market cache TTL (seconds) |
| `filters.climate_keywords` | - | `[temperature, hurricane, ...]` | Keywords for filtering |
| `filters.min_liquidity` | - | `1000` | Minimum liquidity (USD) |
| `opportunities.min_edge` | - | `0.15` | Minimum edge threshold |
| `trading.enabled` | - | `false` | Enable trading |
| `trading.kelly.fraction` | - | `0.25` | Fraction of Kelly to use |
| `monitoring.interval` | - | `300` | Scan interval (seconds) |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `POLYMARKET_PRIVATE_KEY` | For trading | Wallet private key |
| `POLYMARKET_FUNDER_ADDRESS` | For trading | Funder address |
| `DISCORD_WEBHOOK_URL` | For notifications | Discord webhook |
| `NOAA_API_KEY` | Optional | NOAA data access |
| `LOG_LEVEL` | Optional | Logging level |

## Extending the Tool

### Adding New Market Types

1. Define the market type in `models.py`:
```python
class MarketType(str, Enum):
    # ...existing types...
    VOLCANO = "volcano"
```

2. Create the model in `probability_model.py`:
```python
class VolcanoModel(BaseProbabilityModel):
    name = "volcano_model"
    
    def estimate(self, market):
        # Your logic
        pass
```

3. Register with composite model:
```python
composite.register_model(MarketType.VOLCANO, VolcanoModel())
```

### Adding Data Sources

```python
class EnhancedTemperatureModel(TemperatureModel):
    def estimate(self, market):
        # Fetch real-time data
        noaa_data = self._fetch_noaa_data()
        
        # Combine with base estimate
        base_estimate = super().estimate(market)
        
        # Adjust based on new data
        adjusted_prob = self._adjust_probability(
            base_estimate.probability,
            noaa_data
        )
        
        return self._create_estimate(
            probability=adjusted_prob,
            # ...
        )
    
    def _fetch_noaa_data(self):
        # Query NOAA API
        pass
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Disclaimer

This tool is for educational and research purposes only. Prediction markets involve financial risk. Never bet more than you can afford to lose. Past performance does not guarantee future results.

The probability models provided are demonstrations and should not be relied upon for actual trading decisions without extensive validation and customization.

## Support

- **Issues**: GitHub Issues
- **Documentation**: This README
- **API Reference**: [Polymarket Docs](https://docs.polymarket.com/)
