"""
Configuration Management for Polymarket Climate Monitor
========================================================

Handles loading and validation of configuration from:
1. YAML configuration file (config.yaml)
2. Environment variables
3. Command-line overrides

Configuration priority (highest to lowest):
1. Command-line arguments
2. Environment variables  
3. config.local.yaml (user overrides)
4. config.yaml (defaults)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    initial_backoff: float = 1.0
    max_backoff: float = 60.0
    backoff_multiplier: float = 2.0


class APIConfig(BaseModel):
    """API configuration."""
    base_url: str = "https://gamma-api.polymarket.com"
    timeout: int = 30
    max_retries: int = 5
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    page_size: int = 100


class CacheTTLConfig(BaseModel):
    """Cache TTL configuration for different data types."""
    markets: int = 300  # 5 minutes
    tags: int = 3600  # 1 hour
    events: int = 600  # 10 minutes
    historical: int = 86400  # 24 hours


class CacheConfig(BaseModel):
    """Caching configuration."""
    enabled: bool = True
    directory: str = ".cache/polymonitor"
    ttl: CacheTTLConfig = Field(default_factory=CacheTTLConfig)


class FiltersConfig(BaseModel):
    """Market filtering configuration."""
    climate_keywords: list[str] = Field(default_factory=lambda: [
        "temperature", "climate", "weather", "hurricane", "typhoon",
        "cyclone", "storm", "wildfire", "drought", "flood", "sea level",
        "ice", "arctic", "antarctic", "emissions", "carbon", "CO2",
        "greenhouse", "el niño", "la niña", "NOAA", "IPCC", "hottest",
        "coldest", "record high", "record low", "heat wave", "polar vortex"
    ])
    min_liquidity: float = 1000
    active_only: bool = True


class TemperatureModelConfig(BaseModel):
    """Temperature probability model configuration."""
    lookback_years: int = 30
    trend_weight: float = 0.3
    percentile_weight: float = 0.7


class HurricaneModelConfig(BaseModel):
    """Hurricane probability model configuration."""
    use_ace_index: bool = True
    seasonal_weight: float = 0.4


class EmissionsModelConfig(BaseModel):
    """Emissions probability model configuration."""
    trend_years: int = 10


class ConvictionConfig(BaseModel):
    """Conviction threshold configuration."""
    low: float = 0.6
    medium: float = 0.7
    high: float = 0.85


class ProbabilityModelConfig(BaseModel):
    """Probability model configuration."""
    default: str = "climate_composite"
    temperature: TemperatureModelConfig = Field(default_factory=TemperatureModelConfig)
    hurricane: HurricaneModelConfig = Field(default_factory=HurricaneModelConfig)
    emissions: EmissionsModelConfig = Field(default_factory=EmissionsModelConfig)
    conviction: ConvictionConfig = Field(default_factory=ConvictionConfig)


class EdgeThresholdsConfig(BaseModel):
    """Edge threshold configuration for opportunity categorization."""
    good: float = 0.15
    great: float = 0.25
    excellent: float = 0.35


class RankingConfig(BaseModel):
    """Opportunity ranking weights."""
    edge_weight: float = 0.5
    liquidity_weight: float = 0.3
    conviction_weight: float = 0.2


class OpportunitiesConfig(BaseModel):
    """Opportunity detection configuration."""
    min_edge: float = 0.15
    edge_thresholds: EdgeThresholdsConfig = Field(default_factory=EdgeThresholdsConfig)
    ranking: RankingConfig = Field(default_factory=RankingConfig)


class KellyConfig(BaseModel):
    """Kelly Criterion configuration."""
    fraction: float = 0.25  # Quarter Kelly for safety
    max_bet_fraction: float = 0.05


class RiskConfig(BaseModel):
    """Risk management configuration."""
    max_bet_per_market: float = 100
    max_total_exposure: float = 1000
    max_open_positions: int = 10
    min_auto_trade_edge: float = 0.20


class SlippageConfig(BaseModel):
    """Slippage protection configuration."""
    max_slippage: float = 0.02
    use_limit_orders: bool = True


class TradingConfig(BaseModel):
    """Trading configuration."""
    enabled: bool = False
    chain_id: int = 137
    clob_host: str = "https://clob.polymarket.com"
    kelly: KellyConfig = Field(default_factory=KellyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    interval: int = 300  # 5 minutes
    log_file: str = "logs/polymonitor.log"
    log_level: str = "INFO"
    opportunities_file: str = "data/opportunities.json"
    track_performance: bool = True
    performance_file: str = "data/performance.json"


class DiscordConfig(BaseModel):
    """Discord notification configuration."""
    enabled: bool = False
    min_edge_notify: float = 0.25


class EmailConfig(BaseModel):
    """Email notification configuration."""
    enabled: bool = False


class NotificationsConfig(BaseModel):
    """Notifications configuration."""
    enabled: bool = False
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    email: EmailConfig = Field(default_factory=EmailConfig)


class DashboardConfig(BaseModel):
    """Dashboard configuration."""
    host: str = "localhost"
    port: int = 8501
    refresh_interval: int = 60


class BacktestConfig(BaseModel):
    """Backtesting configuration."""
    data_dir: str = "data/historical"
    start_date: str = "2024-01-01"
    initial_bankroll: float = 10000


class Config(BaseModel):
    """
    Main configuration model for Polymarket Climate Monitor.
    
    This aggregates all configuration sections and provides
    validation and type safety.
    """
    api: APIConfig = Field(default_factory=APIConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    probability_model: ProbabilityModelConfig = Field(default_factory=ProbabilityModelConfig)
    opportunities: OpportunitiesConfig = Field(default_factory=OpportunitiesConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    notifications: NotificationsConfig = Field(default_factory=NotificationsConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    
    class Config:
        extra = "allow"


class EnvironmentSettings(BaseSettings):
    """
    Environment-based settings for sensitive data.
    
    These should NEVER be stored in config files. Use environment
    variables or a .env file (which should be gitignored).
    """
    
    model_config = SettingsConfigDict(
        env_prefix="POLYMARKET_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Trading credentials
    private_key: Optional[SecretStr] = None
    funder_address: Optional[str] = None
    api_key: Optional[SecretStr] = None
    
    # Notification credentials
    discord_webhook_url: Optional[SecretStr] = Field(
        default=None,
        validation_alias="DISCORD_WEBHOOK_URL"
    )
    
    # External API keys
    noaa_api_key: Optional[SecretStr] = None
    openweather_api_key: Optional[SecretStr] = None
    
    # App settings
    log_level: str = "INFO"
    config_path: str = "config.yaml"
    cache_dir: str = ".cache/polymonitor"
    data_dir: str = "data"


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
    
    Returns:
        Dictionary containing the configuration
    """
    if not config_path.exists():
        return {}
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge two configuration dictionaries.
    
    Args:
        base: Base configuration
        override: Override configuration (takes precedence)
    
    Returns:
        Merged configuration dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def load_config(
    config_path: Optional[str] = None,
    local_config_path: Optional[str] = None
) -> Config:
    """
    Load and validate configuration from files and environment.
    
    Configuration is loaded in the following order (later overrides earlier):
    1. config.yaml (defaults)
    2. config.local.yaml (user overrides, gitignored)
    3. Environment variables (for sensitive data)
    
    Args:
        config_path: Path to main config file (default: config.yaml)
        local_config_path: Path to local config file (default: config.local.yaml)
    
    Returns:
        Validated Config object
    
    Example:
        >>> config = load_config()
        >>> print(config.api.base_url)
        'https://gamma-api.polymarket.com'
    """
    # Determine paths
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    
    base_path = Path(config_path)
    
    if local_config_path is None:
        local_config_path = base_path.stem + ".local" + base_path.suffix
    
    local_path = Path(local_config_path)
    
    # Load configurations
    base_config = load_yaml_config(base_path)
    local_config = load_yaml_config(local_path)
    
    # Merge configurations
    merged_config = merge_configs(base_config, local_config)
    
    # Create and validate config
    return Config(**merged_config)


def get_env_settings() -> EnvironmentSettings:
    """
    Get environment-based settings.
    
    Returns:
        EnvironmentSettings object with validated environment variables
    """
    return EnvironmentSettings()


# Global config instance (lazy-loaded)
_config: Optional[Config] = None
_env_settings: Optional[EnvironmentSettings] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Loads configuration on first call and caches it.
    
    Returns:
        Config object
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_env() -> EnvironmentSettings:
    """
    Get the global environment settings instance.
    
    Returns:
        EnvironmentSettings object
    """
    global _env_settings
    if _env_settings is None:
        _env_settings = get_env_settings()
    return _env_settings


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config, _env_settings
    _config = None
    _env_settings = None
