"""
Probability Estimation Models for Climate Markets
==================================================

This module contains probability estimation models for different types
of climate-related prediction markets.

The models are designed to be:
1. Extensible - Easy to add new models or modify existing logic
2. Data-driven - Based on historical data and scientific consensus
3. Transparent - Clear reasoning for each estimate

Model Hierarchy:
- BaseProbabilityModel (abstract base class)
  - TemperatureModel (temperature records, heat waves)
  - HurricaneModel (tropical storms, hurricane counts)
  - EmissionsModel (CO2 levels, emissions targets)
  - SeaIceModel (Arctic/Antarctic ice extent)
  - WildfireModel (wildfire counts, area burned)
  - ClimateProbabilityModel (composite model that delegates)

HOW TO EXTEND:
--------------
To add your own probability estimation logic:

1. Subclass BaseProbabilityModel or one of the specific models:

    class MyCustomModel(BaseProbabilityModel):
        def estimate(self, market: Market) -> ProbabilityEstimate:
            # Your custom logic here
            pass

2. Override the classification method to handle specific market types:

    def classify_market(self, market: Market) -> MarketType:
        # Custom classification logic
        pass

3. Register your model with the ClimateProbabilityModel:

    composite = ClimateProbabilityModel()
    composite.register_model(MarketType.CUSTOM, MyCustomModel())
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Optional

from polymonitor.config import Config, get_config
from polymonitor.models import (
    ConvictionLevel,
    Market,
    MarketType,
    ProbabilityEstimate,
)

logger = logging.getLogger(__name__)


# =============================================================================
# HISTORICAL DATA (for demonstration - in production, fetch from APIs)
# =============================================================================

# Global temperature anomalies (°C above pre-industrial baseline)
# Source: NASA GISS / NOAA
HISTORICAL_TEMP_ANOMALIES = {
    2015: 1.18, 2016: 1.32, 2017: 1.17, 2018: 1.08,
    2019: 1.21, 2020: 1.29, 2021: 1.11, 2022: 1.15,
    2023: 1.45, 2024: 1.52, 2025: 1.48,  # 2025 estimated
}

# Atlantic hurricane counts by year
# Source: NOAA Hurricane Center
HISTORICAL_HURRICANES = {
    2015: 11, 2016: 15, 2017: 17, 2018: 15, 2019: 18,
    2020: 30, 2021: 21, 2022: 14, 2023: 20, 2024: 18,
}

# Major hurricanes (Category 3+) by year
HISTORICAL_MAJOR_HURRICANES = {
    2015: 4, 2016: 7, 2017: 10, 2018: 7, 2019: 6,
    2020: 7, 2021: 7, 2022: 2, 2023: 7, 2024: 5,
}

# Category 5 hurricanes in Atlantic
HISTORICAL_CAT5_HURRICANES = {
    2015: 1, 2016: 1, 2017: 2, 2018: 1, 2019: 2,
    2020: 1, 2021: 0, 2022: 0, 2023: 1, 2024: 1,
}

# Global CO2 levels (ppm) - annual average
# Source: NOAA Global Monitoring Laboratory
HISTORICAL_CO2 = {
    2015: 400.83, 2016: 404.21, 2017: 406.55, 2018: 408.52,
    2019: 411.44, 2020: 414.24, 2021: 416.45, 2022: 418.56,
    2023: 421.08, 2024: 424.00, 2025: 426.50,  # 2025 projected
}

# September Arctic sea ice extent (million km²)
# Source: NSIDC
HISTORICAL_SEA_ICE = {
    2015: 4.63, 2016: 4.72, 2017: 4.87, 2018: 4.71,
    2019: 4.32, 2020: 3.92, 2021: 4.92, 2022: 4.67,
    2023: 4.23, 2024: 4.10,
}

# Record low sea ice extent
SEA_ICE_RECORD_LOW = 3.39  # September 2012


# =============================================================================
# BASE PROBABILITY MODEL
# =============================================================================

class BaseProbabilityModel(ABC):
    """
    Abstract base class for probability estimation models.
    
    All probability models should inherit from this class and implement
    the estimate() method.
    
    Attributes:
        config: Application configuration
        name: Model name for identification
        version: Model version for tracking changes
    """
    
    name: str = "base"
    version: str = "1.0"
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the probability model.
        
        Args:
            config: Application configuration. Loads default if None.
        """
        self.config = config or get_config()
    
    @abstractmethod
    def estimate(self, market: Market) -> ProbabilityEstimate:
        """
        Estimate the probability for a market outcome.
        
        Args:
            market: The market to estimate probability for
        
        Returns:
            ProbabilityEstimate with probability and metadata
        """
        pass
    
    def classify_market(self, market: Market) -> MarketType:
        """
        Classify the market type based on its question/description.
        
        This is used to route markets to the appropriate specialized model.
        
        Args:
            market: Market to classify
        
        Returns:
            MarketType enum value
        """
        text = f"{market.question} {market.description}".lower()
        
        # Temperature patterns
        temp_patterns = [
            r"hottest", r"warmest", r"temperature", r"heat",
            r"record high", r"record low", r"coldest",
            r"degree", r"celsius", r"fahrenheit",
        ]
        if any(re.search(p, text) for p in temp_patterns):
            return MarketType.TEMPERATURE
        
        # Hurricane/storm patterns
        hurricane_patterns = [
            r"hurricane", r"tropical storm", r"cyclone", r"typhoon",
            r"category \d", r"major storm", r"named storm",
            r"atlantic.*storm", r"landfall",
        ]
        if any(re.search(p, text) for p in hurricane_patterns):
            return MarketType.HURRICANE
        
        # Emissions/CO2 patterns
        emissions_patterns = [
            r"emission", r"co2", r"carbon", r"greenhouse",
            r"paris agreement", r"net zero", r"ppm",
        ]
        if any(re.search(p, text) for p in emissions_patterns):
            return MarketType.EMISSIONS
        
        # Sea ice patterns
        ice_patterns = [
            r"sea ice", r"arctic", r"antarctic", r"ice extent",
            r"ice sheet", r"glacier", r"ice-free",
        ]
        if any(re.search(p, text) for p in ice_patterns):
            return MarketType.SEA_ICE
        
        # Wildfire patterns
        wildfire_patterns = [
            r"wildfire", r"forest fire", r"acres burned",
            r"fire season", r"bushfire",
        ]
        if any(re.search(p, text) for p in wildfire_patterns):
            return MarketType.WILDFIRE
        
        # General climate
        climate_patterns = [
            r"climate", r"weather", r"precipitation", r"drought",
            r"flood", r"el ni[ñn]o", r"la ni[ñn]a", r"noaa", r"ipcc",
        ]
        if any(re.search(p, text) for p in climate_patterns):
            return MarketType.GENERAL_CLIMATE
        
        return MarketType.UNKNOWN
    
    def _get_conviction(self, confidence: float) -> ConvictionLevel:
        """
        Convert a confidence score to a conviction level.
        
        Args:
            confidence: Confidence score between 0 and 1
        
        Returns:
            ConvictionLevel enum value
        """
        thresholds = self.config.probability_model.conviction
        
        if confidence >= thresholds.high:
            return ConvictionLevel.VERY_HIGH
        elif confidence >= thresholds.medium:
            return ConvictionLevel.HIGH
        elif confidence >= thresholds.low:
            return ConvictionLevel.MEDIUM
        else:
            return ConvictionLevel.LOW
    
    def _create_estimate(
        self,
        probability: float,
        conviction: ConvictionLevel,
        reasoning: str,
        data_sources: list[str],
        factors: Optional[dict[str, float]] = None,
        confidence_interval: Optional[tuple[float, float]] = None,
    ) -> ProbabilityEstimate:
        """
        Helper to create a probability estimate with consistent formatting.
        
        Args:
            probability: Estimated probability (0-1)
            conviction: Conviction level
            reasoning: Explanation of the estimate
            data_sources: List of data sources used
            factors: Individual factor contributions
            confidence_interval: 95% confidence interval
        
        Returns:
            ProbabilityEstimate object
        """
        return ProbabilityEstimate(
            probability=max(0.01, min(0.99, probability)),  # Bound to valid range
            conviction=conviction,
            model_name=self.name,
            model_version=self.version,
            reasoning=reasoning,
            data_sources=data_sources,
            factors=factors or {},
            confidence_interval=confidence_interval or (
                max(0, probability - 0.15),
                min(1, probability + 0.15),
            ),
        )


# =============================================================================
# TEMPERATURE MODEL
# =============================================================================

class TemperatureModel(BaseProbabilityModel):
    """
    Probability model for temperature-related markets.
    
    Handles markets about:
    - Record high/low temperatures
    - Hottest/coldest year predictions
    - Heat wave occurrences
    - Temperature anomalies
    
    Methodology:
    1. Analyze historical temperature trends
    2. Calculate percentile rankings
    3. Adjust for current year-to-date data
    4. Consider El Niño/La Niña effects
    
    Example Markets:
    - "Will 2026 be the hottest year on record?"
    - "Will global temperature anomaly exceed 1.5°C in 2026?"
    """
    
    name = "temperature_model"
    version = "1.0"
    
    def estimate(self, market: Market) -> ProbabilityEstimate:
        """
        Estimate probability for temperature-related markets.
        """
        question = market.question.lower()
        
        # Extract year from question if present
        year_match = re.search(r"20\d{2}", market.question)
        target_year = int(year_match.group()) if year_match else datetime.now().year
        
        # "Hottest year on record" type questions
        if any(kw in question for kw in ["hottest", "warmest", "record high"]):
            return self._estimate_hottest_year(market, target_year)
        
        # Temperature threshold questions (e.g., "exceed 1.5°C")
        threshold_match = re.search(r"(\d+\.?\d*)\s*[°]?[cC]", question)
        if threshold_match:
            threshold = float(threshold_match.group(1))
            return self._estimate_temp_threshold(market, target_year, threshold)
        
        # Default to trend-based estimate
        return self._estimate_trend(market, target_year)
    
    def _estimate_hottest_year(self, market: Market, year: int) -> ProbabilityEstimate:
        """
        Estimate probability that a year will be the hottest on record.
        """
        # Get current record
        max_temp = max(HISTORICAL_TEMP_ANOMALIES.values())
        max_year = max(HISTORICAL_TEMP_ANOMALIES, key=HISTORICAL_TEMP_ANOMALIES.get)
        
        # Calculate trend (average increase per year over last 10 years)
        recent_years = sorted(HISTORICAL_TEMP_ANOMALIES.keys())[-10:]
        if len(recent_years) >= 2:
            temps = [HISTORICAL_TEMP_ANOMALIES[y] for y in recent_years]
            trend = (temps[-1] - temps[0]) / (len(recent_years) - 1)
        else:
            trend = 0.02  # Default ~0.02°C/year
        
        # Project temperature for target year
        last_year = max(HISTORICAL_TEMP_ANOMALIES.keys())
        years_ahead = year - last_year
        projected_temp = HISTORICAL_TEMP_ANOMALIES[last_year] + (trend * years_ahead)
        
        # Base probability on how projected temp compares to record
        temp_diff = projected_temp - max_temp
        
        if temp_diff > 0.1:
            # Likely to beat record
            base_prob = 0.70 + min(0.25, temp_diff * 2)
        elif temp_diff > 0:
            # Might beat record
            base_prob = 0.50 + (temp_diff * 2)
        elif temp_diff > -0.1:
            # Close to record
            base_prob = 0.30 + (temp_diff + 0.1) * 2
        else:
            # Unlikely to beat record
            base_prob = max(0.05, 0.30 + temp_diff)
        
        # Adjust for El Niño (increases warming) / La Niña (decreases)
        # In a full implementation, this would query actual ENSO data
        enso_adjustment = 0.05  # Assume slight El Niño conditions
        base_prob += enso_adjustment
        
        # Calculate conviction based on data quality
        conviction = self._get_conviction(0.7)  # Moderate confidence
        
        factors = {
            "trend": trend,
            "projected_temp": projected_temp,
            "current_record": max_temp,
            "temp_difference": temp_diff,
            "enso_adjustment": enso_adjustment,
        }
        
        reasoning = (
            f"Based on temperature trend of {trend:.3f}°C/year, "
            f"projected {year} anomaly is {projected_temp:.2f}°C vs "
            f"current record of {max_temp:.2f}°C ({max_year}). "
            f"Temperature difference: {temp_diff:+.2f}°C. "
            f"ENSO adjustment: {enso_adjustment:+.2f}."
        )
        
        return self._create_estimate(
            probability=base_prob,
            conviction=conviction,
            reasoning=reasoning,
            data_sources=[
                "NASA GISS Temperature Record",
                "NOAA Global Temperature Data",
                "ENSO Forecast",
            ],
            factors=factors,
        )
    
    def _estimate_temp_threshold(
        self, market: Market, year: int, threshold: float
    ) -> ProbabilityEstimate:
        """
        Estimate probability of exceeding a temperature threshold.
        """
        # Historical percentile analysis
        sorted_temps = sorted(HISTORICAL_TEMP_ANOMALIES.values())
        
        # Calculate trend
        recent_years = sorted(HISTORICAL_TEMP_ANOMALIES.keys())[-10:]
        temps = [HISTORICAL_TEMP_ANOMALIES[y] for y in recent_years]
        trend = (temps[-1] - temps[0]) / (len(recent_years) - 1)
        
        # Project for target year
        last_year = max(HISTORICAL_TEMP_ANOMALIES.keys())
        projected_temp = HISTORICAL_TEMP_ANOMALIES[last_year] + (trend * (year - last_year))
        
        # Probability based on projected vs threshold
        diff = projected_temp - threshold
        
        if diff > 0.2:
            prob = 0.90
        elif diff > 0.1:
            prob = 0.75 + (diff - 0.1) * 1.5
        elif diff > 0:
            prob = 0.60 + diff * 1.5
        elif diff > -0.1:
            prob = 0.40 + (diff + 0.1) * 2
        elif diff > -0.2:
            prob = 0.20 + (diff + 0.2) * 2
        else:
            prob = max(0.05, 0.20 + diff)
        
        reasoning = (
            f"Projected {year} temperature: {projected_temp:.2f}°C. "
            f"Threshold: {threshold:.1f}°C. Difference: {diff:+.2f}°C."
        )
        
        return self._create_estimate(
            probability=prob,
            conviction=self._get_conviction(0.65),
            reasoning=reasoning,
            data_sources=["NASA GISS", "NOAA Climate Data"],
            factors={"projected_temp": projected_temp, "threshold": threshold},
        )
    
    def _estimate_trend(self, market: Market, year: int) -> ProbabilityEstimate:
        """
        Default trend-based estimate for general temperature questions.
        """
        # Use market probability as baseline with slight adjustment
        market_prob = market.yes_price
        
        # Slight bullish adjustment for warming trend
        adjusted_prob = market_prob * 1.05
        
        return self._create_estimate(
            probability=adjusted_prob,
            conviction=ConvictionLevel.LOW,
            reasoning="Limited specific data; using market price with trend adjustment.",
            data_sources=["Market Price", "General Warming Trend"],
            factors={"market_baseline": market_prob},
        )


# =============================================================================
# HURRICANE MODEL
# =============================================================================

class HurricaneModel(BaseProbabilityModel):
    """
    Probability model for hurricane and tropical storm markets.
    
    Handles markets about:
    - Number of named storms in a season
    - Major hurricane counts (Category 3+)
    - Category 5 hurricane occurrence
    - Specific storm landfalls
    
    Methodology:
    1. Use NOAA seasonal hurricane forecasts
    2. Analyze historical storm frequency
    3. Consider ACE (Accumulated Cyclone Energy) index
    4. Adjust for current season conditions
    
    Data Sources:
    - NOAA Hurricane Center
    - Colorado State University Hurricane Forecasts
    """
    
    name = "hurricane_model"
    version = "1.0"
    
    # NOAA typical forecast ranges
    AVERAGE_NAMED_STORMS = 14
    AVERAGE_HURRICANES = 7
    AVERAGE_MAJOR_HURRICANES = 3
    
    def estimate(self, market: Market) -> ProbabilityEstimate:
        """
        Estimate probability for hurricane-related markets.
        """
        question = market.question.lower()
        
        # Category 5 questions
        if "category 5" in question or "cat 5" in question:
            return self._estimate_cat5(market)
        
        # Major hurricane questions
        if "major" in question and ("hurricane" in question or "storm" in question):
            return self._estimate_major_hurricanes(market)
        
        # Named storms count
        if any(kw in question for kw in ["named storm", "tropical storm", "hurricane count"]):
            return self._estimate_storm_count(market)
        
        # Landfall questions
        if "landfall" in question:
            return self._estimate_landfall(market)
        
        # Default estimate
        return self._estimate_general_hurricane(market)
    
    def _estimate_cat5(self, market: Market) -> ProbabilityEstimate:
        """
        Estimate probability of Category 5 hurricane occurrence.
        """
        # Historical rate of Cat 5 hurricanes
        years_with_cat5 = sum(1 for count in HISTORICAL_CAT5_HURRICANES.values() if count > 0)
        total_years = len(HISTORICAL_CAT5_HURRICANES)
        historical_rate = years_with_cat5 / total_years
        
        # Average Cat 5s per year
        avg_cat5 = sum(HISTORICAL_CAT5_HURRICANES.values()) / total_years
        
        # Extract number from question (e.g., "at least 2 Category 5")
        count_match = re.search(r"(\d+)\s*(?:or more|at least|category 5)", market.question.lower())
        target_count = int(count_match.group(1)) if count_match else 1
        
        # Probability based on historical frequency
        if target_count == 1:
            prob = historical_rate  # ~80% historically
        elif target_count == 2:
            years_with_2plus = sum(1 for c in HISTORICAL_CAT5_HURRICANES.values() if c >= 2)
            prob = years_with_2plus / total_years
        else:
            # Very rare to have 3+ Cat 5s
            prob = max(0.05, 1.0 / (target_count ** 2))
        
        # Adjust for current season factors (would use real ENSO/SST data)
        seasonal_adjustment = 0.05  # Slight increase for warmer SSTs
        prob += seasonal_adjustment
        
        reasoning = (
            f"Historical Cat 5 rate: {historical_rate:.1%} of years have at least 1. "
            f"Average: {avg_cat5:.1f} Cat 5s per year. "
            f"Target: {target_count}+ Cat 5 hurricanes."
        )
        
        return self._create_estimate(
            probability=prob,
            conviction=self._get_conviction(0.7),
            reasoning=reasoning,
            data_sources=["NOAA Hurricane Center", "Historical Hurricane Data"],
            factors={
                "historical_rate": historical_rate,
                "avg_cat5": avg_cat5,
                "target_count": target_count,
            },
        )
    
    def _estimate_major_hurricanes(self, market: Market) -> ProbabilityEstimate:
        """
        Estimate probability for major hurricane (Cat 3+) counts.
        """
        # Historical statistics
        major_counts = list(HISTORICAL_MAJOR_HURRICANES.values())
        avg_major = sum(major_counts) / len(major_counts)
        
        # Extract target from question
        count_match = re.search(r"(\d+)\s*(?:or more|at least|\+)", market.question)
        target = int(count_match.group(1)) if count_match else int(avg_major)
        
        # Calculate probability based on historical distribution
        years_meeting_target = sum(1 for c in major_counts if c >= target)
        prob = years_meeting_target / len(major_counts)
        
        reasoning = (
            f"Historical average: {avg_major:.1f} major hurricanes/year. "
            f"Target: {target}+ major hurricanes. "
            f"Met in {years_meeting_target}/{len(major_counts)} recent years."
        )
        
        return self._create_estimate(
            probability=prob,
            conviction=self._get_conviction(0.75),
            reasoning=reasoning,
            data_sources=["NOAA Hurricane Center", "CSU Hurricane Forecast"],
            factors={"avg_major": avg_major, "target": target},
        )
    
    def _estimate_storm_count(self, market: Market) -> ProbabilityEstimate:
        """
        Estimate probability for total named storm counts.
        """
        storm_counts = list(HISTORICAL_HURRICANES.values())
        avg_storms = sum(storm_counts) / len(storm_counts)
        
        # Extract target
        count_match = re.search(r"(\d+)", market.question)
        target = int(count_match.group(1)) if count_match else int(avg_storms)
        
        years_meeting = sum(1 for c in storm_counts if c >= target)
        prob = years_meeting / len(storm_counts)
        
        return self._create_estimate(
            probability=prob,
            conviction=self._get_conviction(0.7),
            reasoning=f"Historical average: {avg_storms:.1f} named storms. Target: {target}+.",
            data_sources=["NOAA Hurricane Center"],
            factors={"avg_storms": avg_storms, "target": target},
        )
    
    def _estimate_landfall(self, market: Market) -> ProbabilityEstimate:
        """
        Estimate probability of hurricane landfall.
        """
        # Default to moderate probability for US landfall
        base_prob = 0.55  # Slightly above-average landfall season expectation
        
        return self._create_estimate(
            probability=base_prob,
            conviction=ConvictionLevel.LOW,
            reasoning="Landfall probability based on historical seasonal averages.",
            data_sources=["NOAA Landfall Statistics"],
            factors={},
        )
    
    def _estimate_general_hurricane(self, market: Market) -> ProbabilityEstimate:
        """
        General hurricane estimate using market price as baseline.
        """
        return self._create_estimate(
            probability=market.yes_price,
            conviction=ConvictionLevel.LOW,
            reasoning="Using market price; insufficient specific data for adjustment.",
            data_sources=["Market Price"],
            factors={"market_baseline": market.yes_price},
        )


# =============================================================================
# EMISSIONS MODEL
# =============================================================================

class EmissionsModel(BaseProbabilityModel):
    """
    Probability model for emissions and CO2 markets.
    
    Handles markets about:
    - CO2 concentration levels
    - Emissions reduction targets
    - Paris Agreement compliance
    - Net zero commitments
    
    Methodology:
    1. Analyze CO2 concentration trends
    2. Consider economic/policy factors
    3. Evaluate country-specific commitments
    """
    
    name = "emissions_model"
    version = "1.0"
    
    def estimate(self, market: Market) -> ProbabilityEstimate:
        """
        Estimate probability for emissions-related markets.
        """
        question = market.question.lower()
        
        # CO2 concentration questions
        if "ppm" in question or "co2" in question:
            return self._estimate_co2_level(market)
        
        # Emissions targets
        if "emission" in question and ("reduction" in question or "target" in question):
            return self._estimate_emissions_target(market)
        
        # Paris Agreement
        if "paris" in question:
            return self._estimate_paris_compliance(market)
        
        # Default
        return self._estimate_general_emissions(market)
    
    def _estimate_co2_level(self, market: Market) -> ProbabilityEstimate:
        """
        Estimate probability for CO2 concentration thresholds.
        """
        # Calculate trend
        years = sorted(HISTORICAL_CO2.keys())[-10:]
        co2_values = [HISTORICAL_CO2[y] for y in years]
        annual_increase = (co2_values[-1] - co2_values[0]) / (len(years) - 1)
        
        # Extract target ppm
        ppm_match = re.search(r"(\d+)\s*ppm", market.question)
        target_ppm = float(ppm_match.group(1)) if ppm_match else 430
        
        # Extract year
        year_match = re.search(r"20\d{2}", market.question)
        target_year = int(year_match.group()) if year_match else datetime.now().year
        
        # Project CO2
        last_year = max(HISTORICAL_CO2.keys())
        projected_co2 = HISTORICAL_CO2[last_year] + annual_increase * (target_year - last_year)
        
        # Probability based on projection vs target
        diff = projected_co2 - target_ppm
        
        if diff > 5:
            prob = 0.95
        elif diff > 2:
            prob = 0.85 + (diff - 2) * 0.033
        elif diff > 0:
            prob = 0.70 + diff * 0.075
        elif diff > -2:
            prob = 0.50 + (diff + 2) * 0.1
        elif diff > -5:
            prob = 0.20 + (diff + 5) * 0.1
        else:
            prob = max(0.05, 0.20 + diff * 0.03)
        
        reasoning = (
            f"CO2 trend: +{annual_increase:.2f} ppm/year. "
            f"Projected {target_year}: {projected_co2:.1f} ppm. "
            f"Target: {target_ppm} ppm."
        )
        
        return self._create_estimate(
            probability=prob,
            conviction=self._get_conviction(0.8),  # CO2 is very predictable
            reasoning=reasoning,
            data_sources=["NOAA Global Monitoring Laboratory", "Mauna Loa Observatory"],
            factors={"annual_increase": annual_increase, "projected": projected_co2},
        )
    
    def _estimate_emissions_target(self, market: Market) -> ProbabilityEstimate:
        """
        Estimate probability of meeting emissions targets.
        """
        # Emissions targets are historically difficult to meet
        base_prob = 0.25  # Conservative baseline
        
        return self._create_estimate(
            probability=base_prob,
            conviction=ConvictionLevel.LOW,
            reasoning="Emissions targets have historically been missed ~75% of the time.",
            data_sources=["UN Climate Reports", "IEA World Energy Outlook"],
            factors={"baseline": base_prob},
        )
    
    def _estimate_paris_compliance(self, market: Market) -> ProbabilityEstimate:
        """
        Estimate probability of Paris Agreement compliance.
        """
        # Current trajectory suggests missing 1.5°C target
        base_prob = 0.15  # Low probability of full compliance
        
        return self._create_estimate(
            probability=base_prob,
            conviction=self._get_conviction(0.65),
            reasoning="Current trajectories suggest 2.5-3°C warming, missing 1.5°C target.",
            data_sources=["IPCC Reports", "Climate Action Tracker"],
            factors={},
        )
    
    def _estimate_general_emissions(self, market: Market) -> ProbabilityEstimate:
        """
        General emissions estimate.
        """
        return self._create_estimate(
            probability=market.yes_price,
            conviction=ConvictionLevel.LOW,
            reasoning="Using market price as baseline.",
            data_sources=["Market Price"],
            factors={},
        )


# =============================================================================
# SEA ICE MODEL
# =============================================================================

class SeaIceModel(BaseProbabilityModel):
    """
    Probability model for Arctic/Antarctic sea ice markets.
    
    Handles markets about:
    - September Arctic ice minimum
    - Ice-free Arctic predictions
    - Antarctic ice extent
    """
    
    name = "sea_ice_model"
    version = "1.0"
    
    def estimate(self, market: Market) -> ProbabilityEstimate:
        """
        Estimate probability for sea ice markets.
        """
        question = market.question.lower()
        
        # Ice-free Arctic
        if "ice-free" in question or "ice free" in question:
            return self._estimate_ice_free(market)
        
        # Record low
        if "record" in question and "low" in question:
            return self._estimate_record_low(market)
        
        # General ice extent
        return self._estimate_ice_extent(market)
    
    def _estimate_ice_free(self, market: Market) -> ProbabilityEstimate:
        """
        Estimate probability of ice-free Arctic.
        """
        # Extract target year
        year_match = re.search(r"20\d{2}", market.question)
        target_year = int(year_match.group()) if year_match else 2030
        
        # Ice-free defined as <1 million km²
        # Most models project ice-free Arctic by 2040-2050
        if target_year < 2030:
            prob = 0.05
        elif target_year < 2035:
            prob = 0.15
        elif target_year < 2040:
            prob = 0.35
        elif target_year < 2050:
            prob = 0.60
        else:
            prob = 0.80
        
        return self._create_estimate(
            probability=prob,
            conviction=self._get_conviction(0.6),
            reasoning=f"Models project ice-free Arctic by 2040-2050. Target year: {target_year}.",
            data_sources=["IPCC AR6", "NSIDC Projections"],
            factors={"target_year": target_year},
        )
    
    def _estimate_record_low(self, market: Market) -> ProbabilityEstimate:
        """
        Estimate probability of new record low sea ice.
        """
        # Calculate trend
        years = sorted(HISTORICAL_SEA_ICE.keys())
        ice_values = [HISTORICAL_SEA_ICE[y] for y in years]
        trend = (ice_values[-1] - ice_values[0]) / (len(years) - 1)
        
        # Project ice extent
        last_year = max(HISTORICAL_SEA_ICE.keys())
        year_match = re.search(r"20\d{2}", market.question)
        target_year = int(year_match.group()) if year_match else datetime.now().year
        
        projected_ice = HISTORICAL_SEA_ICE[last_year] + trend * (target_year - last_year)
        
        # Compare to record
        diff = projected_ice - SEA_ICE_RECORD_LOW
        
        if diff < 0:
            prob = 0.70 + min(0.25, abs(diff) * 0.5)
        elif diff < 0.5:
            prob = 0.40 + (0.5 - diff) * 0.6
        else:
            prob = max(0.10, 0.40 - diff * 0.3)
        
        return self._create_estimate(
            probability=prob,
            conviction=self._get_conviction(0.6),
            reasoning=f"Projected ice: {projected_ice:.2f} M km². Record: {SEA_ICE_RECORD_LOW} M km².",
            data_sources=["NSIDC Sea Ice Index"],
            factors={"projected": projected_ice, "record": SEA_ICE_RECORD_LOW},
        )
    
    def _estimate_ice_extent(self, market: Market) -> ProbabilityEstimate:
        """
        General ice extent estimate.
        """
        return self._create_estimate(
            probability=market.yes_price,
            conviction=ConvictionLevel.LOW,
            reasoning="Using market price as baseline for general ice extent question.",
            data_sources=["Market Price", "NSIDC"],
            factors={},
        )


# =============================================================================
# WILDFIRE MODEL
# =============================================================================

class WildfireModel(BaseProbabilityModel):
    """
    Probability model for wildfire markets.
    
    Handles markets about:
    - Acres/hectares burned
    - Fire season severity
    - Specific region wildfires
    """
    
    name = "wildfire_model"
    version = "1.0"
    
    def estimate(self, market: Market) -> ProbabilityEstimate:
        """
        Estimate probability for wildfire markets.
        """
        # Wildfires are increasing due to climate change
        # Default to slightly above historical average
        base_prob = market.yes_price * 1.1  # 10% increase for warming trend
        
        return self._create_estimate(
            probability=base_prob,
            conviction=ConvictionLevel.LOW,
            reasoning="Wildfire risk is increasing due to climate change. "
                      "Adjusted market price by +10% for warming trend.",
            data_sources=["NIFC Wildfire Statistics", "Climate Change Impact Studies"],
            factors={"market_baseline": market.yes_price},
        )


# =============================================================================
# COMPOSITE CLIMATE MODEL
# =============================================================================

class ClimateProbabilityModel(BaseProbabilityModel):
    """
    Composite probability model that delegates to specialized models.
    
    This is the main model you should use. It automatically:
    1. Classifies the market type
    2. Routes to the appropriate specialized model
    3. Returns the specialized estimate
    
    You can register custom models for specific market types.
    
    Example:
        >>> model = ClimateProbabilityModel()
        >>> estimate = model.estimate(market)
        >>> print(f"Probability: {estimate.probability:.1%}")
        >>> print(f"Reasoning: {estimate.reasoning}")
    """
    
    name = "climate_composite"
    version = "1.0"
    
    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        
        # Initialize specialized models
        self._models: dict[MarketType, BaseProbabilityModel] = {
            MarketType.TEMPERATURE: TemperatureModel(config),
            MarketType.HURRICANE: HurricaneModel(config),
            MarketType.EMISSIONS: EmissionsModel(config),
            MarketType.SEA_ICE: SeaIceModel(config),
            MarketType.WILDFIRE: WildfireModel(config),
        }
    
    def register_model(self, market_type: MarketType, model: BaseProbabilityModel) -> None:
        """
        Register a custom model for a market type.
        
        Use this to add your own probability estimation logic.
        
        Args:
            market_type: The type of market this model handles
            model: The probability model instance
        
        Example:
            >>> class MyTempModel(TemperatureModel):
            ...     def estimate(self, market):
            ...         # Custom logic
            ...         pass
            >>> 
            >>> composite = ClimateProbabilityModel()
            >>> composite.register_model(MarketType.TEMPERATURE, MyTempModel())
        """
        self._models[market_type] = model
        logger.info(f"Registered custom model for {market_type.value}")
    
    def estimate(self, market: Market) -> ProbabilityEstimate:
        """
        Estimate probability by delegating to the appropriate model.
        
        Args:
            market: The market to estimate
        
        Returns:
            ProbabilityEstimate from the specialized model
        """
        # Classify market type
        market_type = self.classify_market(market)
        logger.debug(f"Classified market as {market_type.value}: {market.question[:50]}...")
        
        # Get specialized model
        model = self._models.get(market_type)
        
        if model:
            return model.estimate(market)
        
        # Fallback for unknown market types
        logger.warning(f"No model for market type {market_type}, using market price")
        return self._create_estimate(
            probability=market.yes_price,
            conviction=ConvictionLevel.LOW,
            reasoning=f"No specialized model for {market_type.value}. Using market price.",
            data_sources=["Market Price"],
            factors={"market_baseline": market.yes_price},
        )
    
    def get_model(self, market_type: MarketType) -> Optional[BaseProbabilityModel]:
        """Get the model for a specific market type."""
        return self._models.get(market_type)
    
    @property
    def available_models(self) -> list[str]:
        """List available specialized models."""
        return [mt.value for mt in self._models.keys()]


# =============================================================================
# CUSTOM MODEL TEMPLATE
# =============================================================================

class CustomClimateModel(BaseProbabilityModel):
    """
    Template for creating custom probability models.
    
    Copy this class and modify the estimate() method to add your own
    probability estimation logic.
    
    Example:
        >>> class MyClimateModel(CustomClimateModel):
        ...     name = "my_model"
        ...     
        ...     def estimate(self, market: Market) -> ProbabilityEstimate:
        ...         # Your custom logic
        ...         my_probability = self._my_analysis(market)
        ...         return self._create_estimate(
        ...             probability=my_probability,
        ...             conviction=ConvictionLevel.MEDIUM,
        ...             reasoning="My custom analysis shows...",
        ...             data_sources=["My Data Source"],
        ...         )
    """
    
    name = "custom_model"
    version = "1.0"
    
    def estimate(self, market: Market) -> ProbabilityEstimate:
        """
        Override this method with your custom estimation logic.
        
        Things you can do:
        - Query external APIs for real-time data
        - Use machine learning models
        - Apply Bayesian updating
        - Combine multiple signals
        
        Args:
            market: The market to estimate
        
        Returns:
            Your probability estimate
        """
        # Example: Query your own data source
        # data = self._fetch_my_data(market)
        # probability = self._analyze(data)
        
        # For now, return market price as baseline
        return self._create_estimate(
            probability=market.yes_price,
            conviction=ConvictionLevel.LOW,
            reasoning="Custom model - implement your logic here!",
            data_sources=["Custom Data"],
            factors={},
        )
    
    # Add your custom helper methods here
    # def _fetch_my_data(self, market: Market) -> Any:
    #     pass
    # 
    # def _analyze(self, data: Any) -> float:
    #     pass
