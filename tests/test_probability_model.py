"""
Tests for probability estimation models.
"""

import pytest
from polymonitor.models import Market, MarketType, ConvictionLevel
from polymonitor.probability_model import (
    BaseProbabilityModel,
    ClimateProbabilityModel,
    TemperatureModel,
    HurricaneModel,
    EmissionsModel,
    SeaIceModel,
    WildfireModel,
)


class TestMarketClassification:
    """Tests for market type classification."""
    
    @pytest.fixture
    def model(self):
        return ClimateProbabilityModel()
    
    def test_classify_temperature_market(self, model):
        """Test classification of temperature markets."""
        market = Market(question="Will 2026 be the hottest year on record?")
        assert model.classify_market(market) == MarketType.TEMPERATURE
        
        market = Market(question="Will global temperatures exceed 1.5°C above pre-industrial levels?")
        assert model.classify_market(market) == MarketType.TEMPERATURE
        
        market = Market(question="Will there be a record high temperature in July?")
        assert model.classify_market(market) == MarketType.TEMPERATURE
    
    def test_classify_hurricane_market(self, model):
        """Test classification of hurricane markets."""
        market = Market(question="Will there be a Category 5 hurricane in 2026?")
        assert model.classify_market(market) == MarketType.HURRICANE
        
        market = Market(question="Will 20+ named storms form in the Atlantic?")
        assert model.classify_market(market) == MarketType.HURRICANE
        
        market = Market(question="Will a major hurricane make landfall in Florida?")
        assert model.classify_market(market) == MarketType.HURRICANE
    
    def test_classify_emissions_market(self, model):
        """Test classification of emissions markets."""
        market = Market(question="Will CO2 levels exceed 430 ppm in 2026?")
        assert model.classify_market(market) == MarketType.EMISSIONS
        
        market = Market(question="Will the US meet its emissions reduction targets?")
        assert model.classify_market(market) == MarketType.EMISSIONS
        
        market = Market(question="Will carbon emissions fall below 2019 levels?")
        assert model.classify_market(market) == MarketType.EMISSIONS
    
    def test_classify_sea_ice_market(self, model):
        """Test classification of sea ice markets."""
        market = Market(question="Will Arctic sea ice hit a record low in September?")
        assert model.classify_market(market) == MarketType.SEA_ICE
        
        market = Market(question="Will the Arctic be ice-free by 2035?")
        assert model.classify_market(market) == MarketType.SEA_ICE
    
    def test_classify_wildfire_market(self, model):
        """Test classification of wildfire markets."""
        market = Market(question="Will California wildfires burn over 500,000 acres?")
        assert model.classify_market(market) == MarketType.WILDFIRE
    
    def test_classify_general_climate(self, model):
        """Test classification of general climate markets."""
        market = Market(question="Will El Niño conditions persist through summer?")
        assert model.classify_market(market) == MarketType.GENERAL_CLIMATE
        
        market = Market(question="Will NOAA declare a drought emergency?")
        assert model.classify_market(market) == MarketType.GENERAL_CLIMATE
    
    def test_classify_unknown(self, model):
        """Test classification of non-climate markets."""
        market = Market(question="Will Bitcoin reach $100,000?")
        assert model.classify_market(market) == MarketType.UNKNOWN


class TestTemperatureModel:
    """Tests for temperature probability model."""
    
    @pytest.fixture
    def model(self):
        return TemperatureModel()
    
    def test_estimate_hottest_year(self, model):
        """Test hottest year estimation."""
        market = Market(
            question="Will 2026 be the hottest year on record?",
            outcomePrices=[0.5, 0.5],
        )
        
        estimate = model.estimate(market)
        
        assert 0 < estimate.probability < 1
        assert estimate.model_name == "temperature_model"
        assert "NASA" in estimate.data_sources or "NOAA" in estimate.data_sources
        assert len(estimate.reasoning) > 0
    
    def test_estimate_temperature_threshold(self, model):
        """Test temperature threshold estimation."""
        market = Market(
            question="Will global temperature anomaly exceed 1.5°C in 2026?",
            outcomePrices=[0.6, 0.4],
        )
        
        estimate = model.estimate(market)
        
        assert 0 < estimate.probability < 1
        assert "1.5" in estimate.reasoning or "threshold" in estimate.reasoning.lower()
    
    def test_estimate_includes_factors(self, model):
        """Test that estimate includes factor breakdown."""
        market = Market(
            question="Will 2026 be the hottest year on record?",
            outcomePrices=[0.5, 0.5],
        )
        
        estimate = model.estimate(market)
        
        assert len(estimate.factors) > 0
        assert "trend" in estimate.factors or "projected_temp" in estimate.factors


class TestHurricaneModel:
    """Tests for hurricane probability model."""
    
    @pytest.fixture
    def model(self):
        return HurricaneModel()
    
    def test_estimate_cat5(self, model):
        """Test Category 5 hurricane estimation."""
        market = Market(
            question="Will there be at least 1 Category 5 hurricane in 2026?",
            outcomePrices=[0.5, 0.5],
        )
        
        estimate = model.estimate(market)
        
        assert 0 < estimate.probability < 1
        assert "NOAA" in estimate.data_sources or "Hurricane" in str(estimate.data_sources)
    
    def test_estimate_major_hurricanes(self, model):
        """Test major hurricane count estimation."""
        market = Market(
            question="Will there be 5 or more major hurricanes in 2026?",
            outcomePrices=[0.5, 0.5],
        )
        
        estimate = model.estimate(market)
        
        assert 0 < estimate.probability < 1
    
    def test_estimate_storm_count(self, model):
        """Test named storm count estimation."""
        market = Market(
            question="Will there be 20+ named storms in the Atlantic?",
            outcomePrices=[0.5, 0.5],
        )
        
        estimate = model.estimate(market)
        
        assert 0 < estimate.probability < 1


class TestEmissionsModel:
    """Tests for emissions probability model."""
    
    @pytest.fixture
    def model(self):
        return EmissionsModel()
    
    def test_estimate_co2_level(self, model):
        """Test CO2 level estimation."""
        market = Market(
            question="Will CO2 levels exceed 430 ppm by end of 2026?",
            outcomePrices=[0.5, 0.5],
        )
        
        estimate = model.estimate(market)
        
        assert 0 < estimate.probability < 1
        # CO2 is highly predictable so should have higher conviction
        assert estimate.conviction in [ConvictionLevel.MEDIUM, ConvictionLevel.HIGH, ConvictionLevel.VERY_HIGH]
    
    def test_estimate_emissions_target(self, model):
        """Test emissions target estimation."""
        market = Market(
            question="Will global emissions decrease 10% by 2026?",
            outcomePrices=[0.5, 0.5],
        )
        
        estimate = model.estimate(market)
        
        # Emissions targets are historically missed
        assert estimate.probability < 0.5


class TestClimateProbabilityModel:
    """Tests for composite climate model."""
    
    @pytest.fixture
    def model(self):
        return ClimateProbabilityModel()
    
    def test_delegates_to_temperature_model(self, model):
        """Test delegation to temperature model."""
        market = Market(
            question="Will 2026 be the hottest year?",
            outcomePrices=[0.5, 0.5],
        )
        
        estimate = model.estimate(market)
        
        # Should delegate to temperature model
        assert estimate.model_name == "temperature_model"
    
    def test_delegates_to_hurricane_model(self, model):
        """Test delegation to hurricane model."""
        market = Market(
            question="Will there be a Category 5 hurricane?",
            outcomePrices=[0.5, 0.5],
        )
        
        estimate = model.estimate(market)
        
        assert estimate.model_name == "hurricane_model"
    
    def test_register_custom_model(self, model):
        """Test registering a custom model."""
        class CustomModel(TemperatureModel):
            name = "custom_temp"
            
            def estimate(self, market):
                estimate = super().estimate(market)
                estimate.model_name = "custom_temp"
                return estimate
        
        model.register_model(MarketType.TEMPERATURE, CustomModel())
        
        market = Market(
            question="Will 2026 be the hottest year?",
            outcomePrices=[0.5, 0.5],
        )
        
        estimate = model.estimate(market)
        assert estimate.model_name == "custom_temp"
    
    def test_fallback_for_unknown(self, model):
        """Test fallback behavior for unknown markets."""
        market = Market(
            question="Will aliens visit Earth in 2026?",
            outcomePrices=[0.01, 0.99],
        )
        
        estimate = model.estimate(market)
        
        # Should fall back to market price
        assert estimate.conviction == ConvictionLevel.LOW
    
    def test_available_models(self, model):
        """Test listing available models."""
        available = model.available_models
        
        assert "temperature" in available
        assert "hurricane" in available
        assert "emissions" in available


class TestConvictionLevels:
    """Tests for conviction level assignment."""
    
    def test_conviction_thresholds(self):
        model = TemperatureModel()
        
        # Test low conviction
        assert model._get_conviction(0.5) == ConvictionLevel.LOW
        
        # Test medium conviction
        assert model._get_conviction(0.65) == ConvictionLevel.MEDIUM
        
        # Test high conviction
        assert model._get_conviction(0.75) == ConvictionLevel.HIGH
        
        # Test very high conviction
        assert model._get_conviction(0.9) == ConvictionLevel.VERY_HIGH
