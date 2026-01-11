"""
Tests for trading functionality.
"""

import pytest
from polymonitor.models import Market, Opportunity, ProbabilityEstimate, ConvictionLevel
from polymonitor.trading import (
    KellyCriterion,
    TradingEngine,
    SimulatedTradingEngine,
    Order,
    OrderSide,
    OrderType,
    Position,
)


class TestKellyCriterion:
    """Tests for Kelly Criterion calculator."""
    
    @pytest.fixture
    def kelly(self):
        return KellyCriterion(fraction=0.25, max_bet_fraction=0.05)
    
    def test_positive_edge_yes(self, kelly):
        """Test Kelly calculation when YES is underpriced."""
        bet = kelly.calculate(
            bankroll=1000,
            true_prob=0.7,  # We think 70%
            market_price=0.5,  # Market says 50%
            side="yes",
        )
        
        assert bet > 0
        assert bet <= 50  # Max 5% of bankroll
    
    def test_positive_edge_no(self, kelly):
        """Test Kelly calculation when NO is underpriced."""
        bet = kelly.calculate(
            bankroll=1000,
            true_prob=0.3,  # We think YES is 30%
            market_price=0.5,  # Market says YES is 50%
            side="no",
        )
        
        assert bet > 0
        assert bet <= 50
    
    def test_no_edge(self, kelly):
        """Test Kelly returns 0 when no edge."""
        bet = kelly.calculate(
            bankroll=1000,
            true_prob=0.5,
            market_price=0.5,
            side="yes",
        )
        
        assert bet == 0
    
    def test_negative_edge_returns_zero(self, kelly):
        """Test Kelly returns 0 for negative edge."""
        bet = kelly.calculate(
            bankroll=1000,
            true_prob=0.4,  # We think 40%
            market_price=0.6,  # Market says 60%
            side="yes",  # Betting YES when underpriced
        )
        
        # Should not bet when we think probability is lower than market
        assert bet == 0
    
    def test_max_bet_fraction_enforced(self, kelly):
        """Test that max bet fraction is enforced."""
        # Large edge that would normally suggest bigger bet
        bet = kelly.calculate(
            bankroll=1000,
            true_prob=0.9,  # Very confident
            market_price=0.5,
            side="yes",
        )
        
        assert bet <= 50  # Max 5% enforced
    
    def test_edge_calculation(self, kelly):
        """Test edge calculation."""
        edge_yes = kelly.calculate_edge(
            true_prob=0.7,
            market_price=0.5,
            side="yes",
        )
        assert edge_yes == 0.2  # 70% - 50%
        
        edge_no = kelly.calculate_edge(
            true_prob=0.3,  # YES is 30%, NO is 70%
            market_price=0.5,  # YES is 50%, NO is 50%
            side="no",
        )
        assert edge_no == 0.2  # 70% - 50%
    
    def test_ev_calculation(self, kelly):
        """Test expected value calculation."""
        ev = kelly.calculate_ev(
            true_prob=0.6,
            market_price=0.4,
            side="yes",
            bet_size=100,
        )
        
        # EV should be positive for favorable bet
        assert ev > 0


class TestOrder:
    """Tests for Order model."""
    
    def test_order_creation(self):
        """Test creating an order."""
        order = Order(
            market_id="market123",
            token_id="token456",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=100,
            price=0.55,
        )
        
        assert order.market_id == "market123"
        assert order.side == OrderSide.BUY
        assert order.size == 100
        assert order.price == 0.55
        assert order.status == "pending"
    
    def test_order_to_dict(self):
        """Test order serialization."""
        order = Order(
            market_id="market123",
            token_id="token456",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=100,
            price=0.55,
        )
        
        d = order.to_dict()
        
        assert d["tokenID"] == "token456"
        assert d["side"] == "BUY"
        assert d["size"] == 100


class TestPosition:
    """Tests for Position model."""
    
    def test_position_creation(self):
        """Test creating a position."""
        from datetime import datetime
        
        pos = Position(
            market_id="market123",
            token_id="token456",
            market_question="Will it rain?",
            side="yes",
            size=100,
            avg_price=0.5,
            entry_time=datetime.now(),
            current_price=0.6,
        )
        
        assert pos.value == 60  # 100 * 0.6
        assert pos.cost_basis == 50  # 100 * 0.5


class TestTradingEngine:
    """Tests for trading engine."""
    
    @pytest.fixture
    def engine(self):
        return TradingEngine()
    
    @pytest.fixture
    def engine_enabled(self):
        """Engine with trading flag enabled (but no credentials)."""
        return TradingEngine(enable_trading=True)
    
    def test_trading_disabled_by_default(self, engine):
        """Test that trading is disabled by default."""
        assert engine.is_enabled == False
        assert engine.is_dry_run == True
    
    def test_trading_enabled_via_flag(self, engine_enabled):
        """Test trading can be enabled via flag (but needs credentials)."""
        # Still disabled because no credentials
        assert engine_enabled.is_enabled == False
    
    def test_place_order_when_disabled_dry_run(self, engine):
        """Test order placement returns dry run when disabled."""
        order = Order(
            market_id="test",
            token_id="test",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=100,
            price=0.5,
        )
        
        result = engine.place_order(order)
        
        # Dry run should succeed with a message
        assert result.success == True
        assert "DRY RUN" in result.message
        assert order.status == "dry_run"
    
    def test_kelly_calculator_available(self, engine):
        """Test Kelly calculator is available."""
        assert engine.kelly is not None
        assert isinstance(engine.kelly, KellyCriterion)


class TestSimulatedTradingEngine:
    """Tests for simulated trading."""
    
    @pytest.fixture
    def engine(self):
        return SimulatedTradingEngine(initial_bankroll=10000)
    
    def test_simulation_always_enabled(self, engine):
        """Test simulation is always enabled."""
        assert engine.is_enabled == True
    
    def test_initial_bankroll(self, engine):
        """Test initial bankroll is set."""
        assert engine.bankroll == 10000
        assert engine.pnl == 0
    
    def test_place_simulated_order(self, engine):
        """Test placing a simulated order."""
        order = Order(
            market_id="test",
            token_id="test",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=100,
            price=0.5,
        )
        
        result = engine.place_order(order)
        
        assert result.success == True
        assert "SIMULATED" in result.message
        assert engine.bankroll == 9900  # 10000 - 100
    
    def test_risk_limits_in_simulation(self, engine):
        """Test risk limits are enforced in simulation."""
        # Try to place order exceeding per-market limit
        order = Order(
            market_id="test",
            token_id="test",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=500,  # Exceeds default 100 limit
            price=0.5,
        )
        
        result = engine.place_order(order)
        
        assert result.success == False
        assert "Risk limit" in result.message
    
    def test_simulation_report(self, engine):
        """Test simulation report generation."""
        # Place some orders
        for i in range(3):
            order = Order(
                market_id=f"test{i}",
                token_id=f"token{i}",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                size=50,
                price=0.5,
            )
            engine.place_order(order)
        
        report = engine.get_simulation_report()
        
        assert report["initial_bankroll"] == 10000
        assert report["current_bankroll"] == 9850  # 10000 - 150
        assert report["total_trades"] == 3
        assert report["pnl"] == -150
    
    def test_resolve_winning_position(self, engine):
        """Test resolving a winning position."""
        from datetime import datetime
        
        # Place order
        order = Order(
            market_id="win_market",
            token_id="token",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=100,
            price=0.5,
        )
        engine.place_order(order)
        
        # Manually add position (normally done by order execution)
        engine._positions["win_market"] = Position(
            market_id="win_market",
            token_id="token",
            market_question="Test",
            side="yes",
            size=200,  # 100 / 0.5 = 200 tokens
            avg_price=0.5,
            entry_time=datetime.now(),
        )
        
        # Resolve as win
        pnl = engine.resolve_position("win_market", "yes")
        
        assert pnl > 0  # Should be profitable


class TestOpportunityExecution:
    """Tests for executing opportunities."""
    
    def test_opportunity_with_simulated_engine(self):
        """Test executing an opportunity in simulation."""
        engine = SimulatedTradingEngine(initial_bankroll=1000)
        
        market = Market(
            question="Test market",
            conditionId="cond123",
            outcomePrices=[0.5, 0.5],
            clobTokenIds=["token_yes", "token_no"],
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
        
        result = engine.execute_opportunity(opp, bankroll=1000)
        
        assert result.success == True
        assert result.order.size > 0
