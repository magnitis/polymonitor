"""
Trading Engine for Polymarket
==============================

This module provides trading functionality for Polymarket using the
py-clob-client library.

⚠️  WARNING: TRADING IS DISABLED BY DEFAULT
    To enable trading, you must:
    1. Set trading.enabled = true in config.yaml OR
    2. Pass --enable-trading flag to CLI commands OR
    3. Set enable_trading=True when creating TradingEngine

Features:
- Kelly Criterion bet sizing with fractional Kelly
- Single and batch order placement
- Slippage protection
- Position tracking
- Risk management limits
- Auto-trading mode

Dependencies:
- py-clob-client: pip install py-clob-client
- Polygon wallet with USDC
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from polymonitor.config import Config, EnvironmentSettings, get_config, get_env
from polymonitor.models import Market, Opportunity

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    """Order side (buy/sell)."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""
    LIMIT = "LIMIT"
    MARKET = "MARKET"


@dataclass
class Position:
    """
    Represents an open position in a market.
    
    Attributes:
        market_id: The market condition ID
        token_id: The CLOB token ID
        side: Whether we hold YES or NO tokens
        size: Number of tokens held
        avg_price: Average entry price
        current_price: Current market price
        pnl: Unrealized profit/loss
    """
    market_id: str
    token_id: str
    market_question: str
    side: str  # "yes" or "no"
    size: float
    avg_price: float
    entry_time: datetime
    current_price: float = 0.0
    pnl: float = 0.0
    
    @property
    def value(self) -> float:
        """Current position value."""
        return self.size * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Total cost to acquire position."""
        return self.size * self.avg_price


@dataclass
class Order:
    """
    Represents a trading order.
    
    Attributes:
        market_id: Market condition ID
        token_id: CLOB token ID for the outcome
        side: BUY or SELL
        order_type: LIMIT or MARKET
        size: Order size in USD
        price: Limit price (for LIMIT orders)
        slippage: Maximum allowed slippage
    """
    market_id: str
    token_id: str
    side: OrderSide
    order_type: OrderType
    size: float
    price: Optional[float] = None
    slippage: float = 0.02
    
    # Filled order info
    order_id: Optional[str] = None
    filled_size: float = 0.0
    filled_price: float = 0.0
    status: str = "pending"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API submission."""
        return {
            "tokenID": self.token_id,
            "side": self.side.value,
            "size": self.size,
            "price": self.price,
        }


@dataclass
class TradeResult:
    """Result of a trade execution."""
    success: bool
    order: Order
    message: str
    transaction_hash: Optional[str] = None
    error: Optional[str] = None


class KellyCriterion:
    """
    Kelly Criterion bet sizing calculator.
    
    The Kelly Criterion determines the optimal bet size to maximize
    long-term growth rate while managing risk.
    
    Formula: f* = (bp - q) / b
    
    Where:
    - f* = fraction of bankroll to bet
    - b = odds received (decimal - 1)
    - p = probability of winning
    - q = probability of losing (1 - p)
    
    In practice, we use "fractional Kelly" (typically 25-50% of full Kelly)
    to reduce volatility and account for estimation errors.
    
    Example:
        >>> kelly = KellyCriterion(fraction=0.25)
        >>> bet_size = kelly.calculate(
        ...     bankroll=1000,
        ...     true_prob=0.6,
        ...     market_price=0.4,
        ...     side="yes"
        ... )
        >>> print(f"Recommended bet: ${bet_size:.2f}")
    """
    
    def __init__(self, fraction: float = 0.25, max_bet_fraction: float = 0.05):
        """
        Initialize Kelly Calculator.
        
        Args:
            fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
            max_bet_fraction: Maximum bet as fraction of bankroll
        """
        self.fraction = fraction
        self.max_bet_fraction = max_bet_fraction
    
    def calculate(
        self,
        bankroll: float,
        true_prob: float,
        market_price: float,
        side: str,
    ) -> float:
        """
        Calculate recommended bet size.
        
        Args:
            bankroll: Total bankroll in USD
            true_prob: Our estimated probability of YES outcome
            market_price: Current market price for YES
            side: Which side to bet ("yes" or "no")
        
        Returns:
            Recommended bet size in USD
        """
        if side == "yes":
            p = true_prob
            b = (1 / market_price) - 1 if market_price > 0 else 0
        else:
            p = 1 - true_prob
            b = (1 / (1 - market_price)) - 1 if market_price < 1 else 0
        
        q = 1 - p
        
        if b <= 0:
            return 0.0
        
        # Full Kelly fraction
        kelly = (b * p - q) / b
        kelly = max(0, kelly)
        
        # Apply fractional Kelly
        bet_fraction = kelly * self.fraction
        
        # Apply maximum bet limit
        bet_fraction = min(bet_fraction, self.max_bet_fraction)
        
        # Calculate actual bet size
        bet_size = bankroll * bet_fraction
        
        return round(bet_size, 2)
    
    def calculate_edge(self, true_prob: float, market_price: float, side: str) -> float:
        """
        Calculate expected edge.
        
        Args:
            true_prob: Our probability estimate
            market_price: Market price
            side: Bet side
        
        Returns:
            Expected edge as a fraction
        """
        if side == "yes":
            return true_prob - market_price
        else:
            return (1 - true_prob) - (1 - market_price)
    
    def calculate_ev(
        self,
        true_prob: float,
        market_price: float,
        side: str,
        bet_size: float = 1.0,
    ) -> float:
        """
        Calculate expected value of a bet.
        
        Args:
            true_prob: Our probability estimate
            market_price: Market price
            side: Bet side
            bet_size: Size of bet in USD
        
        Returns:
            Expected value in USD
        """
        if side == "yes":
            p = true_prob
            payout = bet_size / market_price if market_price > 0 else 0
        else:
            p = 1 - true_prob
            payout = bet_size / (1 - market_price) if market_price < 1 else 0
        
        ev = (p * (payout - bet_size)) - ((1 - p) * bet_size)
        return ev


class TradingEngine:
    """
    Trading engine for executing orders on Polymarket.
    
    ⚠️  TRADING IS DISABLED BY DEFAULT
    
    This class provides methods for:
    - Order placement (single and batch)
    - Position tracking
    - Risk management
    - Auto-trading on opportunities
    
    To enable trading:
    1. Set trading.enabled = true in config OR
    2. Pass enable_trading=True to constructor OR
    3. Use --enable-trading CLI flag
    
    Example:
        >>> # Explicitly enable trading
        >>> engine = TradingEngine(enable_trading=True)
        >>> if engine.is_enabled:
        ...     result = engine.place_order(order)
        ...     print(f"Order placed: {result.success}")
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        env: Optional[EnvironmentSettings] = None,
        enable_trading: bool = False,
    ):
        """
        Initialize the trading engine.
        
        Args:
            config: Application configuration
            env: Environment settings with credentials
            enable_trading: Override flag to enable trading (default False)
        """
        self.config = config or get_config()
        self.env = env or get_env()
        self._enable_trading_override = enable_trading
        
        # Trading client (initialized lazily)
        self._client = None
        
        # Position tracking
        self._positions: dict[str, Position] = {}
        
        # Order history
        self._order_history: list[Order] = []
        
        # Kelly calculator
        self.kelly = KellyCriterion(
            fraction=self.config.trading.kelly.fraction,
            max_bet_fraction=self.config.trading.kelly.max_bet_fraction,
        )
        
        # Risk limits
        self._total_exposure: float = 0.0
        
        # Log trading status
        if self.is_enabled:
            logger.warning("⚠️  TRADING IS ENABLED - Real money will be used!")
        else:
            logger.info("Trading is disabled (use --enable-trading to enable)")
    
    @property
    def is_enabled(self) -> bool:
        """
        Check if trading is enabled.
        
        Trading requires:
        1. Either config.trading.enabled=True OR enable_trading=True passed to constructor
        2. Valid credentials (private key and funder address)
        """
        # Check if trading is enabled via config or override
        trading_enabled = self.config.trading.enabled or self._enable_trading_override
        
        if not trading_enabled:
            return False
        
        # Check for required credentials
        if not self.env.private_key or not self.env.funder_address:
            logger.warning(
                "Trading enabled but credentials not set. "
                "Set POLYMARKET_PRIVATE_KEY and POLYMARKET_FUNDER_ADDRESS in .env"
            )
            return False
        
        return True
    
    @property
    def is_dry_run(self) -> bool:
        """Check if we're in dry-run mode (trading disabled)."""
        return not self.is_enabled
    
    @property
    def client(self):
        """
        Get or create the CLOB client.
        
        Returns None if trading is disabled or credentials are missing.
        """
        if self._client is None and self.is_enabled:
            try:
                from py_clob_client.client import ClobClient
                
                self._client = ClobClient(
                    host=self.config.trading.clob_host,
                    key=self.env.private_key.get_secret_value(),
                    chain_id=self.config.trading.chain_id,
                    signature_type=1,  # 1 for Magic/email wallets
                    funder=self.env.funder_address,
                )
                
                # Create or derive API credentials
                api_creds = self._client.create_or_derive_api_creds()
                self._client.set_api_creds(api_creds)
                
                logger.info("CLOB client initialized successfully")
                
            except ImportError:
                logger.error(
                    "py-clob-client not installed. "
                    "Install with: pip install py-clob-client"
                )
                return None
            except Exception as e:
                logger.error(f"Failed to initialize CLOB client: {e}")
                return None
        
        return self._client
    
    def get_balance(self) -> float:
        """
        Get available USDC balance.
        
        Returns:
            Balance in USD, or 0 if trading disabled
        """
        if not self.is_enabled or self.client is None:
            return 0.0
        
        try:
            balance = self.client.get_balance()
            return float(balance)
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0
    
    def place_order(self, order: Order) -> TradeResult:
        """
        Place a single order.
        
        Args:
            order: The order to place
        
        Returns:
            TradeResult with success status and details
        """
        # Check if trading is enabled
        if not self.is_enabled:
            # Dry run mode - simulate the order
            logger.info(
                f"[DRY RUN] Would place order: {order.side.value} "
                f"${order.size:.2f} at {order.price}"
            )
            order.status = "dry_run"
            return TradeResult(
                success=True,
                order=order,
                message="[DRY RUN] Order simulated (trading disabled)",
            )
        
        # Risk checks
        risk_check = self._check_risk_limits(order)
        if not risk_check["allowed"]:
            return TradeResult(
                success=False,
                order=order,
                message="Risk limit exceeded",
                error=risk_check["reason"],
            )
        
        # Ensure client is initialized
        if self.client is None:
            return TradeResult(
                success=False,
                order=order,
                message="Trading client not available",
                error="Failed to initialize CLOB client",
            )
        
        try:
            from py_clob_client.order_builder.builder import OrderBuilder
            
            builder = OrderBuilder(self.client)
            
            if order.order_type == OrderType.LIMIT:
                clob_order = builder.build_limit_order(
                    token_id=order.token_id,
                    side=order.side.value,
                    size=order.size,
                    price=order.price,
                )
            else:
                clob_order = builder.build_market_order(
                    token_id=order.token_id,
                    side=order.side.value,
                    size=order.size,
                )
            
            # Place order
            result = self.client.place_order(clob_order)
            
            order.order_id = result.get("orderID")
            order.status = "placed"
            
            self._order_history.append(order)
            self._update_exposure(order)
            
            logger.info(f"Order placed: {order.order_id}")
            
            return TradeResult(
                success=True,
                order=order,
                message=f"Order placed: {order.order_id}",
                transaction_hash=result.get("transactionHash"),
            )
            
        except ImportError:
            logger.error("py-clob-client not installed")
            return TradeResult(
                success=False,
                order=order,
                message="Trading library not available",
                error="Install py-clob-client: pip install py-clob-client",
            )
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return TradeResult(
                success=False,
                order=order,
                message="Order placement failed",
                error=str(e),
            )
    
    def place_batch_orders(self, orders: list[Order]) -> list[TradeResult]:
        """
        Place multiple orders in a batch.
        
        Args:
            orders: List of orders to place
        
        Returns:
            List of TradeResults
        """
        results = []
        
        for order in orders:
            result = self.place_order(order)
            results.append(result)
            
            # Stop if risk limits hit
            if not result.success and "Risk limit" in result.message:
                break
        
        return results
    
    def execute_opportunity(
        self,
        opportunity: Opportunity,
        bankroll: float,
    ) -> TradeResult:
        """
        Execute a trade based on an identified opportunity.
        
        This method:
        1. Calculates optimal bet size using Kelly Criterion
        2. Applies risk limits
        3. Creates and places the order
        
        Args:
            opportunity: The opportunity to trade
            bankroll: Available bankroll for betting
        
        Returns:
            TradeResult with execution details
        """
        # Get token ID for the side we want to buy
        if opportunity.side == "yes":
            token_idx = 0
            price = opportunity.market_probability
        else:
            token_idx = 1
            price = 1 - opportunity.market_probability
        
        token_ids = opportunity.market.clob_token_ids
        if len(token_ids) <= token_idx:
            return TradeResult(
                success=False,
                order=Order(
                    market_id=opportunity.market.condition_id,
                    token_id="",
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    size=0,
                ),
                message="Invalid market: no token IDs",
                error="Market missing CLOB token IDs",
            )
        
        token_id = token_ids[token_idx]
        
        # Calculate bet size using Kelly
        bet_size = self.kelly.calculate(
            bankroll=bankroll,
            true_prob=opportunity.estimate.probability,
            market_price=opportunity.market_probability,
            side=opportunity.side,
        )
        
        # Apply per-market limit
        bet_size = min(bet_size, self.config.trading.risk.max_bet_per_market)
        
        if bet_size < 1:  # Minimum bet
            return TradeResult(
                success=False,
                order=Order(
                    market_id=opportunity.market.condition_id,
                    token_id=token_id,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    size=0,
                ),
                message="Bet size too small",
                error=f"Calculated bet ${bet_size:.2f} below minimum",
            )
        
        # Apply slippage protection - place limit order slightly above market
        slippage = self.config.trading.slippage.max_slippage
        limit_price = price * (1 + slippage) if opportunity.side == "yes" else price * (1 - slippage)
        
        # Create order
        order = Order(
            market_id=opportunity.market.condition_id,
            token_id=token_id,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=bet_size,
            price=limit_price,
            slippage=slippage,
        )
        
        mode = "[LIVE]" if self.is_enabled else "[DRY RUN]"
        logger.info(
            f"{mode} Executing opportunity: {opportunity.side.upper()} on "
            f"'{opportunity.market.question[:50]}...' "
            f"size=${bet_size:.2f} at {limit_price:.4f}"
        )
        
        return self.place_order(order)
    
    def auto_trade(
        self,
        opportunities: list[Opportunity],
        bankroll: float,
        min_edge: Optional[float] = None,
    ) -> list[TradeResult]:
        """
        Automatically trade on a list of opportunities.
        
        This executes trades on opportunities that meet the minimum
        edge threshold and pass risk checks.
        
        Args:
            opportunities: List of opportunities to consider
            bankroll: Available bankroll
            min_edge: Minimum edge to trade (default from config)
        
        Returns:
            List of TradeResults
        """
        min_edge = min_edge or self.config.trading.risk.min_auto_trade_edge
        results = []
        
        mode = "LIVE" if self.is_enabled else "DRY RUN"
        logger.info(f"[{mode}] Auto-trading {len(opportunities)} opportunities")
        
        for opp in opportunities:
            # Check minimum edge
            if abs(opp.edge) < min_edge:
                continue
            
            # Check position limit
            if len(self._positions) >= self.config.trading.risk.max_open_positions:
                logger.warning("Max open positions reached, stopping auto-trade")
                break
            
            # Execute trade
            result = self.execute_opportunity(opp, bankroll)
            results.append(result)
            
            if result.success and self.is_enabled:
                # Update available bankroll for live trading
                bankroll -= result.order.size
            
            # Check total exposure
            if self._total_exposure >= self.config.trading.risk.max_total_exposure:
                logger.warning("Max exposure reached, stopping auto-trade")
                break
        
        return results
    
    def _check_risk_limits(self, order: Order) -> dict[str, Any]:
        """
        Check if an order is within risk limits.
        
        Args:
            order: The order to check
        
        Returns:
            Dict with 'allowed' bool and 'reason' if not allowed
        """
        risk = self.config.trading.risk
        
        # Check per-market limit
        if order.size > risk.max_bet_per_market:
            return {
                "allowed": False,
                "reason": f"Bet ${order.size} exceeds max ${risk.max_bet_per_market}",
            }
        
        # Check total exposure
        new_exposure = self._total_exposure + order.size
        if new_exposure > risk.max_total_exposure:
            return {
                "allowed": False,
                "reason": f"Would exceed max exposure ${risk.max_total_exposure}",
            }
        
        # Check position count
        if len(self._positions) >= risk.max_open_positions:
            if order.market_id not in self._positions:
                return {
                    "allowed": False,
                    "reason": f"Max {risk.max_open_positions} positions reached",
                }
        
        return {"allowed": True}
    
    def _update_exposure(self, order: Order) -> None:
        """Update exposure tracking after order."""
        if order.side == OrderSide.BUY:
            self._total_exposure += order.size
    
    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        return list(self._positions.values())
    
    def get_total_exposure(self) -> float:
        """Get total USD exposure across all positions."""
        return self._total_exposure
    
    def get_order_history(self) -> list[Order]:
        """Get order history."""
        return self._order_history.copy()
    
    def close_position(self, market_id: str) -> TradeResult:
        """
        Close a position by selling all tokens.
        
        Args:
            market_id: Market condition ID to close
        
        Returns:
            TradeResult
        """
        if market_id not in self._positions:
            return TradeResult(
                success=False,
                order=Order(
                    market_id=market_id,
                    token_id="",
                    side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    size=0,
                ),
                message="No position found",
                error=f"No open position for {market_id}",
            )
        
        position = self._positions[market_id]
        
        order = Order(
            market_id=market_id,
            token_id=position.token_id,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            size=position.size,
        )
        
        result = self.place_order(order)
        
        if result.success:
            del self._positions[market_id]
            self._total_exposure -= position.value
        
        return result
    
    def close_all_positions(self) -> list[TradeResult]:
        """Close all open positions."""
        results = []
        
        for market_id in list(self._positions.keys()):
            result = self.close_position(market_id)
            results.append(result)
        
        return results


# =============================================================================
# SIMULATION MODE
# =============================================================================

class SimulatedTradingEngine(TradingEngine):
    """
    Simulated trading engine for backtesting and paper trading.
    
    This engine simulates trades without connecting to the real
    Polymarket API. Useful for:
    - Testing strategies
    - Backtesting on historical data
    - Paper trading before going live
    
    Example:
        >>> engine = SimulatedTradingEngine(initial_bankroll=10000)
        >>> result = engine.execute_opportunity(opportunity, engine.bankroll)
        >>> print(f"Simulated P&L: ${engine.pnl:.2f}")
    """
    
    def __init__(
        self,
        initial_bankroll: float = 10000,
        config: Optional[Config] = None,
    ):
        # Initialize parent with trading disabled (we handle it ourselves)
        super().__init__(config, enable_trading=False)
        
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self._simulated_trades: list[dict] = []
    
    @property
    def is_enabled(self) -> bool:
        """Simulation is always enabled."""
        return True
    
    @property
    def is_dry_run(self) -> bool:
        """Simulation is never dry run - it always simulates."""
        return False
    
    @property
    def pnl(self) -> float:
        """Calculate total profit/loss."""
        return self.bankroll - self.initial_bankroll
    
    @property
    def roi(self) -> float:
        """Calculate return on investment."""
        return (self.bankroll / self.initial_bankroll - 1) * 100
    
    def place_order(self, order: Order) -> TradeResult:
        """
        Simulate order placement.
        
        For simulation, we assume orders fill at the limit price.
        """
        # Risk checks still apply
        risk_check = self._check_risk_limits(order)
        if not risk_check["allowed"]:
            return TradeResult(
                success=False,
                order=order,
                message="Risk limit exceeded",
                error=risk_check["reason"],
            )
        
        # Simulate fill
        order.filled_size = order.size
        order.filled_price = order.price or 0.5
        order.status = "simulated_fill"
        order.order_id = f"SIM-{len(self._simulated_trades)}"
        
        # Update bankroll
        self.bankroll -= order.size
        
        # Record trade
        self._simulated_trades.append({
            "timestamp": datetime.now(),
            "order": order.to_dict(),
            "bankroll_after": self.bankroll,
        })
        
        self._order_history.append(order)
        self._update_exposure(order)
        
        logger.info(f"[SIMULATED] Order filled: {order.side.value} ${order.size:.2f}")
        
        return TradeResult(
            success=True,
            order=order,
            message=f"[SIMULATED] Order filled at {order.filled_price:.4f}",
        )
    
    def resolve_position(
        self,
        market_id: str,
        outcome: str,  # "yes" or "no"
    ) -> float:
        """
        Resolve a simulated position.
        
        Args:
            market_id: Market that resolved
            outcome: The winning outcome ("yes" or "no")
        
        Returns:
            Profit/loss from the resolution
        """
        if market_id not in self._positions:
            return 0.0
        
        position = self._positions[market_id]
        
        # Calculate payout
        if position.side == outcome:
            # We won - payout is full value
            payout = position.size / position.avg_price
        else:
            # We lost - no payout
            payout = 0
        
        pnl = payout - position.cost_basis
        self.bankroll += payout
        
        del self._positions[market_id]
        self._total_exposure -= position.cost_basis
        
        logger.info(
            f"[SIMULATED] Position resolved: {outcome.upper()} won, "
            f"P&L: ${pnl:+.2f}"
        )
        
        return pnl
    
    def get_simulation_report(self) -> dict[str, Any]:
        """Get comprehensive simulation report."""
        return {
            "initial_bankroll": self.initial_bankroll,
            "current_bankroll": self.bankroll,
            "pnl": self.pnl,
            "roi_percent": self.roi,
            "total_trades": len(self._simulated_trades),
            "open_positions": len(self._positions),
            "total_exposure": self._total_exposure,
        }
