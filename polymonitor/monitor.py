"""
Opportunity Detection and Monitoring
=====================================

This module handles:
1. Scanning markets for betting opportunities
2. Continuous monitoring with configurable intervals
3. Opportunity logging and tracking
4. Performance measurement
5. Notifications for high-conviction opportunities

The core workflow is:
1. Fetch climate markets from Polymarket
2. For each market, estimate our probability
3. Compare to market probability to find edge
4. Rank and filter opportunities
5. Log and optionally notify
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from polymonitor.api_client import PolymarketClient
from polymonitor.config import Config, get_config
from polymonitor.models import (
    ConvictionLevel,
    Event,
    Market,
    Opportunity,
    PerformanceRecord,
    ProbabilityEstimate,
)
from polymonitor.probability_model import BaseProbabilityModel, ClimateProbabilityModel

logger = logging.getLogger(__name__)
console = Console()


class OpportunityMonitor:
    """
    Main class for monitoring Polymarket and detecting opportunities.

    This class orchestrates the entire monitoring workflow:
    1. Fetch markets from the API
    2. Run probability models
    3. Detect opportunities
    4. Rank and display results
    5. Log for tracking

    Example:
        >>> client = PolymarketClient()
        >>> model = ClimateProbabilityModel()
        >>> monitor = OpportunityMonitor(client, model)
        >>>
        >>> # Single scan
        >>> opportunities = monitor.scan_once()
        >>>
        >>> # Continuous monitoring
        >>> monitor.start_monitoring(interval=300)
    """

    def __init__(
        self,
        client: Optional[PolymarketClient] = None,
        model: Optional[BaseProbabilityModel] = None,
        config: Optional[Config] = None,
    ):
        """
        Initialize the opportunity monitor.

        Args:
            client: Polymarket API client. Creates default if None.
            model: Probability model to use. Creates default if None.
            config: Configuration. Loads from file if None.
        """
        self.config = config or get_config()
        self.client = client or PolymarketClient(self.config)
        self.model = model or ClimateProbabilityModel(self.config)

        # Tracking
        self._opportunities_log: list[Opportunity] = []
        self._performance_records: list[PerformanceRecord] = []
        self._running = False

        # Callbacks for notifications
        self._notification_callbacks: list[Callable[[Opportunity], None]] = []

        # Ensure data directories exist
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create necessary directories for logging."""
        paths = [
            Path(self.config.monitoring.log_file).parent,
            Path(self.config.monitoring.opportunities_file).parent,
            Path(self.config.monitoring.performance_file).parent,
        ]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)

    def add_notification_callback(
        self, callback: Callable[[Opportunity], None]
    ) -> None:
        """
        Add a callback function for opportunity notifications.

        The callback will be called for each high-conviction opportunity.

        Args:
            callback: Function that takes an Opportunity and handles notification

        Example:
            >>> def send_discord(opp: Opportunity):
            ...     # Send to Discord
            ...     pass
            >>>
            >>> monitor.add_notification_callback(send_discord)
        """
        self._notification_callbacks.append(callback)

    def scan_once(
        self,
        min_edge: Optional[float] = None,
        min_liquidity: Optional[float] = None,
        display: bool = True,
    ) -> list[Opportunity]:
        """
        Perform a single scan for opportunities.

        Args:
            min_edge: Minimum edge threshold (default from config)
            min_liquidity: Minimum market liquidity (default from config)
            display: Whether to display results in console

        Returns:
            List of detected opportunities, ranked by score

        Example:
            >>> opportunities = monitor.scan_once(min_edge=0.10)
            >>> print(f"Found {len(opportunities)} opportunities")
        """
        min_edge = min_edge or self.config.opportunities.min_edge
        min_liquidity = min_liquidity or self.config.filters.min_liquidity

        console.print(
            "\n[bold blue]🔍 Scanning Polymarket for climate opportunities...[/bold blue]\n"
        )

        # Fetch climate markets
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching climate markets...", total=None)

            try:
                events = self.client.get_climate_events(
                    min_liquidity=min_liquidity,
                    active_only=True,
                )
            except Exception as e:
                logger.error(f"Failed to fetch markets: {e}")
                console.print(f"[red]Error fetching markets: {e}[/red]")
                return []

            progress.update(task, description=f"Found {len(events)} climate events")

        # Extract all markets
        all_markets: list[tuple[Market, str]] = []
        for event in events:
            for market in event.markets:
                all_markets.append((market, event.title))

        console.print(f"[dim]Analyzing {len(all_markets)} markets...[/dim]\n")

        # Analyze each market
        opportunities: list[Opportunity] = []

        for market, event_title in all_markets:
            try:
                opp = self._analyze_market(market, event_title)
                if opp and abs(opp.edge) >= min_edge:
                    opportunities.append(opp)
            except Exception as e:
                logger.warning(f"Error analyzing market {market.id}: {e}")
                continue

        # Rank opportunities
        opportunities = self._rank_opportunities(opportunities)

        # Log opportunities
        self._log_opportunities(opportunities)

        # Display results
        if display:
            self._display_opportunities(opportunities)

        # Send notifications for high-conviction opportunities
        self._send_notifications(opportunities)

        return opportunities

    def _analyze_market(
        self, market: Market, event_title: str
    ) -> Optional[Opportunity]:
        """
        Analyze a single market for opportunity.

        Args:
            market: The market to analyze
            event_title: Title of the parent event

        Returns:
            Opportunity if edge exists, None otherwise
        """
        # Skip resolved or closed markets
        if market.closed or market.resolution:
            return None

        # Get our probability estimate
        estimate = self.model.estimate(market)

        # Get market probability
        market_prob = market.yes_price

        # Calculate edge
        # Positive edge = we think YES is underpriced
        # Negative edge = we think NO is underpriced
        edge = estimate.probability - market_prob

        # Determine which side to bet
        if edge > 0:
            side = "yes"
            bet_edge = edge
        else:
            side = "no"
            bet_edge = -edge  # Absolute edge for betting

        # Calculate Kelly fraction
        kelly = self._calculate_kelly(estimate.probability, market_prob, side)

        # Calculate recommended bet (using configured fraction of Kelly)
        bankroll = 1000  # Default bankroll for sizing
        recommended_bet = kelly * bankroll * self.config.trading.kelly.fraction

        # Create opportunity
        return Opportunity(
            market=market,
            event_title=event_title,
            estimate=estimate,
            market_probability=market_prob,
            side=side,
            edge=edge,
            kelly_fraction=kelly,
            recommended_bet=recommended_bet,
            score=0,  # Will be set during ranking
        )

    def _calculate_kelly(
        self,
        true_prob: float,
        market_prob: float,
        side: str,
    ) -> float:
        """
        Calculate Kelly Criterion fraction.

        Kelly formula: f* = (bp - q) / b
        where:
        - b = odds received (decimal - 1)
        - p = probability of winning
        - q = probability of losing

        Args:
            true_prob: Our estimated true probability of YES
            market_prob: Market probability of YES
            side: Which side we're betting ('yes' or 'no')

        Returns:
            Kelly fraction (0-1)
        """
        if side == "yes":
            p = true_prob
            b = (1 / market_prob) - 1 if market_prob > 0 else 0
        else:
            p = 1 - true_prob
            b = (1 / (1 - market_prob)) - 1 if market_prob < 1 else 0

        q = 1 - p

        if b <= 0:
            return 0.0

        kelly = (b * p - q) / b
        return max(0, min(kelly, 1))  # Clamp to [0, 1]

    def _rank_opportunities(
        self, opportunities: list[Opportunity]
    ) -> list[Opportunity]:
        """
        Rank opportunities by composite score.

        Score is weighted combination of:
        - Edge size (normalized)
        - Market liquidity (log-normalized)
        - Conviction level

        Args:
            opportunities: List of opportunities to rank

        Returns:
            Sorted list with scores set
        """
        if not opportunities:
            return []

        weights = self.config.opportunities.ranking

        # Normalize edge (0-1 scale)
        max_edge = max(abs(o.edge) for o in opportunities) or 1

        # Normalize liquidity (log scale)
        max_liquidity = max(o.market.liquidity for o in opportunities) or 1

        # Conviction weights
        conviction_weights = {
            ConvictionLevel.LOW: 0.25,
            ConvictionLevel.MEDIUM: 0.50,
            ConvictionLevel.HIGH: 0.75,
            ConvictionLevel.VERY_HIGH: 1.0,
        }

        for opp in opportunities:
            edge_score = abs(opp.edge) / max_edge

            # Log-normalize liquidity to reduce impact of outliers
            import math

            liquidity_score = math.log1p(opp.market.liquidity) / math.log1p(
                max_liquidity
            )

            conviction_score = conviction_weights.get(opp.estimate.conviction, 0.5)

            opp.score = (
                weights.edge_weight * edge_score
                + weights.liquidity_weight * liquidity_score
                + weights.conviction_weight * conviction_score
            )

        # Sort by score (highest first)
        return sorted(opportunities, key=lambda o: o.score, reverse=True)

    def _display_opportunities(self, opportunities: list[Opportunity]) -> None:
        """
        Display opportunities in a formatted table.

        Uses rich for colorful, readable output.
        """
        if not opportunities:
            console.print("[yellow]No opportunities found above threshold.[/yellow]")
            return

        # Create summary
        console.print(
            Panel(
                f"[bold green]Found {len(opportunities)} opportunities[/bold green]",
                title="Scan Results",
                border_style="green",
            )
        )

        # Create table
        table = Table(
            title="Climate Market Opportunities",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Rank", style="dim", width=4)
        table.add_column("Market", style="cyan", max_width=45)
        table.add_column("Side", style="bold", width=6)
        table.add_column("Edge", style="green", width=8)
        table.add_column("Market", width=8)
        table.add_column("Fair", width=8)
        table.add_column("Conviction", width=10)
        table.add_column("Liquidity", width=10)
        table.add_column("Score", width=6)

        thresholds = self.config.opportunities.edge_thresholds

        for i, opp in enumerate(opportunities[:20], 1):  # Top 20
            # Color code by edge size
            edge_pct = abs(opp.edge) * 100
            if edge_pct >= thresholds.excellent * 100:
                edge_style = "bold red"
            elif edge_pct >= thresholds.great * 100:
                edge_style = "bold yellow"
            else:
                edge_style = "green"

            # Color code side
            side_style = "bold green" if opp.side == "yes" else "bold red"

            # Truncate question
            question = opp.market.question
            if len(question) > 42:
                question = question[:42] + "..."

            table.add_row(
                str(i),
                question,
                f"[{side_style}]{opp.side.upper()}[/{side_style}]",
                f"[{edge_style}]{edge_pct:+.1f}%[/{edge_style}]",
                f"{opp.market_probability:.1%}",
                f"{opp.estimate.probability:.1%}",
                opp.conviction_label,
                f"${opp.market.liquidity:,.0f}",
                f"{opp.score:.2f}",
            )

        console.print(table)

        # Show top opportunity details
        if opportunities:
            top = opportunities[0]
            console.print(
                Panel(
                    f"[bold]Market:[/bold] {top.market.question}\n\n"
                    f"[bold]Event:[/bold] {top.event_title}\n\n"
                    f"[bold]Analysis:[/bold]\n{top.estimate.reasoning}\n\n"
                    f"[bold]Data Sources:[/bold] {', '.join(top.estimate.data_sources)}\n\n"
                    f"[bold]Kelly Fraction:[/bold] {top.kelly_fraction:.2%}\n"
                    f"[bold]Recommended Bet:[/bold] ${top.recommended_bet:.2f}",
                    title="[bold green]Top Opportunity Details[/bold green]",
                    border_style="green",
                )
            )

    def _log_opportunities(self, opportunities: list[Opportunity]) -> None:
        """
        Log opportunities to JSON file for tracking.
        """
        if not opportunities:
            return

        log_path = Path(self.config.monitoring.opportunities_file)

        # Load existing log
        existing: list[dict] = []
        if log_path.exists():
            try:
                with open(log_path, "r") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing = []

        # Add new opportunities
        for opp in opportunities:
            record = {
                "detected_at": opp.detected_at.isoformat(),
                "market_id": opp.market.id,
                "market_question": opp.market.question,
                "event_title": opp.event_title,
                "side": opp.side,
                "edge": opp.edge,
                "market_probability": opp.market_probability,
                "our_probability": opp.estimate.probability,
                "conviction": opp.estimate.conviction.value,
                "liquidity": opp.market.liquidity,
                "score": opp.score,
                "reasoning": opp.estimate.reasoning,
            }
            existing.append(record)

        # Keep last 1000 records
        existing = existing[-1000:]

        # Save
        with open(log_path, "w") as f:
            json.dump(existing, f, indent=2, default=str)

        logger.info(f"Logged {len(opportunities)} opportunities to {log_path}")

    def _send_notifications(self, opportunities: list[Opportunity]) -> None:
        """
        Send notifications for high-conviction opportunities.
        """
        if not self.config.notifications.enabled:
            return

        min_edge = self.config.notifications.discord.min_edge_notify

        for opp in opportunities:
            # Only notify for high-edge opportunities
            if abs(opp.edge) < min_edge:
                continue

            for callback in self._notification_callbacks:
                try:
                    callback(opp)
                except Exception as e:
                    logger.error(f"Notification callback failed: {e}")

    def start_monitoring(
        self,
        interval: Optional[int] = None,
        min_edge: Optional[float] = None,
    ) -> None:
        """
        Start continuous monitoring loop.

        Args:
            interval: Seconds between scans (default from config)
            min_edge: Minimum edge threshold

        Note:
            Use Ctrl+C to stop monitoring.
        """
        interval = interval or self.config.monitoring.interval
        self._running = True

        console.print(
            Panel(
                f"[bold]Starting continuous monitoring[/bold]\n\n"
                f"Interval: {interval} seconds\n"
                f"Min Edge: {(min_edge or self.config.opportunities.min_edge)*100:.1f}%\n\n"
                f"Press Ctrl+C to stop.",
                title="[bold blue]Polymonitor[/bold blue]",
                border_style="blue",
            )
        )

        scan_count = 0

        try:
            while self._running:
                scan_count += 1
                console.print(
                    f"\n[dim]─── Scan #{scan_count} at {datetime.now().strftime('%H:%M:%S')} ───[/dim]"
                )

                try:
                    self.scan_once(min_edge=min_edge)
                except Exception as e:
                    logger.error(f"Scan failed: {e}")
                    console.print(f"[red]Scan error: {e}[/red]")

                # Wait for next interval
                console.print(f"[dim]Next scan in {interval} seconds...[/dim]")
                time.sleep(interval)

        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped by user.[/yellow]")
        finally:
            self._running = False

    def stop_monitoring(self) -> None:
        """Stop the monitoring loop."""
        self._running = False

    def get_performance_summary(self) -> dict:
        """
        Get summary of performance metrics.

        Returns:
            Dictionary with performance statistics
        """
        log_path = Path(self.config.monitoring.opportunities_file)

        if not log_path.exists():
            return {
                "total_opportunities": 0,
                "total_scans": 0,
            }

        try:
            with open(log_path, "r") as f:
                records = json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"total_opportunities": 0}

        # Calculate statistics
        total = len(records)

        if total == 0:
            return {"total_opportunities": 0}

        edges = [r["edge"] for r in records]
        avg_edge = sum(abs(e) for e in edges) / total

        by_conviction = {}
        for r in records:
            conv = r.get("conviction", "low")
            by_conviction[conv] = by_conviction.get(conv, 0) + 1

        return {
            "total_opportunities": total,
            "average_edge": avg_edge,
            "by_conviction": by_conviction,
            "latest_scan": records[-1]["detected_at"] if records else None,
        }

    def export_opportunities_csv(self, output_path: str = "opportunities.csv") -> None:
        """
        Export opportunities to CSV for analysis.

        Args:
            output_path: Path for output CSV file
        """
        import csv

        log_path = Path(self.config.monitoring.opportunities_file)

        if not log_path.exists():
            console.print("[yellow]No opportunities to export.[/yellow]")
            return

        with open(log_path, "r") as f:
            records = json.load(f)

        if not records:
            console.print("[yellow]No opportunities to export.[/yellow]")
            return

        # Write CSV
        fieldnames = list(records[0].keys())

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        console.print(
            f"[green]Exported {len(records)} opportunities to {output_path}[/green]"
        )


# =============================================================================
# NOTIFICATION HELPERS
# =============================================================================


def create_discord_notifier(webhook_url: str) -> Callable[[Opportunity], None]:
    """
    Create a Discord notification callback.

    Args:
        webhook_url: Discord webhook URL

    Returns:
        Callback function for sending Discord notifications

    Example:
        >>> notifier = create_discord_notifier("https://discord.com/api/webhooks/...")
        >>> monitor.add_notification_callback(notifier)
    """
    from discord_webhook import DiscordWebhook, DiscordEmbed

    def notify(opp: Opportunity) -> None:
        webhook = DiscordWebhook(url=webhook_url)

        # Create embed
        embed = DiscordEmbed(
            title=f"🎯 New Opportunity: {opp.side.upper()}",
            description=opp.market.question,
            color="03b2f8" if opp.side == "yes" else "f85a3e",
        )

        embed.add_embed_field(
            name="Edge", value=f"{opp.edge_percentage:.1f}%", inline=True
        )
        embed.add_embed_field(
            name="Market Price", value=f"{opp.market_probability:.1%}", inline=True
        )
        embed.add_embed_field(
            name="Fair Price", value=f"{opp.estimate.probability:.1%}", inline=True
        )
        embed.add_embed_field(
            name="Conviction", value=opp.conviction_label, inline=True
        )
        embed.add_embed_field(
            name="Liquidity", value=f"${opp.market.liquidity:,.0f}", inline=True
        )
        embed.add_embed_field(name="Score", value=f"{opp.score:.2f}", inline=True)
        embed.add_embed_field(
            name="Reasoning", value=opp.estimate.reasoning[:1000], inline=False
        )

        embed.set_footer(
            text=f"Detected at {opp.detected_at.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        webhook.add_embed(embed)
        webhook.execute()

    return notify
