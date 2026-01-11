#!/usr/bin/env python3
"""
Polymarket Climate Monitor - Main Entry Point
==============================================

A production-ready tool for monitoring Polymarket's climate science
markets and identifying betting opportunities.

Usage:
    # Single scan for opportunities
    python main.py scan
    
    # Continuous monitoring
    python main.py monitor --interval 300
    
    # Export opportunities to CSV
    python main.py export --output opportunities.csv
    
    # Run Streamlit dashboard
    python main.py dashboard
    
    # Backtest probability model
    python main.py backtest --start 2024-01-01

For more options:
    python main.py --help
    python main.py scan --help
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

# Setup console
console = Console()


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure logging with rich formatting.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
    """
    handlers = [
        RichHandler(
            console=console,
            show_path=False,
            rich_tracebacks=True,
        )
    ]
    
    if log_file:
        # Ensure log directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )


def cmd_scan(args: argparse.Namespace) -> int:
    """
    Perform a single scan for opportunities.
    """
    from polymonitor import PolymarketClient, ClimateProbabilityModel
    from polymonitor.config import load_config
    from polymonitor.monitor import OpportunityMonitor
    
    # Load configuration
    config = load_config(args.config)
    
    # Override from command line
    min_edge = args.min_edge if args.min_edge else config.opportunities.min_edge
    min_liquidity = args.min_liquidity if args.min_liquidity else config.filters.min_liquidity
    
    console.print(Panel(
        f"[bold blue]Polymarket Climate Monitor[/bold blue]\n\n"
        f"Mode: Single Scan\n"
        f"Min Edge: {min_edge*100:.1f}%\n"
        f"Min Liquidity: ${min_liquidity:,.0f}",
        title="Configuration",
        border_style="blue",
    ))
    
    # Initialize components
    with PolymarketClient(config) as client:
        model = ClimateProbabilityModel(config)
        monitor = OpportunityMonitor(client, model, config)
        
        # Perform scan
        opportunities = monitor.scan_once(
            min_edge=min_edge,
            min_liquidity=min_liquidity,
            display=True,
        )
        
        console.print(f"\n[bold]Total opportunities found: {len(opportunities)}[/bold]")
    
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    """
    Start continuous monitoring.
    """
    from polymonitor import PolymarketClient, ClimateProbabilityModel
    from polymonitor.config import load_config, get_env
    from polymonitor.monitor import OpportunityMonitor, create_discord_notifier
    from polymonitor.trading import TradingEngine
    
    config = load_config(args.config)
    env = get_env()
    
    # Override from command line
    interval = args.interval if args.interval else config.monitoring.interval
    min_edge = args.min_edge if args.min_edge else config.opportunities.min_edge
    
    # Check auto-trading setup
    auto_trade = getattr(args, 'auto_trade', False)
    enable_trading = getattr(args, 'enable_trading', False)
    
    if auto_trade and not enable_trading:
        console.print(Panel(
            "[bold red]Auto-trading requires --enable-trading flag![/bold red]\n\n"
            "Use: python main.py --enable-trading monitor --auto-trade",
            title="Configuration Error",
            border_style="red",
        ))
        return 1
    
    mode_str = "Continuous Monitoring"
    if auto_trade:
        mode_str += " [bold red]+ AUTO-TRADING[/bold red]"
    
    console.print(Panel(
        f"[bold blue]Polymarket Climate Monitor[/bold blue]\n\n"
        f"Mode: {mode_str}\n"
        f"Interval: {interval} seconds\n"
        f"Min Edge: {min_edge*100:.1f}%\n"
        f"Trading: {'[bold red]ENABLED[/bold red]' if enable_trading else '[green]Disabled[/green]'}\n\n"
        f"[dim]Press Ctrl+C to stop[/dim]",
        title="Starting Monitor",
        border_style="blue" if not auto_trade else "red",
    ))
    
    # Create trading engine if auto-trading
    engine = None
    if auto_trade and enable_trading:
        engine = TradingEngine(config, env, enable_trading=True)
        balance = engine.get_balance()
        console.print(f"[bold]Trading balance: ${balance:,.2f}[/bold]")
    
    with PolymarketClient(config) as client:
        model = ClimateProbabilityModel(config)
        monitor = OpportunityMonitor(client, model, config)
        
        # Add Discord notifications if configured
        if config.notifications.discord.enabled and env.discord_webhook_url:
            notifier = create_discord_notifier(
                env.discord_webhook_url.get_secret_value()
            )
            monitor.add_notification_callback(notifier)
            console.print("[green]Discord notifications enabled[/green]")
        
        # Add auto-trading callback if enabled
        if engine:
            def auto_trade_callback(opp):
                if abs(opp.edge) >= config.trading.risk.min_auto_trade_edge:
                    result = engine.execute_opportunity(opp, engine.get_balance())
                    if result.success:
                        console.print(f"[green]✅ Traded: {result.message}[/green]")
                    else:
                        console.print(f"[red]❌ Trade failed: {result.error}[/red]")
            
            monitor.add_notification_callback(auto_trade_callback)
        
        # Start monitoring
        monitor.start_monitoring(
            interval=interval,
            min_edge=min_edge,
        )
    
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """
    Export opportunities to CSV.
    """
    from polymonitor.config import load_config
    from polymonitor.monitor import OpportunityMonitor
    
    config = load_config(args.config)
    
    # Create monitor just for export functionality
    monitor = OpportunityMonitor(config=config)
    monitor.export_opportunities_csv(args.output)
    
    return 0


def cmd_performance(args: argparse.Namespace) -> int:
    """
    Show performance summary.
    """
    from polymonitor.config import load_config
    from polymonitor.monitor import OpportunityMonitor
    from rich.table import Table
    
    config = load_config(args.config)
    monitor = OpportunityMonitor(config=config)
    
    summary = monitor.get_performance_summary()
    
    console.print(Panel(
        f"[bold]Performance Summary[/bold]\n\n"
        f"Total Opportunities: {summary.get('total_opportunities', 0)}\n"
        f"Average Edge: {summary.get('average_edge', 0)*100:.1f}%\n"
        f"Latest Scan: {summary.get('latest_scan', 'N/A')}",
        title="Polymonitor Performance",
        border_style="green",
    ))
    
    # Breakdown by conviction
    by_conviction = summary.get("by_conviction", {})
    if by_conviction:
        table = Table(title="By Conviction Level")
        table.add_column("Conviction")
        table.add_column("Count")
        
        for conv, count in sorted(by_conviction.items()):
            table.add_row(conv.title(), str(count))
        
        console.print(table)
    
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """
    Launch Streamlit dashboard.
    """
    import subprocess
    
    dashboard_path = Path(__file__).parent / "dashboard.py"
    
    if not dashboard_path.exists():
        console.print("[red]Dashboard not found. Create dashboard.py first.[/red]")
        return 1
    
    console.print("[blue]Launching Streamlit dashboard...[/blue]")
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(args.port),
    ]
    
    if args.headless:
        cmd.extend(["--server.headless", "true"])
    
    subprocess.run(cmd)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """
    Run backtesting on historical data.
    """
    from polymonitor.config import load_config
    from polymonitor.trading import SimulatedTradingEngine, KellyCriterion
    
    console.print(Panel(
        f"[bold blue]Backtesting Mode[/bold blue]\n\n"
        f"Start Date: {args.start}\n"
        f"Initial Bankroll: ${args.bankroll:,.0f}\n\n"
        f"[yellow]Note: Backtesting requires historical data.[/yellow]\n"
        f"[dim]Store resolved markets in data/historical/[/dim]",
        title="Backtest Configuration",
        border_style="blue",
    ))
    
    config = load_config(args.config)
    
    # Initialize simulated trading
    engine = SimulatedTradingEngine(
        initial_bankroll=args.bankroll,
        config=config,
    )
    
    # TODO: Implement full backtesting logic
    # This would:
    # 1. Load historical market data
    # 2. Run probability model on each market at the time
    # 3. Simulate trades based on opportunities
    # 4. Calculate P&L after resolution
    
    console.print("[yellow]Full backtesting not yet implemented.[/yellow]")
    console.print("[dim]To implement, add historical market data and resolution outcomes.[/dim]")
    
    # Show simulation report
    report = engine.get_simulation_report()
    console.print(f"\nSimulation Report: {report}")
    
    return 0


def cmd_clear_cache(args: argparse.Namespace) -> int:
    """
    Clear the API cache.
    """
    from polymonitor.config import load_config
    from polymonitor import PolymarketClient
    
    config = load_config(args.config)
    
    with PolymarketClient(config) as client:
        client.clear_cache()
    
    console.print("[green]Cache cleared successfully.[/green]")
    return 0


def cmd_trade(args: argparse.Namespace) -> int:
    """
    Execute trades on identified opportunities.
    """
    from polymonitor import PolymarketClient, ClimateProbabilityModel
    from polymonitor.config import load_config, get_env
    from polymonitor.monitor import OpportunityMonitor
    from polymonitor.trading import TradingEngine
    
    config = load_config(args.config)
    env = get_env()
    
    # Check if trading is enabled
    if not args.enable_trading:
        console.print(Panel(
            "[bold red]Trading is disabled![/bold red]\n\n"
            "To enable trading, use the --enable-trading flag:\n"
            "  python main.py trade --enable-trading\n\n"
            "[yellow]⚠️  WARNING: This will use real money![/yellow]",
            title="Trading Disabled",
            border_style="red",
        ))
        return 1
    
    # Verify credentials
    if not env.private_key or not env.funder_address:
        console.print(Panel(
            "[bold red]Missing credentials![/bold red]\n\n"
            "Set the following in your .env file:\n"
            "  POLYMARKET_PRIVATE_KEY=your_key\n"
            "  POLYMARKET_FUNDER_ADDRESS=your_address",
            title="Configuration Error",
            border_style="red",
        ))
        return 1
    
    # Confirm with user
    console.print(Panel(
        "[bold red]⚠️  LIVE TRADING MODE ⚠️[/bold red]\n\n"
        f"Min Edge: {(args.min_edge or config.trading.risk.min_auto_trade_edge)*100:.1f}%\n"
        f"Max Bet Per Market: ${config.trading.risk.max_bet_per_market}\n"
        f"Max Total Exposure: ${config.trading.risk.max_total_exposure}\n\n"
        "[yellow]Real money will be used![/yellow]",
        title="Trading Configuration",
        border_style="red",
    ))
    
    if not args.yes:
        confirm = console.input("[bold]Type 'CONFIRM' to proceed: [/bold]")
        if confirm != "CONFIRM":
            console.print("[yellow]Trading cancelled.[/yellow]")
            return 0
    
    # Initialize components
    with PolymarketClient(config) as client:
        model = ClimateProbabilityModel(config)
        monitor = OpportunityMonitor(client, model, config)
        engine = TradingEngine(config, env, enable_trading=True)
        
        # Get balance
        balance = engine.get_balance()
        console.print(f"[bold]Available balance: ${balance:,.2f}[/bold]")
        
        if balance <= 0:
            console.print("[red]No balance available for trading.[/red]")
            return 1
        
        # Scan for opportunities
        min_edge = args.min_edge or config.trading.risk.min_auto_trade_edge
        opportunities = monitor.scan_once(min_edge=min_edge, display=True)
        
        if not opportunities:
            console.print("[yellow]No opportunities found.[/yellow]")
            return 0
        
        # Filter by trading threshold
        tradeable = [o for o in opportunities if abs(o.edge) >= min_edge]
        
        if not tradeable:
            console.print(f"[yellow]No opportunities above {min_edge*100:.1f}% edge.[/yellow]")
            return 0
        
        console.print(f"\n[bold]Found {len(tradeable)} tradeable opportunities[/bold]")
        
        # Execute trades
        if args.auto:
            results = engine.auto_trade(tradeable, balance, min_edge)
            
            successes = sum(1 for r in results if r.success)
            console.print(f"\n[bold green]Executed {successes}/{len(results)} trades[/bold green]")
            
            for result in results:
                status = "✅" if result.success else "❌"
                console.print(f"  {status} {result.message}")
        else:
            # Interactive mode - trade one at a time
            for opp in tradeable[:args.max_trades]:
                console.print(f"\n[bold]Opportunity: {opp.side.upper()}[/bold]")
                console.print(f"  Market: {opp.market.question[:60]}...")
                console.print(f"  Edge: {opp.edge_percentage:.1f}%")
                
                bet_size = engine.kelly.calculate(
                    balance, opp.estimate.probability,
                    opp.market_probability, opp.side
                )
                bet_size = min(bet_size, config.trading.risk.max_bet_per_market)
                console.print(f"  Recommended bet: ${bet_size:.2f}")
                
                if not args.yes:
                    action = console.input("  [Trade/Skip/Quit] (t/s/q): ").lower()
                    if action == 'q':
                        break
                    if action != 't':
                        continue
                
                result = engine.execute_opportunity(opp, balance)
                
                if result.success:
                    console.print(f"  [green]✅ {result.message}[/green]")
                    balance -= result.order.size
                else:
                    console.print(f"  [red]❌ {result.message}[/red]")
    
    return 0


def main() -> int:
    """
    Main entry point.
    """
    # Create main parser
    parser = argparse.ArgumentParser(
        prog="polymonitor",
        description="Monitor Polymarket climate markets and identify betting opportunities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick scan for opportunities
    python main.py scan

    # Continuous monitoring every 5 minutes
    python main.py monitor --interval 300

    # Scan with custom thresholds
    python main.py scan --min-edge 0.10 --min-liquidity 5000

    # Launch web dashboard
    python main.py dashboard

    # Export data to CSV
    python main.py export --output my_opportunities.csv

    # Execute trades (requires --enable-trading flag)
    python main.py --enable-trading trade

    # Auto-trade all opportunities above threshold
    python main.py --enable-trading trade --auto --min-edge 0.25

    # Monitor with auto-trading enabled
    python main.py --enable-trading monitor --auto-trade
        """,
    )
    
    # Global arguments
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--log-level", "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        help="Optional log file path",
    )
    parser.add_argument(
        "--enable-trading",
        action="store_true",
        help="Enable live trading (USE WITH CAUTION - real money!)",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="%(prog)s 0.1.0",
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # === SCAN command ===
    scan_parser = subparsers.add_parser(
        "scan",
        help="Perform a single scan for opportunities",
    )
    scan_parser.add_argument(
        "--min-edge", "-e",
        type=float,
        help="Minimum edge threshold (0-1, e.g., 0.15 for 15%%)",
    )
    scan_parser.add_argument(
        "--min-liquidity", "-L",
        type=float,
        help="Minimum market liquidity in USD",
    )
    scan_parser.set_defaults(func=cmd_scan)
    
    # === MONITOR command ===
    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Start continuous monitoring",
    )
    monitor_parser.add_argument(
        "--interval", "-i",
        type=int,
        help="Seconds between scans (default: from config)",
    )
    monitor_parser.add_argument(
        "--min-edge", "-e",
        type=float,
        help="Minimum edge threshold",
    )
    monitor_parser.add_argument(
        "--auto-trade",
        action="store_true",
        help="Automatically execute trades (requires --enable-trading)",
    )
    monitor_parser.set_defaults(func=cmd_monitor)
    
    # === EXPORT command ===
    export_parser = subparsers.add_parser(
        "export",
        help="Export opportunities to CSV",
    )
    export_parser.add_argument(
        "--output", "-o",
        default="opportunities.csv",
        help="Output CSV file path",
    )
    export_parser.set_defaults(func=cmd_export)
    
    # === PERFORMANCE command ===
    perf_parser = subparsers.add_parser(
        "performance",
        help="Show performance summary",
    )
    perf_parser.set_defaults(func=cmd_performance)
    
    # === DASHBOARD command ===
    dash_parser = subparsers.add_parser(
        "dashboard",
        help="Launch Streamlit web dashboard",
    )
    dash_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Port for dashboard (default: 8501)",
    )
    dash_parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no browser)",
    )
    dash_parser.set_defaults(func=cmd_dashboard)
    
    # === BACKTEST command ===
    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Backtest probability model on historical data",
    )
    backtest_parser.add_argument(
        "--start", "-s",
        default="2024-01-01",
        help="Start date for backtest (YYYY-MM-DD)",
    )
    backtest_parser.add_argument(
        "--bankroll", "-b",
        type=float,
        default=10000,
        help="Initial bankroll for simulation (default: 10000)",
    )
    backtest_parser.set_defaults(func=cmd_backtest)
    
    # === CLEAR-CACHE command ===
    cache_parser = subparsers.add_parser(
        "clear-cache",
        help="Clear the API cache",
    )
    cache_parser.set_defaults(func=cmd_clear_cache)
    
    # === TRADE command ===
    trade_parser = subparsers.add_parser(
        "trade",
        help="Execute trades on identified opportunities (requires --enable-trading)",
    )
    trade_parser.add_argument(
        "--min-edge", "-e",
        type=float,
        help="Minimum edge threshold for trading (default: 20%%)",
    )
    trade_parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically trade all opportunities above threshold",
    )
    trade_parser.add_argument(
        "--max-trades",
        type=int,
        default=5,
        help="Maximum number of trades in interactive mode (default: 5)",
    )
    trade_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompts (dangerous!)",
    )
    trade_parser.set_defaults(func=cmd_trade)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Show help if no command
    if not args.command:
        parser.print_help()
        return 0
    
    # Execute command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logging.exception("Unhandled exception")
        return 1


if __name__ == "__main__":
    sys.exit(main())
