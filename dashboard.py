"""
Polymarket Climate Monitor - Streamlit Dashboard
=================================================

A web-based dashboard for visualizing opportunities and monitoring
performance.

Run with:
    streamlit run dashboard.py
    
Or via CLI:
    python main.py dashboard
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Polymonitor Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
    }
    .opportunity-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin-bottom: 0.5rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def load_opportunities() -> pd.DataFrame:
    """Load opportunities from JSON log file."""
    log_path = Path("data/opportunities.json")
    
    if not log_path.exists():
        return pd.DataFrame()
    
    try:
        with open(log_path, "r") as f:
            data = json.load(f)
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        df["detected_at"] = pd.to_datetime(df["detected_at"])
        df["edge_pct"] = df["edge"].abs() * 100
        
        return df
    except Exception as e:
        st.error(f"Error loading opportunities: {e}")
        return pd.DataFrame()


def load_config():
    """Load configuration."""
    try:
        from polymonitor.config import load_config as load_app_config
        return load_app_config()
    except ImportError:
        return None


def main():
    """Main dashboard application."""
    # Header
    st.markdown('<h1 class="main-header">🌍 Polymarket Climate Monitor</h1>', unsafe_allow_html=True)
    st.markdown("Real-time monitoring of climate science prediction markets")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Refresh interval
        refresh_interval = st.slider(
            "Auto-refresh interval (seconds)",
            min_value=30,
            max_value=600,
            value=60,
            step=30,
        )
        
        # Filter settings
        st.subheader("Filters")
        min_edge_filter = st.slider(
            "Minimum Edge (%)",
            min_value=0,
            max_value=50,
            value=15,
        )
        
        min_liquidity_filter = st.number_input(
            "Minimum Liquidity ($)",
            min_value=0,
            max_value=100000,
            value=1000,
            step=500,
        )
        
        # Time range
        time_range = st.selectbox(
            "Time Range",
            options=["Last 24 hours", "Last 7 days", "Last 30 days", "All time"],
            index=1,
        )
        
        # Actions
        st.subheader("Actions")
        if st.button("🔄 Refresh Now"):
            st.rerun()
        
        if st.button("🗑️ Clear Cache"):
            try:
                from polymonitor import PolymarketClient
                client = PolymarketClient()
                client.clear_cache()
                st.success("Cache cleared!")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Load data
    df = load_opportunities()
    
    if df.empty:
        st.warning("No opportunities found. Run a scan first:")
        st.code("python main.py scan")
        
        # Show quick start guide
        with st.expander("📚 Quick Start Guide"):
            st.markdown("""
            1. **Install dependencies:**
               ```bash
               pip install -r requirements.txt
               ```
            
            2. **Configure the monitor:**
               - Copy `config.yaml` and customize
               - Set environment variables in `.env`
            
            3. **Run your first scan:**
               ```bash
               python main.py scan
               ```
            
            4. **Start continuous monitoring:**
               ```bash
               python main.py monitor
               ```
            """)
        return
    
    # Apply filters
    # Time filter
    now = datetime.now()
    if time_range == "Last 24 hours":
        cutoff = now - timedelta(hours=24)
    elif time_range == "Last 7 days":
        cutoff = now - timedelta(days=7)
    elif time_range == "Last 30 days":
        cutoff = now - timedelta(days=30)
    else:
        cutoff = df["detected_at"].min()
    
    df_filtered = df[df["detected_at"] >= cutoff]
    df_filtered = df_filtered[df_filtered["edge_pct"] >= min_edge_filter]
    df_filtered = df_filtered[df_filtered["liquidity"] >= min_liquidity_filter]
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Opportunities",
            len(df_filtered),
            delta=f"+{len(df_filtered[df_filtered['detected_at'] >= now - timedelta(hours=24)])}" if not df_filtered.empty else None,
        )
    
    with col2:
        avg_edge = df_filtered["edge_pct"].mean() if not df_filtered.empty else 0
        st.metric(
            "Average Edge",
            f"{avg_edge:.1f}%",
        )
    
    with col3:
        total_liquidity = df_filtered["liquidity"].sum() if not df_filtered.empty else 0
        st.metric(
            "Total Liquidity",
            f"${total_liquidity:,.0f}",
        )
    
    with col4:
        high_conviction = len(df_filtered[df_filtered["conviction"].isin(["high", "very_high"])]) if not df_filtered.empty else 0
        st.metric(
            "High Conviction",
            high_conviction,
        )
    
    st.divider()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Opportunities", "📈 Analytics", "🎯 Best Bets", "📜 History"])
    
    with tab1:
        st.subheader("Current Opportunities")
        
        if df_filtered.empty:
            st.info("No opportunities match the current filters.")
        else:
            # Sort by score (or edge if no score)
            sort_col = "score" if "score" in df_filtered.columns else "edge_pct"
            df_display = df_filtered.sort_values(sort_col, ascending=False).head(20)
            
            # Display as cards
            for _, row in df_display.iterrows():
                edge_color = "🟢" if row["edge_pct"] >= 25 else "🟡" if row["edge_pct"] >= 15 else "🔵"
                side_emoji = "✅" if row["side"] == "yes" else "❌"
                
                col1, col2, col3 = st.columns([4, 1, 1])
                
                with col1:
                    st.markdown(f"**{row['market_question'][:80]}...**")
                    st.caption(f"Event: {row.get('event_title', 'N/A')}")
                
                with col2:
                    st.markdown(f"{side_emoji} **{row['side'].upper()}**")
                    st.markdown(f"{edge_color} Edge: **{row['edge_pct']:.1f}%**")
                
                with col3:
                    st.markdown(f"Market: {row['market_probability']:.1%}")
                    st.markdown(f"Fair: {row['our_probability']:.1%}")
                
                st.divider()
    
    with tab2:
        st.subheader("Analytics")
        
        if df_filtered.empty:
            st.info("Not enough data for analytics.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                # Edge distribution
                fig = px.histogram(
                    df_filtered,
                    x="edge_pct",
                    nbins=20,
                    title="Edge Distribution",
                    labels={"edge_pct": "Edge (%)"},
                    color_discrete_sequence=["#1f77b4"],
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Conviction breakdown
                conviction_counts = df_filtered["conviction"].value_counts()
                fig = px.pie(
                    values=conviction_counts.values,
                    names=conviction_counts.index,
                    title="By Conviction Level",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Opportunities over time
            df_filtered["date"] = df_filtered["detected_at"].dt.date
            daily_counts = df_filtered.groupby("date").size().reset_index(name="count")
            
            fig = px.line(
                daily_counts,
                x="date",
                y="count",
                title="Opportunities Over Time",
                markers=True,
            )
            fig.update_layout(xaxis_title="Date", yaxis_title="Opportunities Found")
            st.plotly_chart(fig, use_container_width=True)
            
            # Edge by side
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.box(
                    df_filtered,
                    x="side",
                    y="edge_pct",
                    title="Edge by Bet Side",
                    color="side",
                    color_discrete_map={"yes": "#28a745", "no": "#dc3545"},
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Liquidity vs Edge scatter
                fig = px.scatter(
                    df_filtered,
                    x="liquidity",
                    y="edge_pct",
                    color="conviction",
                    size="score" if "score" in df_filtered.columns else None,
                    title="Liquidity vs Edge",
                    hover_data=["market_question"],
                )
                fig.update_xaxes(type="log")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🎯 Best Bets")
        st.markdown("Top opportunities ranked by score (edge × liquidity × conviction)")
        
        if df_filtered.empty:
            st.info("No opportunities to display.")
        else:
            # Get top 5
            top_5 = df_filtered.nlargest(5, "score" if "score" in df_filtered.columns else "edge_pct")
            
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                with st.container():
                    st.markdown(f"### #{i}: {row['side'].upper()} - {row['market_question'][:60]}...")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Edge", f"{row['edge_pct']:.1f}%")
                    with col2:
                        st.metric("Market Price", f"{row['market_probability']:.1%}")
                    with col3:
                        st.metric("Fair Price", f"{row['our_probability']:.1%}")
                    with col4:
                        st.metric("Liquidity", f"${row['liquidity']:,.0f}")
                    
                    if row.get("reasoning"):
                        with st.expander("View Analysis"):
                            st.write(row["reasoning"])
                    
                    st.divider()
    
    with tab4:
        st.subheader("📜 Full History")
        
        if df_filtered.empty:
            st.info("No history to display.")
        else:
            # Display as dataframe
            display_cols = [
                "detected_at", "market_question", "side", "edge_pct",
                "market_probability", "our_probability", "conviction", "liquidity"
            ]
            display_cols = [c for c in display_cols if c in df_filtered.columns]
            
            st.dataframe(
                df_filtered[display_cols].sort_values("detected_at", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
            
            # Download button
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                "opportunities.csv",
                "text/csv",
                key="download-csv",
            )
    
    # Footer
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Polymonitor v0.1.0")
    
    # Auto-refresh
    if refresh_interval > 0:
        import time
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
