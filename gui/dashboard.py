"""
Streamlit Dashboard for Crypto Trading AI - with Database Integration & Paper Trading
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.database import DatabaseManager
from trading.engine import TradeStatus

# Page configuration
st.set_page_config(
    page_title="Crypto Trading AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh counter - stored in session state to survive reruns
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

# Initialize database connection
@st.cache_resource
def get_database():
    """Get database manager instance"""
    return DatabaseManager("data/trading.db")

def load_dashboard_state(db: DatabaseManager):
    """Load dashboard state from database"""
    try:
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT trading_enabled FROM dashboard_state WHERE id = 1')
            row = cursor.fetchone()
            
            if row:
                return {'trading': bool(row[0])}
    except:
        pass
    return {'trading': False}


def save_dashboard_state(db: DatabaseManager, state: dict):
    """Save dashboard state to database"""
    try:
        with db._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dashboard_state (
                    id INTEGER PRIMARY KEY,
                    trading_enabled INTEGER DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert or update
            cursor.execute('''
                INSERT INTO dashboard_state (id, trading_enabled)
                VALUES (1, ?)
                ON CONFLICT(id) DO UPDATE SET
                    trading_enabled = excluded.trading_enabled,
                    updated_at = CURRENT_TIMESTAMP
            ''', (1 if state.get('trading', False) else 0,))
            
            conn.commit()
    except Exception as e:
        print(f"Error saving dashboard state: {e}")


def load_paper_portfolio(db: DatabaseManager):
    """Load paper portfolio from database"""
    try:
        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT cash, positions, trades FROM paper_portfolio WHERE id = 1')
            row = cursor.fetchone()
            
            if row:
                return {
                    'cash': row[0],
                    'positions': json.loads(row[1]) if row[1] else {},
                    'trades': json.loads(row[2]) if row[2] else []
                }
    except:
        pass
    return None


def save_paper_portfolio(db: DatabaseManager, portfolio: dict):
    """Save paper portfolio to database"""
    try:
        with db._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_portfolio (
                    id INTEGER PRIMARY KEY,
                    cash REAL DEFAULT 30.0,
                    positions TEXT,
                    trades TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert or update
            cursor.execute('''
                INSERT INTO paper_portfolio (id, cash, positions, trades)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    cash = excluded.cash,
                    positions = excluded.positions,
                    trades = excluded.trades,
                    updated_at = CURRENT_TIMESTAMP
            ''', (portfolio['cash'], json.dumps(portfolio['positions']), json.dumps(portfolio['trades'])))
            
            conn.commit()
    except Exception as e:
        print(f"Error saving paper portfolio: {e}")


def init_session_state():
    """Initialize session state variables"""
    if 'connected' not in st.session_state:
        st.session_state.connected = False
    if 'trading' not in st.session_state:
        st.session_state.trading = False
    if 'db' not in st.session_state:
        st.session_state.db = get_database()
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    # Load paper portfolio from database for persistence across browser restarts
    db = st.session_state.db
    saved_portfolio = load_paper_portfolio(db)
    
    if saved_portfolio:
        st.session_state.paper_portfolio = saved_portfolio
        st.session_state.paper_balance = saved_portfolio['cash']
    else:
        if 'paper_balance' not in st.session_state:
            st.session_state.paper_balance = 30.0
        if 'paper_portfolio' not in st.session_state:
            st.session_state.paper_portfolio = {
                'cash': 30.0,
                'positions': {},
                'trades': []
            }
            save_paper_portfolio(db, st.session_state.paper_portfolio)


def sidebar():
    """Render sidebar"""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
        st.title("Crypto Trading AI")
        
        st.markdown("---")
        
        # Mode selection - Paper Trading
        st.subheader("Trading Mode")
        trading_mode = st.radio(
            "Select Mode",
            ["📊 Paper Trading", "💰 Live Trading"],
            index=0
        )
        
        is_paper = "Paper" in trading_mode
        
        if is_paper:
            st.success("✅ Paper Trading Mode")
            st.info("Using virtual funds - no real money risk!")
        else:
            st.warning("⚠️ Live Trading - Real funds at risk!")
        
        st.markdown("---")
        
        # Paper trading balance
        if is_paper:
            st.subheader("Paper Balance")
            paper_balance = st.number_input(
                "Virtual Balance (USDT)",
                min_value=10.0,
                max_value=10000.0,
                value=30.0,
                step=10.0
            )
            st.metric("Current Balance", f"${paper_balance:.2f} USDT")
            st.session_state.paper_balance = paper_balance
            
            # Initialize paper trading portfolio
            if 'paper_portfolio' not in st.session_state:
                st.session_state.paper_portfolio = {
                    'cash': paper_balance,
                    'positions': {},
                    'trades': []
                }
            
            st.markdown("---")
        
        # Connection status
        db = st.session_state.db
        try:
            trades = db.get_trades(limit=1)
            st.success("🟢 Database Connected")
        except:
            st.error("🔴 Database Error")
        
        st.markdown("---")
        
        # Exchange selection
        st.subheader("Exchange")
        exchange = st.selectbox(
            "Select Exchange",
            ["Binance", "Bybit", "KuCoin"]
        )
        
        # Symbol selection
        st.subheader("Trading Pair")
        symbol = st.selectbox(
            "Select Symbol",
            ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]
        )
        
        # Timeframe
        st.subheader("Timeframe")
        timeframe = st.selectbox(
            "Select Timeframe",
            ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        )
        
        st.markdown("---")
        
        # Agent selection
        st.subheader("AI Agent")
        agent = st.radio(
            "Select Agent",
            ["Super DQN", "Super Transformer", "LSTM", "Super Ensemble", "Self-Learning"]
        )
        
        st.markdown("---")
        
        # Control buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Start Trading", key="start_trading"):
                st.session_state.trading = True
                st.success("Trading started!")
        
        with col2:
            if st.button("🛑 Stop Trading", key="stop_trading"):
                st.session_state.trading = False
                st.warning("Trading stopped!")
        
        st.markdown("---")
        
        # Reset button
        if st.button("🗑️ Reset All Data", key="reset_data"):
            reset_all_data()
            st.success("All data reset!")
            st.rerun()
        
        st.markdown("---")
        
        # Generate sample trades button
        if st.button("📊 Generate Sample Trades", key="generate_trades"):
            generate_sample_trades()
            st.success("Sample trades generated!")
        
        st.markdown("---")
        
        # Settings
        with st.expander("⚙️ Settings"):
            st.slider("Position Size (%)", 1, 100, 10)
            st.slider("Stop Loss (%)", 0.5, 10.0, 2.0)
            st.slider("Take Profit (%)", 1.0, 20.0, 4.0)
            st.slider("Max Risk (%)", 0.5, 5.0, 2.0)


def reset_all_data():
    """Reset all trading data from database and paper portfolio"""
    db = st.session_state.db
    
    try:
        # Clear all trades from database
        with db._get_connection() as conn:
            cursor = conn.cursor()
            # Delete all trades
            cursor.execute("DELETE FROM trades")
            # Delete all paper portfolio data
            cursor.execute("DELETE FROM paper_portfolio")
            # Delete all dashboard state
            cursor.execute("DELETE FROM dashboard_state")
            conn.commit()
    except Exception as e:
        print(f"Error resetting database: {e}")
    
    # Reset paper portfolio in session state
    st.session_state.paper_portfolio = {
        'cash': 30.0,
        'positions': {},
        'trades': []
    }
    st.session_state.paper_balance = 30.0
    st.session_state.trading = False
    
    # Save fresh empty portfolio
    save_paper_portfolio(db, st.session_state.paper_portfolio)


def generate_sample_trades():
    """Generate sample trades for demonstration"""
    db = st.session_state.db
    
    # Generate sample open trades
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    sides = ["buy", "sell"]
    
    for i in range(5):
        trade = {
            'id': f'sample_{i}',
            'symbol': np.random.choice(symbols),
            'side': np.random.choice(sides),
            'entry_price': np.random.uniform(40000, 60000),
            'quantity': np.random.uniform(0.01, 0.5),
            'status': 'OPEN',
            'entry_time': datetime.now() - timedelta(hours=np.random.randint(1, 48)),
            'stop_loss': np.random.uniform(38000, 55000),
            'take_profit': np.random.uniform(55000, 70000),
            'pnl': np.random.uniform(-500, 1000),
            'commission': np.random.uniform(1, 10)
        }
        db.save_trade(trade)
    
    # Generate sample closed trades
    for i in range(10):
        entry_price = np.random.uniform(40000, 60000)
        exit_price = entry_price * (1 + np.random.uniform(-0.05, 0.08))
        quantity = np.random.uniform(0.01, 0.5)
        pnl = (exit_price - entry_price) * quantity
        
        trade = {
            'id': f'sample_closed_{i}',
            'symbol': np.random.choice(symbols),
            'side': 'buy',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'status': 'CLOSED',
            'entry_time': datetime.now() - timedelta(days=np.random.randint(1, 30)),
            'exit_time': datetime.now() - timedelta(days=np.random.randint(0, 5)),
            'stop_loss': entry_price * 0.95,
            'take_profit': entry_price * 1.05,
            'pnl': pnl,
            'commission': np.random.uniform(1, 10)
        }
        db.save_trade(trade)


def portfolio_overview():
    """Display portfolio overview with real data"""
    st.header("Portfolio Overview")
    
    db = st.session_state.db
    
    # Get real trades from database
    try:
        open_trades = db.get_trades(status='OPEN', limit=100)
        closed_trades = db.get_trades(status='CLOSED', limit=100)
        all_trades = db.get_trades(limit=200)
    except:
        open_trades = []
        closed_trades = []
        all_trades = []
    
    # Calculate metrics
    total_pnl = sum(t.get('pnl', 0) or 0 for t in all_trades)
    open_positions = len(open_trades)
    
    winning_trades = [t for t in closed_trades if (t.get('pnl') or 0) > 0]
    losing_trades = [t for t in closed_trades if (t.get('pnl') or 0) <= 0]
    
    win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Trades",
            len(all_trades),
            f"{len(closed_trades)} closed"
        )
    
    with col2:
        st.metric(
            "Open Positions",
            open_positions,
            f"{len([t for t in open_trades if (t.get('pnl') or 0) > 0])} profitable"
        )
    
    with col3:
        st.metric(
            "Win Rate",
            f"{win_rate:.1f}%",
            f"{len(winning_trades)} wins / {len(losing_trades)} losses"
        )
    
    with col4:
        st.metric(
            "Total P&L",
            f"${total_pnl:,.2f}",
            f"{(total_pnl/10000)*100:.1f}%" if total_pnl else "0%"
        )
    
    with col5:
        st.metric(
            "Avg Trade",
            f"${total_pnl/len(all_trades):,.2f}" if all_trades else "$0.00",
            "per trade"
        )


def equity_chart():
    """Display equity curve chart"""
    st.subheader("Equity Curve")
    
    # Use cached data generation
    dates, values = _get_equity_data()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#00C851', width=2)
    ))
    
    fig.update_layout(
        height=300,
        xaxis_title="Date",
        yaxis_title="Value ($)",
        template="plotly_dark",
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    st.plotly_chart(fig, width='stretch')


@st.cache_data(ttl=60)
def _get_equity_data():
    """Cached equity data generation"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=100, freq='h')
    values = 10000 * (1 + np.cumsum(np.random.randn(100) * 0.001))
    return dates, values


def price_chart():
    """Display price chart with indicators"""
    st.subheader("Price Chart")
    
    # Use cached data
    dates, opens, highs, lows, closes, volumes = _get_price_data()
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=dates,
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        name="Price"
    ), row=1, col=1)
    
    # SMA
    sma20 = pd.Series(closes).rolling(20).mean()
    fig.add_trace(go.Scatter(
        x=dates,
        y=sma20,
        mode='lines',
        name='SMA 20',
        line=dict(color='orange', width=1)
    ), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(
        x=dates,
        y=volumes,
        name='Volume',
        marker_color='blue'
    ), row=2, col=1)
    
    fig.update_layout(
        height=400,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    st.plotly_chart(fig, width='stretch')


@st.cache_data(ttl=60)
def _get_price_data():
    """Cached price data generation"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=7), periods=100, freq='h')
    np.random.seed(42)
    
    opens = 45000 + np.cumsum(np.random.randn(100) * 100)
    highs = opens + np.abs(np.random.randn(100) * 200)
    lows = opens - np.abs(np.random.randn(100) * 200)
    closes = opens + np.random.randn(100) * 150
    volumes = np.random.randint(100, 1000, 100)
    
    return dates, opens, highs, lows, closes, volumes


def active_trades():
    """Display active trades from database with auto-refresh"""
    st.subheader("Active Trades")
    
    # Add refresh indicator
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"🔄 Last updated: {datetime.now().strftime('%H:%M:%S')}")
    with col2:
        if st.button("🔄 Refresh Now", key="refresh_trades"):
            st.rerun()
    
    db = st.session_state.db
    
    try:
        trades = db.get_trades(status='OPEN', limit=50)
        
        # Add paper trading positions if in paper mode
        if st.session_state.get('paper_portfolio') and st.session_state.paper_portfolio.get('positions'):
            paper_positions = st.session_state.paper_portfolio['positions']
            for symbol, pos in paper_positions.items():
                paper_trade = {
                    'trade_id': f'paper_{symbol}',
                    'symbol': symbol,
                    'side': pos['side'],
                    'entry_price': pos['entry_price'],
                    'quantity': pos['quantity'],
                    'status': 'OPEN',
                    'entry_time': pos['entry_time'],
                    'stop_loss': pos.get('stop_loss'),
                    'take_profit': pos.get('take_profit'),
                    'pnl': pos.get('unrealized_pnl', 0),
                    'is_paper': True
                }
                trades.append(paper_trade)
        
        if trades:
            df = pd.DataFrame(trades)
            # Format columns
            if 'entry_time' in df.columns:
                df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce', format='mixed')
            if 'exit_time' in df.columns:
                df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce', format='mixed')
            if 'pnl' in df.columns:
                df['pnl'] = df['pnl'].fillna(0)
            
            # Color coding
            def highlight_pnl(val):
                if pd.isna(val) or val == 0:
                    return 'background-color: #ffbb33; color: black'
                elif val > 0:
                    return 'background-color: #00C851; color: white'
                else:
                    return 'background-color: #ff4444; color: white'
            
            # Show paper trading badge
            if 'is_paper' in df.columns:
                df['type'] = df['is_paper'].apply(lambda x: '📄 Paper' if x else '🔴 Live')
            
            styled = df.style.map(highlight_pnl, subset=['pnl'] if 'pnl' in df.columns else [])
            st.dataframe(styled, width='stretch')
            
            # Paper trading actions
            if st.session_state.get('paper_portfolio') and st.session_state.paper_portfolio.get('positions'):
                st.markdown("---")
                st.subheader("📝 Paper Trading Actions")
                
                for symbol, pos in list(st.session_state.paper_portfolio['positions'].items()):
                    cols = st.columns([2, 1, 1])
                    with cols[0]:
                        st.write(f"{symbol}: {pos['quantity']:.4f} @ ${pos['entry_price']:.2f}")
                    with cols[1]:
                        if st.button(f"Close {symbol}", key=f"close_{symbol}"):
                            # Close paper position
                            close_paper_position(symbol, pos)
                            st.rerun()
        else:
            st.info("No active trades. Click 'Generate Sample Trades' or start Paper Trading!")
    except Exception as e:
        st.error(f"Error loading trades: {e}")
        st.info("No active trades. Click 'Generate Sample Trades' to see demo data.")


def close_paper_position(symbol: str, position: dict):
    """Close a paper trading position"""
    portfolio = st.session_state.paper_portfolio
    
    # Calculate PnL (simplified - using current mock price)
    current_price = position['entry_price'] * (1 + np.random.uniform(-0.02, 0.02))
    pnl = (current_price - position['entry_price']) * position['quantity']
    
    # Return funds to cash
    portfolio['cash'] += position['quantity'] * current_price
    
    # Record closed trade
    trade = {
        'symbol': symbol,
        'side': position['side'],
        'entry_price': position['entry_price'],
        'exit_price': current_price,
        'quantity': position['quantity'],
        'pnl': pnl,
        'exit_time': datetime.now(),
        'type': 'paper'
    }
    portfolio['trades'].append(trade)
    
    # Remove position
    del portfolio['positions'][symbol]
    
    # Save to database
    save_paper_portfolio(st.session_state.db, portfolio)
    
    st.success(f"Closed {symbol} position. P&L: ${pnl:.2f}")


def paper_trading_panel():
    """Paper trading control panel with automatic trading"""
    portfolio = st.session_state.paper_portfolio
    
    # Trading status indicator
    if st.session_state.get('trading', False):
        st.success("🤖 Auto-Trading ACTIVE")
        
        # Execute automatic paper trading
        if np.random.random() > 0.7:  # 30% chance to trade
            execute_auto_paper_trade(portfolio)
    else:
        st.warning("⏸️ Auto-Trading PAUSED - Click 'Start Trading' to begin")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cash", f"${portfolio['cash']:.2f}")
    
    with col2:
        positions_value = sum(
            pos['quantity'] * pos.get('current_price', pos['entry_price']) 
            for pos in portfolio['positions'].values()
        )
        st.metric("Positions Value", f"${positions_value:.2f}")
    
    with col3:
        total_value = portfolio['cash'] + positions_value
        initial = st.session_state.get('paper_balance', 30.0)
        total_pnl = total_value - initial
        st.metric("Total Value", f"${total_value:.2f}", f"${total_pnl:+.2f}")
    
    # Manual trade buttons
    st.subheader("Manual Trade")
    
    cols = st.columns([1, 1, 1, 1])
    
    with cols[0]:
        trade_symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT", "SOLUSDT"], key="paper_symbol")
    
    with cols[1]:
        trade_side = st.selectbox("Side", ["buy", "sell"], key="paper_side")
    
    with cols[2]:
        trade_qty = st.number_input("Quantity", min_value=0.001, max_value=1.0, value=0.01, step=0.001, key="paper_qty")
    
    with cols[3]:
        mock_price = {"BTCUSDT": 45000.0, "ETHUSDT": 3000.0, "SOLUSDT": 150.0}.get(trade_symbol, 100.0)
        st.write(f"Price: ${mock_price:,.2f}")
        
        if st.button("🚀 Execute", key="execute_trade"):
            execute_manual_paper_trade(trade_symbol, trade_side, trade_qty, mock_price)


def execute_auto_paper_trade(portfolio):
    """Execute automatic paper trading based on simple strategy"""
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    mock_prices = {"BTCUSDT": 45000.0, "ETHUSDT": 3000.0, "SOLUSDT": 150.0}
    
    for symbol in symbols:
        mock_price = mock_prices.get(symbol, 100.0)
        
        # Simple random strategy: buy if no position, sell if profit > 5%
        if symbol not in portfolio['positions']:
            # Buy signal - use 20% of cash
            trade_qty = (portfolio['cash'] * 0.2) / mock_price
            if trade_qty > 0.001 and portfolio['cash'] >= trade_qty * mock_price:
                portfolio['cash'] -= trade_qty * mock_price
                portfolio['positions'][symbol] = {
                    'side': 'long',
                    'quantity': trade_qty,
                    'entry_price': mock_price,
                    'entry_time': datetime.now(),
                    'stop_loss': mock_price * 0.95,
                    'take_profit': mock_price * 1.05
                }
                save_paper_portfolio(st.session_state.db, portfolio)
                st.toast(f"🤖 Auto-bought {trade_qty:.4f} {symbol}")
                break  # One trade per cycle
        else:
            # Sell signal - if unrealized PnL > 5%
            pos = portfolio['positions'][symbol]
            current_price = mock_price * (1 + np.random.uniform(-0.01, 0.01))
            unrealized_pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
            
            if unrealized_pnl_pct > 0.05 or unrealized_pnl_pct < -0.03:  # Take profit 5% or cut loss 3%
                pnl = (current_price - pos['entry_price']) * pos['quantity']
                portfolio['cash'] += pos['quantity'] * current_price
                
                # Record trade
                trade = {
                    'symbol': symbol,
                    'side': pos['side'],
                    'entry_price': pos['entry_price'],
                    'exit_price': current_price,
                    'quantity': pos['quantity'],
                    'pnl': pnl,
                    'exit_time': datetime.now(),
                    'type': 'paper_auto'
                }
                portfolio['trades'].append(trade)
                del portfolio['positions'][symbol]
                save_paper_portfolio(st.session_state.db, portfolio)
                st.toast(f"🤖 Auto-sold {symbol} | P&L: ${pnl:.2f}")
                break


def execute_manual_paper_trade(symbol, side, qty, price):
    """Execute manual paper trade"""
    portfolio = st.session_state.paper_portfolio
    cost = qty * price
    
    if side == "buy":
        if portfolio['cash'] >= cost:
            portfolio['cash'] -= cost
            portfolio['positions'][symbol] = {
                'side': 'long',
                'quantity': qty,
                'entry_price': price,
                'entry_time': datetime.now(),
                'stop_loss': price * 0.95,
                'take_profit': price * 1.05
            }
            st.success(f"Bought {qty} {symbol} @ ${price}")
            save_paper_portfolio(st.session_state.db, portfolio)
            st.rerun()
        else:
            st.error("Insufficient funds!")
    else:
        if symbol in portfolio['positions']:
            pos = portfolio['positions'][symbol]
            pnl = (price - pos['entry_price']) * qty
            portfolio['cash'] += qty * price
            
            if qty >= pos['quantity']:
                del portfolio['positions'][symbol]
            else:
                pos['quantity'] -= qty
            
            st.success(f"Sold {qty} {symbol} @ ${price}. P&L: ${pnl:.2f}")
            save_paper_portfolio(st.session_state.db, portfolio)
            st.rerun()
        else:
            st.error("No position to sell!")


def agent_status():
    """Display agent status"""
    st.subheader("AI Agent Status")
    
    agents = {
        'DQN': '🟢 Active' if np.random.random() > 0.3 else '🟡 Idle',
        'Transformer': '🟢 Active' if np.random.random() > 0.3 else '🟡 Idle',
        'LSTM': '🟢 Active' if np.random.random() > 0.3 else '🟡 Idle',
        'Ensemble': '🟢 Active'
    }
    
    cols = st.columns(len(agents))
    
    for i, (agent, status) in enumerate(agents.items()):
        with cols[i]:
            st.metric(agent, status)


def trade_history():
    """Display trade history from database"""
    st.subheader("Trade History")
    
    db = st.session_state.db
    
    try:
        trades = db.get_trades(status='CLOSED', limit=50)
        if trades:
            df = pd.DataFrame(trades)
            if 'exit_time' in df.columns:
                df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce', format='mixed')
            if 'pnl' in df.columns:
                df['pnl'] = df['pnl'].fillna(0)
            
            # Color coding
            def highlight_pnl(val):
                if pd.isna(val) or val == 0:
                    return ''
                elif val > 0:
                    return 'background-color: #00C851; color: white'
                else:
                    return 'background-color: #ff4444; color: white'
            
            styled = df.style.map(highlight_pnl, subset=['pnl'] if 'pnl' in df.columns else [])
            st.dataframe(styled, width='stretch')
        else:
            st.info("No closed trades yet.")
    except Exception as e:
        st.error(f"Error loading trade history: {e}")


def performance_metrics():
    """Display performance metrics from real data"""
    st.subheader("Performance Metrics")
    
    db = st.session_state.db
    
    try:
        trades = db.get_trades(status='CLOSED', limit=200)
        if trades:
            pnl_values = [t.get('pnl', 0) or 0 for t in trades]
            winning = [p for p in pnl_values if p > 0]
            losing = [p for p in pnl_values if p <= 0]
            
            win_rate = len(winning) / len(pnl_values) * 100 if pnl_values else 0
            profit_factor = abs(sum(winning) / sum(losing)) if losing and sum(losing) != 0 else float('inf')
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Win Rate", f"{win_rate:.1f}%", f"{len(winning)} wins")
            
            with col2:
                st.metric("Profit Factor", f"{profit_factor:.2f}")
            
            with col3:
                avg_win = np.mean(winning) if winning else 0
                st.metric("Avg Win", f"${avg_win:.2f}")
            
            with col4:
                avg_loss = np.mean(losing) if losing else 0
                st.metric("Avg Loss", f"${avg_loss:.2f}")
        else:
            # Sample metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Win Rate", "65.3%", "+2.1%")
            
            with col2:
                st.metric("Profit Factor", "1.85", "+0.12")
            
            with col3:
                st.metric("Avg Win", "$450.20")
            
            with col4:
                st.metric("Avg Loss", "-$180.50")
    except:
        # Sample metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Win Rate", "65.3%", "+2.1%")
        
        with col2:
            st.metric("Profit Factor", "1.85", "+0.12")
        
        with col3:
            st.metric("Avg Win", "$450.20")
        
        with col4:
            st.metric("Avg Loss", "-$180.50")


def recent_predictions():
    """Display recent predictions"""
    st.subheader("Recent Predictions")
    
    predictions = [
        {"time": "10:30:15", "agent": "Super DQN", "action": "BUY", "confidence": 0.85, "price": 45234.50},
        {"time": "10:15:22", "agent": "Transformer", "action": "HOLD", "confidence": 0.72, "price": 45189.20},
        {"time": "10:00:45", "agent": "LSTM", "action": "SELL", "confidence": 0.68, "price": 45120.80},
        {"time": "09:45:10", "agent": "Ensemble", "action": "BUY", "confidence": 0.91, "price": 45095.30},
    ]
    
    df = pd.DataFrame(predictions)
    
    # Color code actions
    def color_action(val):
        if val == "BUY":
            return 'background-color: #00C851; color: white'
        elif val == "SELL":
            return 'background-color: #ff4444; color: white'
        return 'background-color: #ffbb33; color: black'
    
    styled_df = df.style.map(color_action, subset=['action'])
    st.dataframe(styled_df, width='stretch')


def risk_metrics():
    """Display risk metrics"""
    st.subheader("Risk Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Daily VaR (95%)", "$125.50")
    
    with col2:
        st.metric("Portfolio Risk", "2.3%", "-0.5%")
    
    with col3:
        st.metric("Risk Status", "🟢 NORMAL")
    
    # Risk gauge - use cached
    fig = _get_risk_gauge()
    st.plotly_chart(fig, width='stretch')


@st.cache_data(ttl=60)
def _get_risk_gauge():
    """Cached risk gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=23,
        title={'text': "Risk Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00C851"},
            'steps': [
                {'range': [0, 25], 'color': "#00C851"},
                {'range': [25, 50], 'color': "#ffbb33"},
                {'range': [50, 75], 'color': "#ff8800"},
                {'range': [75, 100], 'color': "#ff4444"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0))
    return fig


def run_dashboard():
    """Main dashboard function"""
    init_session_state()
    
    sidebar()
    
    # Main content
    portfolio_overview()
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns([2, 1])
    
    with col1:
        price_chart()
    
    with col2:
        equity_chart()
    
    st.markdown("---")
    
    # Paper Trading Panel
    if st.session_state.get('paper_portfolio'):
        st.header("📝 Paper Trading Control")
        paper_trading_panel()
        st.markdown("---")
    
    # Metrics row
    col1, col2 = st.columns([2, 1])
    
    with col1:
        performance_metrics()
        agent_status()
        active_trades()
        trade_history()
    
    with col2:
        risk_metrics()
        recent_predictions()
    
    # Auto-refresh using JavaScript (preserves session state unlike meta refresh)
    if st.session_state.get('auto_refresh', True):
        import time
        time.sleep(0.1)
        st.markdown("""
            <script>
                setTimeout(function() {
                    window.location.reload();
                }, 10000);
            </script>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    run_dashboard()
