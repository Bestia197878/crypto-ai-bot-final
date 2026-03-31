"""
Trading Performance Analyzer
Analyzes effectiveness of trading agents and bots
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils.database import DatabaseManager
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_trading_performance():
    """Analyze trading performance from database"""
    db = DatabaseManager("data/trading.db")
    
    print("=" * 60)
    print("📊 TRADING PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Get all trades
    all_trades = db.get_trades(limit=1000)
    
    if not all_trades:
        print("\n⚠️  No trades found in database")
        print("Run 'Generate Sample Trades' in dashboard first!")
        return
    
    df = pd.DataFrame(all_trades)
    
    # Basic statistics
    print(f"\n📈 OVERALL STATISTICS")
    print(f"   Total Trades: {len(df)}")
    print(f"   Open Trades: {len(df[df['status'] == 'OPEN'])}")
    print(f"   Closed Trades: {len(df[df['status'] == 'CLOSED'])}")
    
    # Closed trades analysis
    closed = df[df['status'] == 'CLOSED'].copy()
    if len(closed) > 0:
        closed['pnl'] = closed['pnl'].fillna(0)
        
        winning = closed[closed['pnl'] > 0]
        losing = closed[closed['pnl'] <= 0]
        
        win_rate = len(winning) / len(closed) * 100 if len(closed) > 0 else 0
        
        print(f"\n🏆 WIN/LOSS ANALYSIS")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Winning Trades: {len(winning)}")
        print(f"   Losing Trades: {len(losing)}")
        
        total_pnl = closed['pnl'].sum()
        avg_pnl = closed['pnl'].mean()
        
        print(f"\n💰 PROFITABILITY")
        print(f"   Total P&L: ${total_pnl:,.2f}")
        print(f"   Average P&L per Trade: ${avg_pnl:,.2f}")
        
        if len(winning) > 0:
            avg_win = winning['pnl'].mean()
            max_win = winning['pnl'].max()
            print(f"   Average Win: ${avg_win:,.2f}")
            print(f"   Max Win: ${max_win:,.2f}")
        
        if len(losing) > 0:
            avg_loss = losing['pnl'].mean()
            max_loss = losing['pnl'].min()
            print(f"   Average Loss: ${avg_loss:,.2f}")
            print(f"   Max Loss: ${max_loss:,.2f}")
        
        # Profit factor
        gross_profit = winning['pnl'].sum() if len(winning) > 0 else 0
        gross_loss = abs(losing['pnl'].sum()) if len(losing) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
        
        print(f"\n📊 RATIOS")
        print(f"   Profit Factor: {profit_factor:.2f}")
        print(f"   Risk/Reward Ratio: 1:{abs(avg_win/avg_loss):.1f}" if len(winning) > 0 and len(losing) > 0 else "   Risk/Reward: N/A")
    
    # Symbol analysis
    print(f"\n📉 SYMBOL PERFORMANCE")
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol]
        symbol_closed = symbol_df[symbol_df['status'] == 'CLOSED']
        if len(symbol_closed) > 0:
            symbol_pnl = symbol_closed['pnl'].fillna(0).sum()
            symbol_win_rate = len(symbol_closed[symbol_closed['pnl'] > 0]) / len(symbol_closed) * 100
            print(f"   {symbol}: {len(symbol_closed)} trades, P&L: ${symbol_pnl:,.2f}, Win Rate: {symbol_win_rate:.1f}%")
    
    # Agent effectiveness (from predictions if available)
    try:
        with db._get_connection() as conn:
            predictions = pd.read_sql_query(
                "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 100",
                conn
            )
        
        if not predictions.empty:
            print(f"\n🤖 AGENT PREDICTIONS")
            for agent in predictions['agent_name'].unique():
                agent_preds = predictions[predictions['agent_name'] == agent]
                avg_confidence = agent_preds['confidence'].mean()
                action_counts = agent_preds['action'].value_counts()
                print(f"   {agent}: {len(agent_preds)} predictions, Avg Confidence: {avg_confidence:.2%}")
                for action, count in action_counts.items():
                    print(f"      - {action}: {count}")
    except:
        print("\n🤖 No agent predictions in database yet")
    
    # Performance rating
    print(f"\n⭐ OVERALL RATING")
    if win_rate >= 60 and profit_factor >= 1.5:
        print("   EXCELLENT - High win rate and good profit factor")
    elif win_rate >= 50 and profit_factor >= 1.2:
        print("   GOOD - Profitable trading strategy")
    elif win_rate >= 40:
        print("   MODERATE - Needs improvement")
    else:
        print("   NEEDS WORK - Review strategy")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    analyze_trading_performance()
