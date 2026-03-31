"""
Backtest Engine - Simulates trading on historical data
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, MarketState, Action
from trading.portfolio import Portfolio
from trading.super_risk_manager import SuperRiskManager


@dataclass
class BacktestResult:
    """Backtest results"""
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    total_return: float
    total_return_percent: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    max_drawdown: float
    max_drawdown_percent: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade_return: float
    trades: List[Dict] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)


class BacktestEngine:
    """
    Backtesting engine for testing strategies on historical data
    """
    
    def __init__(
        self,
        agent: BaseAgent,
        initial_balance: float = 10000.0,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0005,   # 0.05%
        risk_manager: Optional[SuperRiskManager] = None
    ):
        self.agent = agent
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.risk_manager = risk_manager or SuperRiskManager()
        
        # Results
        self.portfolio: Optional[Portfolio] = None
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        
        # State
        self.current_position: Optional[Dict] = None
        
        logger.info("BacktestEngine initialized")
    
    def run(
        self,
        data: pd.DataFrame,
        symbol: str = "BTCUSDT",
        verbose: bool = False
    ) -> BacktestResult:
        """
        Run backtest on historical data
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Trading symbol
            verbose: Print detailed output
        
        Returns:
            BacktestResult with statistics
        """
        logger.info(f"Starting backtest with {len(data)} candles")
        
        # Initialize portfolio
        self.portfolio = Portfolio(self.initial_balance)
        self.trades = []
        self.equity_curve = []
        
        # Run simulation
        for i in range(len(data)):
            if i < 50:  # Need enough data for indicators
                continue
            
            # Get current candle and historical data
            current = data.iloc[i]
            historical = data.iloc[:i+1]
            
            # Create market state
            market_state = self._create_market_state(current, historical)
            
            # Get agent prediction
            action = self.agent.predict(market_state)
            
            # Execute action
            self._execute_action(action, market_state, current, symbol)
            
            # Update portfolio prices
            self.portfolio.update_price(symbol, current['close'])
            
            # Record equity
            self.equity_curve.append({
                'timestamp': data.index[i],
                'equity': self.portfolio.total_value,
                'cash': self.portfolio.cash,
                'invested': self.portfolio.invested_value
            })
            
            if verbose and i % 100 == 0:
                logger.info(f"Step {i}/{len(data)} - Equity: ${self.portfolio.total_value:,.2f}")
        
        # Close any open position
        if self.current_position:
            self._close_position(
                data.iloc[-1]['close'],
                data.index[-1],
                symbol,
                "end_of_data"
            )
        
        # Calculate results
        result = self._calculate_results(data)
        
        logger.info(f"Backtest completed. Return: {result.total_return_percent:.2f}%")
        
        return result
    
    def _create_market_state(
        self,
        current: pd.Series,
        historical: pd.DataFrame
    ) -> MarketState:
        """Create market state from data"""
        # Calculate indicators
        indicators = {
            "sma_20": historical['close'].rolling(20).mean().iloc[-1],
            "sma_50": historical['close'].rolling(50).mean().iloc[-1],
            "rsi": self._calculate_rsi(historical['close']),
            "atr": self._calculate_atr(historical),
            "volume_sma": historical['volume'].rolling(20).mean().iloc[-1]
        }
        
        return MarketState(
            price=current['close'],
            volume=current['volume'],
            timestamp=int(current.name.timestamp()),
            indicators=indicators
        )
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR"""
        if len(df) < period + 1:
            return df['close'].iloc[-1] * 0.02
        
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        atr = true_range.rolling(period).mean()
        
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else df['close'].iloc[-1] * 0.02
    
    def _execute_action(
        self,
        action: Action,
        market_state: MarketState,
        candle: pd.Series,
        symbol: str
    ):
        """Execute trading action"""
        if action.action_type == "hold":
            return
        
        # Apply slippage
        if action.action_type == "buy":
            execution_price = candle['close'] * (1 + self.slippage)
        else:
            execution_price = candle['close'] * (1 - self.slippage)
        
        if action.action_type == "buy":
            # Open position
            if not self.current_position:
                self._open_position(execution_price, candle.name, symbol, action)
        
        elif action.action_type == "sell":
            # Close position
            if self.current_position:
                self._close_position(execution_price, candle.name, symbol, "signal")
    
    def _open_position(
        self,
        price: float,
        timestamp: datetime,
        symbol: str,
        action: Action
    ):
        """Open a position"""
        # Calculate position size
        position_value = self.portfolio.cash * action.quantity
        quantity = position_value / price
        
        # Apply commission
        commission = position_value * self.commission
        
        if position_value + commission > self.portfolio.cash:
            logger.warning("Insufficient funds for trade")
            return
        
        self.current_position = {
            'entry_price': price,
            'quantity': quantity,
            'entry_time': timestamp,
            'stop_loss': action.stop_loss,
            'take_profit': action.take_profit,
            'commission': commission
        }
        
        self.portfolio.cash -= (position_value + commission)
        
        logger.debug(f"Opened position: {quantity:.6f} @ {price:.2f}")
    
    def _close_position(
        self,
        price: float,
        timestamp: datetime,
        symbol: str,
        reason: str
    ):
        """Close current position"""
        if not self.current_position:
            return
        
        position = self.current_position
        
        # Calculate exit value
        exit_value = position['quantity'] * price
        
        # Apply commission
        commission = exit_value * self.commission
        
        # Calculate P&L
        entry_value = position['quantity'] * position['entry_price']
        gross_pnl = exit_value - entry_value
        net_pnl = gross_pnl - position['commission'] - commission
        
        # Update portfolio
        self.portfolio.cash += (exit_value - commission)
        
        # Record trade
        trade = {
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'entry_price': position['entry_price'],
            'exit_price': price,
            'quantity': position['quantity'],
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'commissions': position['commission'] + commission,
            'reason': reason
        }
        
        self.trades.append(trade)
        
        logger.debug(f"Closed position: P&L=${net_pnl:.2f} ({reason})")
        
        self.current_position = None
    
    def _calculate_results(self, data: pd.DataFrame) -> BacktestResult:
        """Calculate backtest results"""
        # Basic stats
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t['net_pnl'] > 0)
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Returns
        final_balance = self.portfolio.total_value
        total_return = final_balance - self.initial_balance
        total_return_percent = (total_return / self.initial_balance) * 100
        
        # Drawdown
        equity_values = [e['equity'] for e in self.equity_curve]
        max_drawdown, max_drawdown_percent = self._calculate_drawdown(equity_values)
        
        # Sharpe ratio
        returns = [e['equity'] / equity_values[i-1] - 1 for i, e in enumerate(self.equity_curve) if i > 0]
        sharpe_ratio = self._calculate_sharpe(returns)
        
        # Profit factor
        gross_profit = sum(t['net_pnl'] for t in self.trades if t['net_pnl'] > 0)
        gross_loss = abs(sum(t['net_pnl'] for t in self.trades if t['net_pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade
        avg_trade_return = total_return / total_trades if total_trades > 0 else 0
        
        # Monthly returns
        monthly_returns = self._calculate_monthly_returns()
        
        return BacktestResult(
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_balance=self.initial_balance,
            final_balance=final_balance,
            total_return=total_return,
            total_return_percent=total_return_percent,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            max_drawdown_percent=max_drawdown_percent,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            avg_trade_return=avg_trade_return,
            trades=self.trades,
            equity_curve=self.equity_curve,
            monthly_returns=monthly_returns
        )
    
    def _calculate_drawdown(self, equity_values: List[float]) -> tuple:
        """Calculate maximum drawdown"""
        peak = equity_values[0]
        max_drawdown = 0
        max_drawdown_percent = 0
        
        for value in equity_values:
            if value > peak:
                peak = value
            
            drawdown = peak - value
            drawdown_percent = (drawdown / peak) * 100
            
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_percent = drawdown_percent
        
        return max_drawdown, max_drawdown_percent
    
    def _calculate_sharpe(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - (risk_free_rate / 252)
        
        if np.std(returns_array) == 0:
            return 0
        
        return np.mean(excess_returns) / np.std(returns_array) * np.sqrt(252)
    
    def _calculate_monthly_returns(self) -> Dict[str, float]:
        """Calculate monthly returns"""
        monthly = {}
        
        for point in self.equity_curve:
            month_key = point['timestamp'].strftime('%Y-%m')
            
            if month_key not in monthly:
                monthly[month_key] = {'start': point['equity'], 'end': point['equity']}
            else:
                monthly[month_key]['end'] = point['equity']
        
        monthly_returns = {}
        for month, values in monthly.items():
            ret = (values['end'] - values['start']) / values['start'] * 100
            monthly_returns[month] = ret
        
        return monthly_returns
    
    def get_report(self, result: BacktestResult) -> str:
        """Generate backtest report"""
        report = f"""
{'='*60}
BACKTEST REPORT
{'='*60}
Period: {result.start_date} to {result.end_date}
Initial Balance: ${result.initial_balance:,.2f}
Final Balance: ${result.final_balance:,.2f}
Total Return: ${result.total_return:,.2f} ({result.total_return_percent:.2f}%)

TRADE STATISTICS
{'='*60}
Total Trades: {result.total_trades}
Winning Trades: {result.winning_trades}
Losing Trades: {result.losing_trades}
Win Rate: {result.win_rate:.2f}%
Average Trade Return: ${result.avg_trade_return:.2f}

RISK METRICS
{'='*60}
Max Drawdown: ${result.max_drawdown:,.2f} ({result.max_drawdown_percent:.2f}%)
Sharpe Ratio: {result.sharpe_ratio:.2f}
Profit Factor: {result.profit_factor:.2f}

{'='*60}
"""
        return report
