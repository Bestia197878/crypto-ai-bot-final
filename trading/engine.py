"""
Trading Engine - Main trading execution system
"""
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum
from loguru import logger
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from exchanges.base_exchange import BaseExchange, Order, OrderType, OrderSide
from agents.base_agent import BaseAgent, MarketState, Action
from trading.portfolio import Portfolio
from trading.super_risk_manager import SuperRiskManager


class TradeStatus(Enum):
    """Trade status"""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


@dataclass
class Trade:
    """Trade record"""
    id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float = 0.0
    status: TradeStatus = TradeStatus.PENDING
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pnl: float = 0.0
    pnl_percent: float = 0.0
    commission: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class Position:
    """Open position"""
    symbol: str
    side: str
    quantity: float
    avg_entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: datetime = field(default_factory=datetime.now)
    trades: List[Trade] = field(default_factory=list)


class TradingEngine:
    """
    Main trading engine that orchestrates:
    - Exchange connections
    - Agent predictions
    - Order execution
    - Portfolio management
    - Risk management
    """
    
    def __init__(
        self,
        exchange: BaseExchange,
        agent: BaseAgent,
        portfolio: Portfolio,
        risk_manager: SuperRiskManager,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h"
    ):
        self.exchange = exchange
        self.agent = agent
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.symbol = symbol
        self.timeframe = timeframe
        
        # State
        self.is_running = False
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.pending_orders: Dict[str, Order] = {}
        
        # Callbacks
        self.on_trade_opened: Optional[Callable] = None
        self.on_trade_closed: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        logger.info(f"TradingEngine initialized for {symbol}")
    
    async def start(self):
        """Start trading engine"""
        if self.is_running:
            logger.warning("Trading engine already running")
            return
        
        # Connect to exchange
        if not await self.exchange.connect():
            raise RuntimeError("Failed to connect to exchange")
        
        self.is_running = True
        logger.info("Trading engine started")
        
        # Start main loop
        await self._trading_loop()
    
    async def stop(self):
        """Stop trading engine"""
        self.is_running = False
        
        # Cancel pending orders
        for order_id in list(self.pending_orders.keys()):
            await self.exchange.cancel_order(order_id, self.symbol)
        
        await self.exchange.disconnect()
        logger.info("Trading engine stopped")
    
    async def _trading_loop(self):
        """Main trading loop"""
        while self.is_running:
            try:
                # Get market data
                ticker = await self.exchange.get_ticker(self.symbol)
                candles = await self.exchange.get_candles(
                    self.symbol,
                    self.timeframe,
                    limit=100
                )
                
                # Create market state
                market_state = self._create_market_state(ticker, candles)
                
                # Get prediction from agent
                action = self.agent.predict(market_state)
                
                # Execute action
                await self._execute_action(action, market_state)
                
                # Update positions
                await self._update_positions()
                
                # Update risk metrics
                self.risk_manager.update_drawdown(self.portfolio.total_value)
                
                # Wait for next iteration
                await asyncio.sleep(60)  # 1 minute
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                if self.on_error:
                    self.on_error(e)
                await asyncio.sleep(60)
    
    def _create_market_state(
        self,
        ticker,
        candles: List
    ) -> MarketState:
        """Create market state from data"""
        # Calculate indicators
        df = self.exchange.candles_to_dataframe(candles)
        
        indicators = {
            "sma_20": df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else ticker.price,
            "sma_50": df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else ticker.price,
            "ema_12": df['close'].ewm(span=12).mean().iloc[-1] if len(df) >= 12 else ticker.price,
            "rsi": self._calculate_rsi(df['close']),
            "atr": self._calculate_atr(df),
            "volume_sma": df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else 0,
            "trend_strength": 0.0
        }
        
        return MarketState(
            price=ticker.price,
            volume=ticker.volume_24h,
            timestamp=int(datetime.now().timestamp()),
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
    
    async def _execute_action(self, action: Action, market_state: MarketState):
        """Execute trading action"""
        if action.action_type == "hold":
            return
        
        # Check risk
        position_size = action.quantity * market_state.price
        allowed, reason = self.risk_manager.check_trade_allowed(
            self.symbol,
            action.action_type,
            position_size,
            self.portfolio.total_value
        )
        
        if not allowed:
            logger.warning(f"Trade not allowed: {reason}")
            return
        
        # Execute buy/sell
        if action.action_type == "buy":
            await self._open_long_position(action, market_state)
        elif action.action_type == "sell":
            await self._close_position(action, market_state)
    
    async def _open_long_position(self, action: Action, market_state: MarketState):
        """Open long position"""
        # Check if position already exists
        if self.symbol in self.positions:
            logger.info(f"Position already exists for {self.symbol}")
            return
        
        # Calculate quantity (action.quantity represents % of cash to use)
        max_quantity = (self.portfolio.cash / market_state.price) * action.quantity
        
        # Place order
        try:
            order = await self.exchange.place_order(
                symbol=self.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=max_quantity
            )
            
            self.pending_orders[order.id] = order
            
            # Create position
            position = Position(
                symbol=self.symbol,
                side="long",
                quantity=max_quantity,
                avg_entry_price=market_state.price,
                current_price=market_state.price,
                stop_loss=action.stop_loss,
                take_profit=action.take_profit
            )
            
            self.positions[self.symbol] = position
            
            # Create trade record
            trade = Trade(
                id=order.id,
                symbol=self.symbol,
                side="buy",
                entry_price=market_state.price,
                quantity=max_quantity,
                status=TradeStatus.OPEN,
                stop_loss=action.stop_loss,
                take_profit=action.take_profit,
                metadata={"confidence": action.confidence}
            )
            
            self.trades.append(trade)
            position.trades.append(trade)
            
            # Update portfolio
            self.portfolio.buy(self.symbol, max_quantity, market_state.price)
            
            self.total_trades += 1
            
            logger.info(f"Opened long position: {max_quantity} {self.symbol} at {market_state.price}")
            
            if self.on_trade_opened:
                self.on_trade_opened(trade)
                
        except Exception as e:
            logger.error(f"Error opening position: {e}")
    
    async def _close_position(self, action: Action, market_state: MarketState):
        """Close position"""
        if self.symbol not in self.positions:
            logger.info(f"No position to close for {self.symbol}")
            return
        
        position = self.positions[self.symbol]
        
        try:
            order = await self.exchange.place_order(
                symbol=self.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=position.quantity
            )
            
            # Calculate P&L
            pnl = (market_state.price - position.avg_entry_price) * position.quantity
            pnl_percent = (market_state.price - position.avg_entry_price) / position.avg_entry_price * 100
            
            # Update trade record
            for trade in position.trades:
                if trade.status == TradeStatus.OPEN:
                    trade.exit_price = market_state.price
                    trade.exit_time = datetime.now()
                    trade.pnl = pnl
                    trade.pnl_percent = pnl_percent
                    trade.status = TradeStatus.CLOSED
                    
                    if pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
                    
                    if self.on_trade_closed:
                        self.on_trade_closed(trade)
            
            # Update portfolio
            self.portfolio.sell(self.symbol, position.quantity, market_state.price)
            
            # Remove position
            del self.positions[self.symbol]
            
            logger.info(f"Closed position: {position.quantity} {self.symbol} at {market_state.price}, P&L: {pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    async def _update_positions(self):
        """Update open positions"""
        for symbol, position in list(self.positions.items()):
            try:
                ticker = await self.exchange.get_ticker(symbol)
                position.current_price = ticker.price
                
                # Calculate unrealized P&L
                if position.side == "long":
                    position.unrealized_pnl = (ticker.price - position.avg_entry_price) * position.quantity
                    position.unrealized_pnl_percent = (ticker.price - position.avg_entry_price) / position.avg_entry_price * 100
                
                # Check stop loss
                if position.stop_loss and ticker.price <= position.stop_loss:
                    logger.info(f"Stop loss triggered for {symbol}")
                    await self._close_position(
                        Action("sell", 1.0, ticker.price, position.quantity),
                        MarketState(ticker.price, ticker.volume_24h, int(datetime.now().timestamp()), {})
                    )
                
                # Check take profit
                elif position.take_profit and ticker.price >= position.take_profit:
                    logger.info(f"Take profit triggered for {symbol}")
                    await self._close_position(
                        Action("sell", 1.0, ticker.price, position.quantity),
                        MarketState(ticker.price, ticker.volume_24h, int(datetime.now().timestamp()), {})
                    )
                    
            except Exception as e:
                logger.error(f"Error updating position {symbol}: {e}")
    
    def get_performance_stats(self) -> Dict:
        """Get trading performance statistics"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        closed_trades = [t for t in self.trades if t.status == TradeStatus.CLOSED]
        
        avg_profit = sum(t.pnl for t in closed_trades if t.pnl > 0) / self.winning_trades if self.winning_trades > 0 else 0
        avg_loss = sum(t.pnl for t in closed_trades if t.pnl < 0) / self.losing_trades if self.losing_trades > 0 else 0
        
        profit_factor = abs(sum(t.pnl for t in closed_trades if t.pnl > 0) / sum(t.pnl for t in closed_trades if t.pnl < 0)) if sum(t.pnl for t in closed_trades if t.pnl < 0) != 0 else float('inf')
        
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_pnl": sum(t.pnl for t in closed_trades),
            "open_positions": len(self.positions)
        }
    
    def reset(self):
        """Reset trading engine"""
        self.is_running = False
        self.positions.clear()
        self.trades.clear()
        self.pending_orders.clear()
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        logger.info("Trading engine reset")
