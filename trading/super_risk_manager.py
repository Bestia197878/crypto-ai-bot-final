"""
Super Risk Manager - Advanced risk management system
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import numpy as np
from loguru import logger


class RiskLevel(Enum):
    """Risk levels"""
    VERY_LOW = 0.1
    LOW = 0.25
    MEDIUM = 0.5
    HIGH = 0.75
    VERY_HIGH = 1.0


@dataclass
class RiskMetrics:
    """Risk metrics"""
    var_95: float  # Value at Risk (95%)
    var_99: float  # Value at Risk (99%)
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    beta: float
    correlation: float


@dataclass
class PositionRisk:
    """Risk assessment for a position"""
    symbol: str
    position_size: float
    risk_amount: float
    risk_percent: float
    stop_loss_distance: float
    risk_reward_ratio: float
    recommended_action: str


class SuperRiskManager:
    """
    Advanced risk management system that:
    - Monitors portfolio risk in real-time
    - Calculates VaR and other risk metrics
    - Manages position sizing
    - Implements stop-loss and take-profit
    - Tracks drawdowns
    """
    
    def __init__(
        self,
        max_portfolio_risk: float = 5.0,  # Max portfolio risk %
        max_position_risk: float = 2.0,  # Max risk per position %
        max_daily_loss: float = 5.0,  # Max daily loss %
        max_drawdown: float = 20.0,  # Max drawdown %
        risk_free_rate: float = 0.02,  # Annual risk-free rate
        var_confidence: float = 0.95  # VaR confidence level
    ):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.risk_free_rate = risk_free_rate
        self.var_confidence = var_confidence
        
        # Risk tracking
        self.daily_pnl: deque = deque(maxlen=30)  # 30 days
        self.hourly_returns: deque = deque(maxlen=168)  # 1 week
        self.peak_value: float = 0.0
        self.current_drawdown: float = 0.0
        self.max_observed_drawdown: float = 0.0
        
        # Position tracking
        self.position_risks: Dict[str, PositionRisk] = {}
        
        # Risk events
        self.risk_events: List[Dict] = []
        
        logger.info("SuperRiskManager initialized")
    
    def calculate_position_size(
        self,
        account_value: float,
        entry_price: float,
        stop_loss: float,
        risk_percent: float = None
    ) -> Tuple[float, float]:
        """
        Calculate position size based on risk
        
        Returns:
            (position_size, risk_amount)
        """
        risk_pct = risk_percent or self.max_position_risk
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0:
            logger.warning("Stop loss equals entry price, using default risk")
            risk_per_unit = entry_price * 0.01  # 1% default
        
        # Calculate max risk amount
        max_risk_amount = account_value * (risk_pct / 100)
        
        # Calculate position size
        position_size = max_risk_amount / risk_per_unit
        
        return position_size, max_risk_amount
    
    def calculate_var(self, returns: List[float], confidence: float = None) -> float:
        """Calculate Value at Risk"""
        if len(returns) < 30:
            return 0.0
        
        conf = confidence or self.var_confidence
        return np.percentile(returns, (1 - conf) * 100)
    
    def calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 30:
            return 0.0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - (self.risk_free_rate / 252)  # Daily
        
        if np.std(returns_array) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(returns_array) * np.sqrt(252)
    
    def calculate_volatility(self, returns: List[float]) -> float:
        """Calculate annualized volatility"""
        if len(returns) < 30:
            return 0.0
        
        return np.std(returns) * np.sqrt(252)
    
    def update_drawdown(self, current_value: float):
        """Update drawdown tracking"""
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value * 100
            self.max_observed_drawdown = max(self.max_observed_drawdown, self.current_drawdown)
    
    def assess_position_risk(
        self,
        symbol: str,
        position_size: float,
        entry_price: float,
        current_price: float,
        stop_loss: float,
        take_profit: float,
        account_value: float
    ) -> PositionRisk:
        """Assess risk for a position"""
        # Calculate risk amount
        risk_per_unit = abs(entry_price - stop_loss)
        risk_amount = position_size * risk_per_unit
        risk_percent = (risk_amount / account_value) * 100
        
        # Calculate risk/reward ratio
        potential_reward = abs(take_profit - entry_price)
        risk_reward_ratio = potential_reward / risk_per_unit if risk_per_unit > 0 else 0
        
        # Determine recommended action
        if risk_percent > self.max_position_risk:
            recommended_action = "reduce_position"
        elif risk_reward_ratio < 1.5:
            recommended_action = "review_target"
        elif current_price < stop_loss:
            recommended_action = "close_position"
        else:
            recommended_action = "hold"
        
        position_risk = PositionRisk(
            symbol=symbol,
            position_size=position_size,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
            stop_loss_distance=abs(current_price - stop_loss) / current_price * 100,
            risk_reward_ratio=risk_reward_ratio,
            recommended_action=recommended_action
        )
        
        self.position_risks[symbol] = position_risk
        return position_risk
    
    def check_trade_allowed(
        self,
        symbol: str,
        action: str,
        position_size: float,
        account_value: float
    ) -> Tuple[bool, str]:
        """Check if trade is allowed based on risk rules"""
        # Check daily loss limit
        daily_loss = sum(pnl for pnl in self.daily_pnl if pnl < 0)
        if abs(daily_loss) >= self.max_daily_loss / 100 * account_value:
            return False, "Daily loss limit reached"
        
        # Check max drawdown
        if self.current_drawdown >= self.max_drawdown:
            return False, "Max drawdown reached"
        
        # Check portfolio risk
        total_risk = sum(pr.risk_amount for pr in self.position_risks.values())
        total_risk_percent = (total_risk / account_value) * 100
        
        if total_risk_percent > self.max_portfolio_risk:
            return False, "Max portfolio risk exceeded"
        
        # Check position concentration (position_size is position value)
        position_value = position_size
        concentration = (position_value / account_value) * 100
        
        if concentration > 50:  # Max 50% in single position
            return False, "Position concentration too high"
        
        return True, "Trade allowed"
    
    def update_returns(self, hourly_return: float):
        """Update returns tracking"""
        self.hourly_returns.append(hourly_return)
    
    def record_daily_pnl(self, pnl: float):
        """Record daily P&L"""
        self.daily_pnl.append(pnl)
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics"""
        returns_list = list(self.hourly_returns)
        
        return RiskMetrics(
            var_95=self.calculate_var(returns_list, 0.95),
            var_99=self.calculate_var(returns_list, 0.99),
            max_drawdown=self.max_observed_drawdown,
            sharpe_ratio=self.calculate_sharpe_ratio(returns_list),
            volatility=self.calculate_volatility(returns_list),
            beta=0.0,  # Would need market data
            correlation=0.0  # Would need benchmark data
        )
    
    def get_risk_report(self) -> Dict:
        """Get comprehensive risk report"""
        metrics = self.get_risk_metrics()
        
        return {
            "risk_limits": {
                "max_portfolio_risk": self.max_portfolio_risk,
                "max_position_risk": self.max_position_risk,
                "max_daily_loss": self.max_daily_loss,
                "max_drawdown": self.max_drawdown
            },
            "current_metrics": {
                "current_drawdown": self.current_drawdown,
                "max_observed_drawdown": self.max_observed_drawdown,
                "var_95": metrics.var_95,
                "var_99": metrics.var_99,
                "sharpe_ratio": metrics.sharpe_ratio,
                "volatility": metrics.volatility
            },
            "position_risks": {
                symbol: {
                    "risk_percent": pr.risk_percent,
                    "risk_reward_ratio": pr.risk_reward_ratio,
                    "recommended_action": pr.recommended_action
                }
                for symbol, pr in self.position_risks.items()
            },
            "status": self._get_risk_status()
        }
    
    def _get_risk_status(self) -> str:
        """Get overall risk status"""
        if self.current_drawdown > self.max_drawdown * 0.8:
            return "CRITICAL"
        elif self.current_drawdown > self.max_drawdown * 0.5:
            return "HIGH"
        elif self.current_drawdown > self.max_drawdown * 0.25:
            return "ELEVATED"
        else:
            return "NORMAL"
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        risk_multiplier: float = 2.0,
        side: str = "buy"
    ) -> float:
        """Calculate stop loss based on ATR"""
        stop_distance = atr * risk_multiplier
        
        if side == "buy":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        risk_reward_ratio: float = 2.0
    ) -> float:
        """Calculate take profit based on risk/reward ratio"""
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio
        
        if entry_price > stop_loss:  # Long position
            return entry_price + reward
        else:  # Short position
            return entry_price - reward
    
    def reset(self):
        """Reset risk manager"""
        self.daily_pnl.clear()
        self.hourly_returns.clear()
        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.max_observed_drawdown = 0.0
        self.position_risks.clear()
        self.risk_events.clear()
        logger.info("Risk manager reset")
