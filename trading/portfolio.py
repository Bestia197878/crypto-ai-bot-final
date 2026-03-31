"""
Portfolio management module
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from loguru import logger
import pandas as pd


@dataclass
class Asset:
    """Asset holding"""
    symbol: str
    quantity: float
    avg_buy_price: float
    current_price: float = 0.0
    
    @property
    def value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_buy_price
    
    @property
    def unrealized_pnl(self) -> float:
        return self.value - self.cost_basis
    
    @property
    def unrealized_pnl_percent(self) -> float:
        if self.avg_buy_price == 0:
            return 0.0
        return (self.current_price - self.avg_buy_price) / self.avg_buy_price * 100


@dataclass
class Transaction:
    """Portfolio transaction"""
    timestamp: datetime
    action: str  # 'buy', 'sell', 'deposit', 'withdrawal'
    symbol: str
    quantity: float
    price: float
    value: float
    fees: float = 0.0
    metadata: Dict = field(default_factory=dict)


class Portfolio:
    """Portfolio manager"""
    
    def __init__(self, initial_balance: float = 10000.0, base_currency: str = "USDT"):
        self.initial_balance = initial_balance
        self.base_currency = base_currency
        self.cash = initial_balance
        self.assets: Dict[str, Asset] = {}
        self.transactions: List[Transaction] = []
        self.created_at = datetime.now()
        
        logger.info(f"Portfolio initialized with {initial_balance} {base_currency}")
    
    @property
    def total_value(self) -> float:
        """Total portfolio value"""
        return self.cash + sum(asset.value for asset in self.assets.values())
    
    @property
    def invested_value(self) -> float:
        """Value invested in assets"""
        return sum(asset.value for asset in self.assets.values())
    
    @property
    def total_return(self) -> float:
        """Total return in base currency"""
        return self.total_value - self.initial_balance
    
    @property
    def total_return_percent(self) -> float:
        """Total return percentage"""
        if self.initial_balance == 0:
            return 0.0
        return (self.total_value - self.initial_balance) / self.initial_balance * 100
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss"""
        return sum(asset.unrealized_pnl for asset in self.assets.values())
    
    @property
    def allocation(self) -> Dict[str, float]:
        """Portfolio allocation"""
        total = self.total_value
        if total == 0:
            return {}
        
        alloc = {symbol: asset.value / total for symbol, asset in self.assets.items()}
        alloc[self.base_currency] = self.cash / total
        return alloc
    
    def get_asset(self, symbol: str) -> Optional[Asset]:
        """Get asset by symbol"""
        return self.assets.get(symbol)
    
    def update_price(self, symbol: str, price: float):
        """Update asset price"""
        if symbol in self.assets:
            self.assets[symbol].current_price = price
    
    def buy(self, symbol: str, quantity: float, price: float, fees: float = 0.0) -> bool:
        """Buy asset"""
        cost = quantity * price + fees
        
        if cost > self.cash:
            logger.warning(f"Insufficient cash to buy {quantity} {symbol}")
            return False
        
        # Update or create asset
        if symbol in self.assets:
            asset = self.assets[symbol]
            total_cost = asset.cost_basis + cost
            total_qty = asset.quantity + quantity
            asset.avg_buy_price = total_cost / total_qty
            asset.quantity = total_qty
            asset.current_price = price
        else:
            self.assets[symbol] = Asset(
                symbol=symbol,
                quantity=quantity,
                avg_buy_price=price,
                current_price=price
            )
        
        self.cash -= cost
        
        # Record transaction
        self.transactions.append(Transaction(
            timestamp=datetime.now(),
            action="buy",
            symbol=symbol,
            quantity=quantity,
            price=price,
            value=cost,
            fees=fees
        ))
        
        logger.info(f"Bought {quantity} {symbol} at {price}")
        return True
    
    def sell(self, symbol: str, quantity: float, price: float, fees: float = 0.0) -> bool:
        """Sell asset"""
        if symbol not in self.assets:
            logger.warning(f"No {symbol} in portfolio")
            return False
        
        asset = self.assets[symbol]
        
        if quantity > asset.quantity:
            logger.warning(f"Insufficient {symbol} to sell")
            return False
        
        proceeds = quantity * price - fees
        self.cash += proceeds
        
        # Update asset
        asset.quantity -= quantity
        if asset.quantity == 0:
            del self.assets[symbol]
        
        # Record transaction
        self.transactions.append(Transaction(
            timestamp=datetime.now(),
            action="sell",
            symbol=symbol,
            quantity=quantity,
            price=price,
            value=proceeds,
            fees=fees
        ))
        
        logger.info(f"Sold {quantity} {symbol} at {price}")
        return True
    
    def deposit(self, amount: float):
        """Deposit cash"""
        self.cash += amount
        self.transactions.append(Transaction(
            timestamp=datetime.now(),
            action="deposit",
            symbol=self.base_currency,
            quantity=amount,
            price=1.0,
            value=amount
        ))
        logger.info(f"Deposited {amount} {self.base_currency}")
    
    def withdraw(self, amount: float) -> bool:
        """Withdraw cash"""
        if amount > self.cash:
            logger.warning("Insufficient cash for withdrawal")
            return False
        
        self.cash -= amount
        self.transactions.append(Transaction(
            timestamp=datetime.now(),
            action="withdrawal",
            symbol=self.base_currency,
            quantity=amount,
            price=1.0,
            value=amount
        ))
        logger.info(f"Withdrew {amount} {self.base_currency}")
        return True
    
    def get_performance_summary(self) -> Dict:
        """Get portfolio performance summary"""
        return {
            "initial_balance": self.initial_balance,
            "cash": self.cash,
            "invested_value": self.invested_value,
            "total_value": self.total_value,
            "total_return": self.total_return,
            "total_return_percent": self.total_return_percent,
            "unrealized_pnl": self.unrealized_pnl,
            "num_positions": len(self.assets),
            "allocation": self.allocation
        }
    
    def get_transaction_history(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Get transaction history as DataFrame"""
        filtered = self.transactions
        if symbol:
            filtered = [t for t in filtered if t.symbol == symbol]
        
        data = [{
            "timestamp": t.timestamp,
            "action": t.action,
            "symbol": t.symbol,
            "quantity": t.quantity,
            "price": t.price,
            "value": t.value,
            "fees": t.fees
        } for t in filtered]
        
        return pd.DataFrame(data)
    
    def reset(self):
        """Reset portfolio"""
        self.cash = self.initial_balance
        self.assets.clear()
        self.transactions.clear()
        logger.info("Portfolio reset")
