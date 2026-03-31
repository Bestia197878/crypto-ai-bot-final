"""
Base exchange class for all exchange integrations
"""
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd
from loguru import logger


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order statuses"""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Order data class"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    commission: float = 0.0
    commission_asset: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Balance:
    """Account balance"""
    asset: str
    free: float
    locked: float
    total: float = field(init=False)
    
    def __post_init__(self):
        self.total = self.free + self.locked


@dataclass
class Ticker:
    """Price ticker"""
    symbol: str
    price: float
    bid: float
    ask: float
    volume_24h: float
    change_24h: float
    change_percent_24h: float
    timestamp: datetime


@dataclass
class Candle:
    """OHLCV candle"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class BaseExchange(ABC):
    """Base class for all exchange integrations"""
    
    def __init__(self, name: str, api_key: str = "", secret_key: str = "", testnet: bool = True):
        self.name = name
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self.is_connected = False
        self.rate_limits = {}
        
        logger.info(f"Initialized {name} exchange (testnet={testnet})")
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to exchange API"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from exchange API"""
        pass
    
    @abstractmethod
    async def get_balance(self, asset: Optional[str] = None) -> List[Balance]:
        """Get account balance"""
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current price ticker"""
        pass
    
    @abstractmethod
    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Candle]:
        """Get historical candles (OHLCV)"""
        pass
    
    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        **kwargs
    ) -> Order:
        """Place a new order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an existing order"""
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> Optional[Order]:
        """Get order details"""
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders"""
        pass
    
    @abstractmethod
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Order]:
        """Get order history"""
        pass
    
    def candles_to_dataframe(self, candles: List[Candle]) -> pd.DataFrame:
        """Convert candles to pandas DataFrame"""
        data = {
            "timestamp": [c.timestamp for c in candles],
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            "volume": [c.volume for c in candles]
        }
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate trading symbol format"""
        return len(symbol) >= 6 and symbol.isalnum()
    
    def validate_quantity(self, quantity: float, min_qty: float = 0.0, max_qty: float = float('inf')) -> bool:
        """Validate order quantity"""
        return min_qty <= quantity <= max_qty
    
    def validate_price(self, price: float, min_price: float = 0.0) -> bool:
        """Validate order price"""
        return price > min_price
    
    def handle_error(self, error: Exception, context: str = "") -> None:
        """Handle exchange errors"""
        logger.error(f"{self.name} error in {context}: {error}")
        raise error
