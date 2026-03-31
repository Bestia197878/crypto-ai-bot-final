"""
Binance Exchange Integration
"""
import asyncio
import hmac
import hashlib
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
import aiohttp
from loguru import logger

from .base_exchange import (
    BaseExchange, Order, OrderType, OrderSide, OrderStatus,
    Balance, Ticker, Candle
)


class BinanceExchange(BaseExchange):
    """Binance exchange integration"""
    
    # API endpoints
    REST_URLS = {
        "live": "https://api.binance.com",
        "testnet": "https://testnet.binance.vision"
    }
    
    WS_URLS = {
        "live": "wss://stream.binance.com:9443/ws",
        "testnet": "wss://testnet.binance.vision/ws"
    }
    
    TIMEFRAME_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
        "1w": "1w"
    }
    
    def __init__(self, api_key: str = "", secret_key: str = "", testnet: bool = True):
        super().__init__("Binance", api_key, secret_key, testnet)
        
        self.base_url = self.REST_URLS["testnet"] if testnet else self.REST_URLS["live"]
        self.ws_url = self.WS_URLS["testnet"] if testnet else self.WS_URLS["live"]
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.recv_window = 5000
        
    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC signature"""
        return hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/json"
        }
    
    async def connect(self) -> bool:
        """Connect to Binance API"""
        try:
            self.session = aiohttp.ClientSession(headers=self._get_headers())
            
            # Test connection with ping
            async with self.session.get(f"{self.base_url}/api/v3/ping") as response:
                if response.status == 200:
                    self.is_connected = True
                    logger.info("Connected to Binance API")
                    return True
                else:
                    logger.error(f"Failed to connect to Binance: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error connecting to Binance: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Binance API"""
        if self.session:
            await self.session.close()
            self.session = None
        self.is_connected = False
        logger.info("Disconnected from Binance API")
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        signed: bool = False
    ) -> Dict:
        """Make API request"""
        if not self.session:
            raise RuntimeError("Not connected to Binance API")
        
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = self.recv_window
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            params['signature'] = self._generate_signature(query_string)
        
        async with self.session.request(method, url, params=params) as response:
            data = await response.json()
            
            if response.status != 200:
                logger.error(f"Binance API error: {data}")
                raise Exception(f"API error: {data.get('msg', 'Unknown error')}")
            
            return data
    
    async def get_balance(self, asset: Optional[str] = None) -> List[Balance]:
        """Get account balance"""
        data = await self._make_request("GET", "/api/v3/account", signed=True)
        
        balances = []
        for bal in data.get('balances', []):
            free = float(bal['free'])
            locked = float(bal['locked'])
            
            if free > 0 or locked > 0:
                if asset is None or bal['asset'] == asset:
                    balances.append(Balance(
                        asset=bal['asset'],
                        free=free,
                        locked=locked
                    ))
        
        return balances
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current price ticker"""
        data = await self._make_request(
            "GET",
            "/api/v3/ticker/24hr",
            params={"symbol": symbol}
        )
        
        return Ticker(
            symbol=symbol,
            price=float(data['lastPrice']),
            bid=float(data['bidPrice']),
            ask=float(data['askPrice']),
            volume_24h=float(data['volume']),
            change_24h=float(data['priceChange']),
            change_percent_24h=float(data['priceChangePercent']),
            timestamp=datetime.now()
        )
    
    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Candle]:
        """Get historical candles"""
        params = {
            "symbol": symbol,
            "interval": self.TIMEFRAME_MAP.get(timeframe, "1h"),
            "limit": min(limit, 1000)
        }
        
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        
        data = await self._make_request("GET", "/api/v3/klines", params=params)
        
        candles = []
        for item in data:
            candles.append(Candle(
                timestamp=datetime.fromtimestamp(item[0] / 1000),
                open=float(item[1]),
                high=float(item[2]),
                low=float(item[3]),
                close=float(item[4]),
                volume=float(item[5])
            ))
        
        return candles
    
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
        params = {
            "symbol": symbol,
            "side": side.value.upper(),
            "type": order_type.value.upper(),
            "quantity": quantity
        }
        
        if price:
            params['price'] = price
        if stop_price:
            params['stopPrice'] = stop_price
        
        # Add timeInForce for limit orders
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            params['timeInForce'] = 'GTC'
        
        data = await self._make_request(
            "POST",
            "/api/v3/order",
            params=params,
            signed=True
        )
        
        return self._parse_order(data)
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        try:
            await self._make_request(
                "DELETE",
                "/api/v3/order",
                params={"symbol": symbol, "orderId": order_id},
                signed=True
            )
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    async def get_order(self, order_id: str, symbol: str) -> Optional[Order]:
        """Get order details"""
        try:
            data = await self._make_request(
                "GET",
                "/api/v3/order",
                params={"symbol": symbol, "orderId": order_id},
                signed=True
            )
            return self._parse_order(data)
        except Exception as e:
            logger.error(f"Error getting order: {e}")
            return None
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        data = await self._make_request(
            "GET",
            "/api/v3/openOrders",
            params=params,
            signed=True
        )
        
        return [self._parse_order(item) for item in data]
    
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Order]:
        """Get order history"""
        params = {"limit": limit}
        if symbol:
            params['symbol'] = symbol
        
        data = await self._make_request(
            "GET",
            "/api/v3/allOrders",
            params=params,
            signed=True
        )
        
        return [self._parse_order(item) for item in data]
    
    def _parse_order(self, data: Dict) -> Order:
        """Parse order data from API response"""
        status_map = {
            "NEW": OrderStatus.OPEN,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "FILLED": OrderStatus.FILLED,
            "CANCELED": OrderStatus.CANCELLED,
            "PENDING_CANCEL": OrderStatus.PENDING,
            "REJECTED": OrderStatus.REJECTED,
            "EXPIRED": OrderStatus.EXPIRED
        }
        
        return Order(
            id=str(data.get('orderId', '')),
            symbol=data.get('symbol', ''),
            side=OrderSide.BUY if data.get('side') == 'BUY' else OrderSide.SELL,
            order_type=OrderType.MARKET,  # Simplified
            quantity=float(data.get('origQty', 0)),
            price=float(data.get('price', 0)) if data.get('price') else None,
            status=status_map.get(data.get('status'), OrderStatus.PENDING),
            filled_quantity=float(data.get('executedQty', 0)),
            filled_price=float(data.get('avgPrice', 0)) if data.get('avgPrice') else None,
            created_at=datetime.fromtimestamp(data.get('time', 0) / 1000) if data.get('time') else datetime.now(),
            commission=float(data.get('cumQuote', 0))
        )
    
    async def get_exchange_info(self) -> Dict:
        """Get exchange information"""
        return await self._make_request("GET", "/api/v3/exchangeInfo")
    
    async def get_server_time(self) -> datetime:
        """Get server time"""
        data = await self._make_request("GET", "/api/v3/time")
        return datetime.fromtimestamp(data['serverTime'] / 1000)
