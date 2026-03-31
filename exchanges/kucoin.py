"""
KuCoin Exchange Integration
"""
import asyncio
import hmac
import hashlib
import base64
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
import aiohttp
from loguru import logger

from .base_exchange import (
    BaseExchange, Order, OrderType, OrderSide, OrderStatus,
    Balance, Ticker, Candle
)


class KuCoinExchange(BaseExchange):
    """KuCoin exchange integration"""
    
    REST_URLS = {
        "live": "https://api.kucoin.com",
        "sandbox": "https://openapi-sandbox.kucoin.com"
    }
    
    WS_URLS = {
        "live": "wss://ws-api.kucoin.com/endpoint",
        "sandbox": "wss://ws-api-sandbox.kucoin.com/endpoint"
    }
    
    TIMEFRAME_MAP = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1hour",
        "4h": "4hour",
        "1d": "1day",
        "1w": "1week"
    }
    
    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        passphrase: str = "",
        sandbox: bool = True
    ):
        super().__init__("KuCoin", api_key, secret_key, sandbox)
        
        self.passphrase = passphrase
        self.base_url = self.REST_URLS["sandbox"] if sandbox else self.REST_URLS["live"]
        self.ws_url = self.WS_URLS["sandbox"] if sandbox else self.WS_URLS["live"]
        
        self.session: Optional[aiohttp.ClientSession] = None
        
    def _generate_signature(self, timestamp: str, method: str, endpoint: str, body: str = "") -> tuple:
        """Generate signature for KuCoin"""
        str_for_sign = timestamp + method.upper() + endpoint + body
        signature = base64.b64encode(
            hmac.new(
                self.secret_key.encode('utf-8'),
                str_for_sign.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        passphrase_sig = base64.b64encode(
            hmac.new(
                self.secret_key.encode('utf-8'),
                self.passphrase.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        return signature, passphrase_sig
    
    def _get_headers(self, method: str, endpoint: str, body: str = "") -> Dict[str, str]:
        """Get request headers"""
        timestamp = str(int(time.time() * 1000))
        signature, encrypted_passphrase = self._generate_signature(timestamp, method, endpoint, body)
        
        return {
            "KC-API-KEY": self.api_key,
            "KC-API-SIGN": signature,
            "KC-API-TIMESTAMP": timestamp,
            "KC-API-PASSPHRASE": encrypted_passphrase,
            "KC-API-KEY-VERSION": "2",
            "Content-Type": "application/json"
        }
    
    async def connect(self) -> bool:
        """Connect to KuCoin API"""
        try:
            self.session = aiohttp.ClientSession()
            
            # Test connection
            async with self.session.get(f"{self.base_url}/api/v1/timestamp") as response:
                if response.status == 200:
                    self.is_connected = True
                    logger.info("Connected to KuCoin API")
                    return True
                else:
                    logger.error(f"Failed to connect to KuCoin: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error connecting to KuCoin: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from KuCoin API"""
        if self.session:
            await self.session.close()
            self.session = None
        self.is_connected = False
        logger.info("Disconnected from KuCoin API")
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        data: Dict = None
    ) -> Dict:
        """Make API request"""
        if not self.session:
            raise RuntimeError("Not connected to KuCoin API")
        
        url = f"{self.base_url}{endpoint}"
        body = ""
        if data:
            body = json.dumps(data)
        
        headers = self._get_headers(method, endpoint, body)
        
        async with self.session.request(
            method, url, headers=headers, params=params, data=body if body else None
        ) as response:
            resp_data = await response.json()
            
            if response.status != 200 or not resp_data.get('data'):
                logger.error(f"KuCoin API error: {resp_data}")
                raise Exception(f"API error: {resp_data.get('msg', 'Unknown error')}")
            
            return resp_data.get('data', {})
    
    async def get_balance(self, asset: Optional[str] = None) -> List[Balance]:
        """Get account balance"""
        data = await self._make_request("GET", "/api/v1/accounts")
        
        balances = []
        for account in data:
            if account.get('type') == 'trade':  # Trading account
                balance = float(account.get('balance', 0))
                available = float(account.get('available', 0))
                
                if balance > 0:
                    currency = account.get('currency', '')
                    if asset is None or currency == asset:
                        balances.append(Balance(
                            asset=currency,
                            free=available,
                            locked=balance - available
                        ))
        
        return balances
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current price ticker"""
        data = await self._make_request(
            "GET",
            "/api/v1/market/orderbook/level1",
            params={"symbol": symbol}
        )
        
        # Get 24h stats
        stats_data = await self._make_request(
            "GET",
            "/api/v1/market/stats",
            params={"symbol": symbol}
        )
        
        return Ticker(
            symbol=symbol,
            price=float(data.get('price', 0)),
            bid=float(data.get('bestBid', 0)),
            ask=float(data.get('bestAsk', 0)),
            volume_24h=float(stats_data.get('vol', 0)),
            change_24h=float(stats_data.get('changePrice', 0)),
            change_percent_24h=float(stats_data.get('changeRate', 0)) * 100,
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
            "type": self.TIMEFRAME_MAP.get(timeframe, "1hour")
        }
        
        if start_time:
            params['startAt'] = int(start_time.timestamp())
        if end_time:
            params['endAt'] = int(end_time.timestamp())
        
        data = await self._make_request("GET", "/api/v1/market/candles", params=params)
        
        candles = []
        # KuCoin returns data in reverse chronological order
        for item in reversed(data):
            candles.append(Candle(
                timestamp=datetime.fromtimestamp(int(item[0])),
                open=float(item[1]),
                close=float(item[2]),
                high=float(item[3]),
                low=float(item[4]),
                volume=float(item[5])
            ))
        
        return candles[-limit:] if len(candles) > limit else candles
    
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
        data = {
            "symbol": symbol,
            "side": side.value,
            "type": order_type.value,
            "size": quantity
        }
        
        if price:
            data['price'] = price
        
        resp_data = await self._make_request("POST", "/api/v1/orders", data=data)
        
        return Order(
            id=resp_data.get('orderId', ''),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.OPEN
        )
    
    async def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """Cancel an order"""
        try:
            await self._make_request("DELETE", f"/api/v1/orders/{order_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    async def get_order(self, order_id: str, symbol: str = None) -> Optional[Order]:
        """Get order details"""
        try:
            data = await self._make_request("GET", f"/api/v1/orders/{order_id}")
            return self._parse_order(data)
        except Exception as e:
            logger.error(f"Error getting order: {e}")
            return None
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders"""
        params = {"status": "active"}
        if symbol:
            params['symbol'] = symbol
        
        data = await self._make_request("GET", "/api/v1/orders", params=params)
        
        return [self._parse_order(item) for item in data.get('items', [])]
    
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Order]:
        """Get order history"""
        params = {"status": "done", "pageSize": limit}
        if symbol:
            params['symbol'] = symbol
        
        data = await self._make_request("GET", "/api/v1/orders", params=params)
        
        return [self._parse_order(item) for item in data.get('items', [])]
    
    def _parse_order(self, data: Dict) -> Order:
        """Parse order data"""
        status_map = {
            "active": OrderStatus.OPEN,
            "done": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "pending": OrderStatus.PENDING
        }
        
        side_map = {
            "buy": OrderSide.BUY,
            "sell": OrderSide.SELL
        }
        
        type_map = {
            "market": OrderType.MARKET,
            "limit": OrderType.LIMIT,
            "stop_loss": OrderType.STOP_LOSS,
            "stop_limit": OrderType.STOP_LIMIT
        }
        
        return Order(
            id=data.get('id', ''),
            symbol=data.get('symbol', ''),
            side=side_map.get(data.get('side'), OrderSide.BUY),
            order_type=type_map.get(data.get('type'), OrderType.MARKET),
            quantity=float(data.get('size', 0)),
            price=float(data.get('price', 0)) if data.get('price') else None,
            stop_price=float(data.get('stopPrice', 0)) if data.get('stopPrice') else None,
            status=status_map.get(data.get('status'), OrderStatus.PENDING),
            filled_quantity=float(data.get('dealSize', 0)),
            filled_price=float(data.get('dealFunds', 0)) / float(data.get('dealSize', 1)) if float(data.get('dealSize', 0)) > 0 else None,
            created_at=datetime.fromtimestamp(int(data.get('createdAt', 0)) / 1000) if data.get('createdAt') else datetime.now()
        )
    
    async def get_ws_token(self) -> str:
        """Get WebSocket token"""
        data = await self._make_request("POST", "/api/v1/bullet-public")
        return data.get('token', '')
