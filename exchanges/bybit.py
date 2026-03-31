"""
Bybit Exchange Integration
"""
import asyncio
import hmac
import hashlib
import time
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
import aiohttp
from loguru import logger

from .base_exchange import (
    BaseExchange, Order, OrderType, OrderSide, OrderStatus,
    Balance, Ticker, Candle
)


class BybitExchange(BaseExchange):
    """Bybit exchange integration"""
    
    REST_URLS = {
        "live": "https://api.bybit.com",
        "testnet": "https://api-testnet.bybit.com"
    }
    
    WS_URLS = {
        "live": "wss://stream.bybit.com/v5/public/spot",
        "testnet": "wss://stream-testnet.bybit.com/v5/public/spot"
    }
    
    TIMEFRAME_MAP = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "4h": "240",
        "1d": "D",
        "1w": "W"
    }
    
    def __init__(self, api_key: str = "", secret_key: str = "", testnet: bool = True):
        super().__init__("Bybit", api_key, secret_key, testnet)
        
        self.base_url = self.REST_URLS["testnet"] if testnet else self.REST_URLS["live"]
        self.ws_url = self.WS_URLS["testnet"] if testnet else self.WS_URLS["live"]
        
        self.session: Optional[aiohttp.ClientSession] = None
        
    def _generate_signature(self, params: Dict) -> str:
        """Generate HMAC signature"""
        timestamp = str(int(time.time() * 1000))
        params['timestamp'] = timestamp
        
        param_str = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature, timestamp
    
    def _get_headers(self, signature: str, timestamp: str) -> Dict[str, str]:
        """Get request headers"""
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": "5000",
            "Content-Type": "application/json"
        }
    
    async def connect(self) -> bool:
        """Connect to Bybit API"""
        try:
            self.session = aiohttp.ClientSession()
            
            # Test connection
            async with self.session.get(f"{self.base_url}/v5/market/time") as response:
                if response.status == 200:
                    self.is_connected = True
                    logger.info("Connected to Bybit API")
                    return True
                else:
                    logger.error(f"Failed to connect to Bybit: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error connecting to Bybit: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Bybit API"""
        if self.session:
            await self.session.close()
            self.session = None
        self.is_connected = False
        logger.info("Disconnected from Bybit API")
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Dict = None,
        signed: bool = False
    ) -> Dict:
        """Make API request"""
        if not self.session:
            raise RuntimeError("Not connected to Bybit API")
        
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        
        headers = {}
        if signed:
            signature, timestamp = self._generate_signature(params)
            headers = self._get_headers(signature, timestamp)
        
        async with self.session.request(method, url, headers=headers, params=params) as response:
            data = await response.json()
            
            if response.status != 200 or data.get('retCode') != 0:
                logger.error(f"Bybit API error: {data}")
                raise Exception(f"API error: {data.get('retMsg', 'Unknown error')}")
            
            return data.get('result', {})
    
    async def get_balance(self, asset: Optional[str] = None) -> List[Balance]:
        """Get account balance"""
        data = await self._make_request(
            "GET",
            "/v5/account/wallet-balance",
            params={"accountType": "UNIFIED"},
            signed=True
        )
        
        balances = []
        for coin in data.get('list', [{}])[0].get('coin', []):
            wallet_balance = float(coin.get('walletBalance', 0))
            locked = float(coin.get('locked', 0))
            
            if wallet_balance > 0:
                if asset is None or coin['coin'] == asset:
                    balances.append(Balance(
                        asset=coin['coin'],
                        free=wallet_balance - locked,
                        locked=locked
                    ))
        
        return balances
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current price ticker"""
        data = await self._make_request(
            "GET",
            "/v5/market/tickers",
            params={"category": "spot", "symbol": symbol}
        )
        
        ticker_data = data.get('list', [{}])[0]
        
        return Ticker(
            symbol=symbol,
            price=float(ticker_data.get('lastPrice', 0)),
            bid=float(ticker_data.get('bid1Price', 0)),
            ask=float(ticker_data.get('ask1Price', 0)),
            volume_24h=float(ticker_data.get('volume24h', 0)),
            change_24h=float(ticker_data.get('price24hPcnt', 0)) * float(ticker_data.get('lastPrice', 0)),
            change_percent_24h=float(ticker_data.get('price24hPcnt', 0)) * 100,
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
            "category": "spot",
            "symbol": symbol,
            "interval": self.TIMEFRAME_MAP.get(timeframe, "60"),
            "limit": min(limit, 200)
        }
        
        if start_time:
            params['start'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['end'] = int(end_time.timestamp() * 1000)
        
        data = await self._make_request("GET", "/v5/market/kline", params=params)
        
        candles = []
        for item in data.get('list', []):
            candles.append(Candle(
                timestamp=datetime.fromtimestamp(int(item[0]) / 1000),
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
            "category": "spot",
            "symbol": symbol,
            "side": side.value.capitalize(),
            "orderType": order_type.value.capitalize(),
            "qty": str(quantity)
        }
        
        if price:
            params['price'] = str(price)
        
        data = await self._make_request(
            "POST",
            "/v5/order/create",
            params=params,
            signed=True
        )
        
        return Order(
            id=data.get('orderId', ''),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=OrderStatus.OPEN
        )
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        try:
            await self._make_request(
                "POST",
                "/v5/order/cancel",
                params={
                    "category": "spot",
                    "symbol": symbol,
                    "orderId": order_id
                },
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
                "/v5/order/realtime",
                params={
                    "category": "spot",
                    "symbol": symbol,
                    "orderId": order_id
                },
                signed=True
            )
            
            orders = data.get('list', [])
            if orders:
                return self._parse_order(orders[0])
            return None
            
        except Exception as e:
            logger.error(f"Error getting order: {e}")
            return None
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get open orders"""
        params = {"category": "spot", "openOnly": 0}
        if symbol:
            params['symbol'] = symbol
        
        data = await self._make_request(
            "GET",
            "/v5/order/realtime",
            params=params,
            signed=True
        )
        
        return [self._parse_order(item) for item in data.get('list', [])]
    
    async def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Order]:
        """Get order history"""
        params = {"category": "spot", "limit": limit}
        if symbol:
            params['symbol'] = symbol
        
        data = await self._make_request(
            "GET",
            "/v5/order/history",
            params=params,
            signed=True
        )
        
        return [self._parse_order(item) for item in data.get('list', [])]
    
    def _parse_order(self, data: Dict) -> Order:
        """Parse order data"""
        status_map = {
            "Created": OrderStatus.PENDING,
            "New": OrderStatus.OPEN,
            "Rejected": OrderStatus.REJECTED,
            "PartiallyFilled": OrderStatus.PARTIALLY_FILLED,
            "PartiallyFilledCanceled": OrderStatus.CANCELLED,
            "Filled": OrderStatus.FILLED,
            "Cancelled": OrderStatus.CANCELLED,
            "PendingCancel": OrderStatus.PENDING
        }
        
        return Order(
            id=data.get('orderId', ''),
            symbol=data.get('symbol', ''),
            side=OrderSide.BUY if data.get('side') == 'Buy' else OrderSide.SELL,
            order_type=OrderType.MARKET if data.get('orderType') == 'Market' else OrderType.LIMIT,
            quantity=float(data.get('qty', 0)),
            price=float(data.get('price', 0)) if data.get('price') else None,
            status=status_map.get(data.get('orderStatus'), OrderStatus.PENDING),
            filled_quantity=float(data.get('cumExecQty', 0)),
            filled_price=float(data.get('cumExecValue', 0)) / float(data.get('cumExecQty', 1)) if float(data.get('cumExecQty', 0)) > 0 else None
        )
