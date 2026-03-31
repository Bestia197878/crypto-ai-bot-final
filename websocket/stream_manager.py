"""
WebSocket Stream Manager - Manages real-time data streams
"""
import asyncio
import json
from typing import Dict, List, Optional, Callable, Set
from dataclasses import dataclass
from datetime import datetime
from loguru import logger
import websockets
from websockets.exceptions import ConnectionClosed


@dataclass
class StreamConfig:
    """Stream configuration"""
    url: str
    symbols: List[str]
    channels: List[str]  # 'ticker', 'trade', 'orderbook', 'kline'
    reconnect_interval: int = 5
    ping_interval: int = 30


@dataclass
class StreamMessage:
    """Stream message"""
    channel: str
    symbol: str
    data: Dict
    timestamp: datetime


class StreamManager:
    """
    Manages WebSocket connections for real-time market data
    """
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False
        self.is_running = False
        
        # Message handlers
        self.handlers: Dict[str, List[Callable]] = {
            'ticker': [],
            'trade': [],
            'orderbook': [],
            'kline': []
        }
        
        # Message queue
        self.message_queue: asyncio.Queue = asyncio.Queue()
        
        # Statistics
        self.messages_received = 0
        self.connection_attempts = 0
        self.last_ping = datetime.now()
        
        logger.info(f"StreamManager initialized for {config.url}")
    
    def add_handler(self, channel: str, handler: Callable):
        """Add message handler for a channel"""
        if channel in self.handlers:
            self.handlers[channel].append(handler)
            logger.info(f"Added handler for {channel}")
        else:
            raise ValueError(f"Unknown channel: {channel}")
    
    def remove_handler(self, channel: str, handler: Callable):
        """Remove message handler"""
        if channel in self.handlers and handler in self.handlers[channel]:
            self.handlers[channel].remove(handler)
    
    async def connect(self) -> bool:
        """Connect to WebSocket"""
        try:
            self.connection_attempts += 1
            logger.info(f"Connecting to WebSocket: {self.config.url}")
            
            self.websocket = await websockets.connect(self.config.url)
            self.is_connected = True
            
            # Subscribe to channels
            await self._subscribe()
            
            logger.info("WebSocket connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        self.is_running = False
        
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
        
        self.is_connected = False
        logger.info("WebSocket disconnected")
    
    async def _subscribe(self):
        """Subscribe to channels"""
        # Override in subclasses for exchange-specific subscription
        pass
    
    async def start(self):
        """Start streaming"""
        if self.is_running:
            logger.warning("Stream already running")
            return
        
        self.is_running = True
        
        # Start connection and message handling
        while self.is_running:
            try:
                if not self.is_connected:
                    if not await self.connect():
                        await asyncio.sleep(self.config.reconnect_interval)
                        continue
                
                # Handle messages
                await self._handle_messages()
                
            except ConnectionClosed:
                logger.warning("WebSocket connection closed, reconnecting...")
                self.is_connected = False
                await asyncio.sleep(self.config.reconnect_interval)
                
            except Exception as e:
                logger.error(f"Stream error: {e}")
                await asyncio.sleep(self.config.reconnect_interval)
    
    async def _handle_messages(self):
        """Handle incoming messages"""
        while self.is_running and self.is_connected:
            try:
                # Receive message
                message = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=self.config.ping_interval
                )
                
                self.messages_received += 1
                
                # Parse message
                parsed = self._parse_message(message)
                
                if parsed:
                    # Dispatch to handlers
                    await self._dispatch_message(parsed)
                
                # Send ping if needed
                await self._send_ping()
                
            except asyncio.TimeoutError:
                await self._send_ping()
            except Exception as e:
                logger.error(f"Error handling message: {e}")
    
    def _parse_message(self, message: str) -> Optional[StreamMessage]:
        """Parse incoming message - override in subclasses"""
        try:
            data = json.loads(message)
            
            return StreamMessage(
                channel=data.get('channel', 'unknown'),
                symbol=data.get('symbol', ''),
                data=data,
                timestamp=datetime.now()
            )
            
        except json.JSONDecodeError:
            logger.error(f"Failed to parse message: {message}")
            return None
    
    async def _dispatch_message(self, message: StreamMessage):
        """Dispatch message to handlers"""
        handlers = self.handlers.get(message.channel, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"Handler error: {e}")
    
    async def _send_ping(self):
        """Send ping to keep connection alive"""
        if (datetime.now() - self.last_ping).seconds >= self.config.ping_interval:
            try:
                if self.websocket:
                    await self.websocket.ping()
                    self.last_ping = datetime.now()
            except Exception as e:
                logger.error(f"Ping error: {e}")
    
    async def send_message(self, message: Dict):
        """Send message through WebSocket"""
        if self.websocket and self.is_connected:
            await self.websocket.send(json.dumps(message))
    
    def get_stats(self) -> Dict:
        """Get stream statistics"""
        return {
            "is_connected": self.is_connected,
            "is_running": self.is_running,
            "messages_received": self.messages_received,
            "connection_attempts": self.connection_attempts,
            "subscribed_channels": list(self.handlers.keys())
        }


class BinanceStreamManager(StreamManager):
    """Binance WebSocket stream manager"""
    
    def __init__(self, symbols: List[str], testnet: bool = True):
        base_url = "wss://testnet.binance.vision/ws" if testnet else "wss://stream.binance.com:9443/ws"
        
        config = StreamConfig(
            url=base_url,
            symbols=symbols,
            channels=['ticker', 'trade', 'depth'],
            reconnect_interval=5,
            ping_interval=30
        )
        
        super().__init__(config)
        self.symbol_streams = [f"{s.lower()}@ticker" for s in symbols]
    
    async def _subscribe(self):
        """Subscribe to Binance streams"""
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": self.symbol_streams,
            "id": 1
        }
        await self.send_message(subscribe_msg)
    
    def _parse_message(self, message: str) -> Optional[StreamMessage]:
        """Parse Binance message"""
        try:
            data = json.loads(message)
            
            # Determine channel type
            stream = data.get('stream', '')
            
            if '@ticker' in stream:
                channel = 'ticker'
                symbol = stream.replace('@ticker', '').upper()
            elif '@trade' in stream:
                channel = 'trade'
                symbol = stream.replace('@trade', '').upper()
            elif '@depth' in stream:
                channel = 'orderbook'
                symbol = stream.replace('@depth', '').upper()
            else:
                return None
            
            return StreamMessage(
                channel=channel,
                symbol=symbol,
                data=data.get('data', data),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error parsing Binance message: {e}")
            return None


class BybitStreamManager(StreamManager):
    """Bybit WebSocket stream manager"""
    
    def __init__(self, symbols: List[str], testnet: bool = True):
        base_url = "wss://stream-testnet.bybit.com/v5/public/spot" if testnet else "wss://stream.bybit.com/v5/public/spot"
        
        config = StreamConfig(
            url=base_url,
            symbols=symbols,
            channels=['ticker', 'trade', 'orderbook'],
            reconnect_interval=5,
            ping_interval=30
        )
        
        super().__init__(config)
        self.symbols = symbols
    
    async def _subscribe(self):
        """Subscribe to Bybit streams"""
        subscribe_msg = {
            "op": "subscribe",
            "args": [
                f"tickers.{symbol}" for symbol in self.symbols
            ] + [
                f"publicTrade.{symbol}" for symbol in self.symbols
            ]
        }
        await self.send_message(subscribe_msg)
    
    def _parse_message(self, message: str) -> Optional[StreamMessage]:
        """Parse Bybit message"""
        try:
            data = json.loads(message)
            
            topic = data.get('topic', '')
            
            if 'tickers' in topic:
                channel = 'ticker'
                symbol = topic.replace('tickers.', '')
            elif 'publicTrade' in topic:
                channel = 'trade'
                symbol = topic.replace('publicTrade.', '')
            elif 'orderbook' in topic:
                channel = 'orderbook'
                symbol = topic.replace('orderbook.', '').split('.')[0]
            else:
                return None
            
            return StreamMessage(
                channel=channel,
                symbol=symbol,
                data=data.get('data', data),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error parsing Bybit message: {e}")
            return None
