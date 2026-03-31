"""
Multi-Exchange Stream - Aggregates data from multiple exchanges
"""
import asyncio
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from loguru import logger

from .stream_manager import StreamManager, StreamMessage


@dataclass
class AggregatedTicker:
    """Aggregated ticker from multiple exchanges"""
    symbol: str
    exchanges: Dict[str, Dict]
    best_bid: float
    best_ask: float
    bid_exchange: str
    ask_exchange: str
    spread: float
    timestamp: datetime


class MultiExchangeStream:
    """
    Manages multiple exchange streams and aggregates data
    """
    
    def __init__(self):
        self.streams: Dict[str, StreamManager] = {}
        self.ticker_data: Dict[str, Dict[str, Dict]] = defaultdict(dict)
        self.aggregated_handlers: List[Callable] = []
        
        # Statistics
        self.messages_by_exchange: Dict[str, int] = defaultdict(int)
        self.last_update = datetime.now()
        
        logger.info("MultiExchangeStream initialized")
    
    def add_stream(self, name: str, stream: StreamManager):
        """Add a stream manager"""
        self.streams[name] = stream
        
        # Add handler for ticker data
        stream.add_handler('ticker', lambda msg, name=name: self._on_ticker(msg, name))
        
        logger.info(f"Added stream: {name}")
    
    def remove_stream(self, name: str):
        """Remove a stream manager"""
        if name in self.streams:
            del self.streams[name]
            logger.info(f"Removed stream: {name}")
    
    def add_aggregated_handler(self, handler: Callable):
        """Add handler for aggregated data"""
        self.aggregated_handlers.append(handler)
    
    async def start(self):
        """Start all streams"""
        logger.info("Starting all streams...")
        
        tasks = []
        for name, stream in self.streams.items():
            task = asyncio.create_task(stream.start())
            tasks.append(task)
            logger.info(f"Started stream: {name}")
        
        # Start aggregation task
        agg_task = asyncio.create_task(self._aggregation_loop())
        tasks.append(agg_task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop(self):
        """Stop all streams"""
        logger.info("Stopping all streams...")
        
        for name, stream in self.streams.items():
            await stream.disconnect()
            logger.info(f"Stopped stream: {name}")
    
    def _on_ticker(self, message: StreamMessage, exchange: str):
        """Handle ticker message"""
        self.messages_by_exchange[exchange] += 1
        
        # Store ticker data
        self.ticker_data[message.symbol][exchange] = {
            "data": message.data,
            "timestamp": message.timestamp
        }
        
        self.last_update = datetime.now()
    
    async def _aggregation_loop(self):
        """Aggregation loop"""
        while True:
            try:
                await asyncio.sleep(1)
                
                # Aggregate tickers
                for symbol, exchange_data in list(self.ticker_data.items()):
                    if len(exchange_data) >= 2:  # Need at least 2 exchanges
                        aggregated = self._aggregate_ticker(symbol, exchange_data)
                        
                        if aggregated:
                            # Dispatch to handlers
                            for handler in self.aggregated_handlers:
                                try:
                                    if asyncio.iscoroutinefunction(handler):
                                        await handler(aggregated)
                                    else:
                                        handler(aggregated)
                                except Exception as e:
                                    logger.error(f"Handler error: {e}")
                
            except Exception as e:
                logger.error(f"Aggregation error: {e}")
    
    def _aggregate_ticker(
        self,
        symbol: str,
        exchange_data: Dict[str, Dict]
    ) -> Optional[AggregatedTicker]:
        """Aggregate ticker data from multiple exchanges"""
        try:
            best_bid = 0
            best_ask = float('inf')
            bid_exchange = ""
            ask_exchange = ""
            
            exchanges = {}
            
            for exchange, data in exchange_data.items():
                ticker = data['data']
                
                # Extract bid/ask based on exchange format
                bid = self._extract_price(ticker, 'bid')
                ask = self._extract_price(ticker, 'ask')
                
                if bid and ask:
                    exchanges[exchange] = {
                        "bid": bid,
                        "ask": ask,
                        "timestamp": data['timestamp']
                    }
                    
                    if bid > best_bid:
                        best_bid = bid
                        bid_exchange = exchange
                    
                    if ask < best_ask:
                        best_ask = ask
                        ask_exchange = exchange
            
            if best_bid > 0 and best_ask < float('inf'):
                return AggregatedTicker(
                    symbol=symbol,
                    exchanges=exchanges,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    bid_exchange=bid_exchange,
                    ask_exchange=ask_exchange,
                    spread=(best_ask - best_bid) / best_bid * 100,
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error aggregating ticker: {e}")
            return None
    
    def _extract_price(self, data: Dict, price_type: str) -> Optional[float]:
        """Extract price from exchange-specific format"""
        try:
            if price_type == 'bid':
                # Try different field names
                for field in ['bidPrice', 'bid', 'bestBid', 'b']:
                    if field in data:
                        return float(data[field])
            else:  # ask
                for field in ['askPrice', 'ask', 'bestAsk', 'a']:
                    if field in data:
                        return float(data[field])
            return None
        except (ValueError, TypeError):
            return None
    
    def get_arbitrage_opportunities(self, min_spread: float = 0.1) -> List[Dict]:
        """Find arbitrage opportunities"""
        opportunities = []
        
        for symbol, exchange_data in self.ticker_data.items():
            if len(exchange_data) >= 2:
                aggregated = self._aggregate_ticker(symbol, exchange_data)
                
                if aggregated and aggregated.spread >= min_spread:
                    opportunities.append({
                        "symbol": symbol,
                        "spread_percent": aggregated.spread,
                        "buy_exchange": aggregated.ask_exchange,
                        "sell_exchange": aggregated.bid_exchange,
                        "buy_price": aggregated.best_ask,
                        "sell_price": aggregated.best_bid,
                        "timestamp": aggregated.timestamp
                    })
        
        return sorted(opportunities, key=lambda x: x['spread_percent'], reverse=True)
    
    def get_stats(self) -> Dict:
        """Get multi-exchange stream statistics"""
        return {
            "active_streams": len(self.streams),
            "stream_names": list(self.streams.keys()),
            "tracked_symbols": list(self.ticker_data.keys()),
            "messages_by_exchange": dict(self.messages_by_exchange),
            "last_update": self.last_update.isoformat()
        }
