"""
Database Manager - Handles all database operations
"""
import sqlite3
import json
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from loguru import logger
import pandas as pd


class DatabaseManager:
    """
    Manages SQLite database for trading data
    """
    
    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"DatabaseManager initialized: {db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Get database connection context"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    status TEXT NOT NULL,
                    entry_time TIMESTAMP,
                    exit_time TIMESTAMP,
                    stop_loss REAL,
                    take_profit REAL,
                    pnl REAL,
                    pnl_percent REAL,
                    commission REAL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Market data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            """)
            
            # Portfolio snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_value REAL,
                    cash REAL,
                    invested_value REAL,
                    total_return REAL,
                    total_return_percent REAL,
                    unrealized_pnl REAL,
                    num_positions INTEGER
                )
            """)
            
            # Agent predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    agent_name TEXT,
                    symbol TEXT,
                    action TEXT,
                    confidence REAL,
                    price REAL,
                    quantity REAL,
                    metadata TEXT
                )
            """)
            
            # Risk metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    current_drawdown REAL,
                    max_drawdown REAL,
                    var_95 REAL,
                    var_99 REAL,
                    sharpe_ratio REAL,
                    volatility REAL
                )
            """)
            
            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    win_rate REAL,
                    avg_profit REAL,
                    avg_loss REAL,
                    profit_factor REAL,
                    total_pnl REAL
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol ON market_data(symbol, timeframe)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)")
            
            conn.commit()
            logger.info("Database tables initialized")
    
    def save_trade(self, trade_data: Dict[str, Any]) -> bool:
        """Save trade to database"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO trades (
                        trade_id, symbol, side, entry_price, exit_price,
                        quantity, status, entry_time, exit_time,
                        stop_loss, take_profit, pnl, pnl_percent,
                        commission, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_data.get('id'),
                    trade_data.get('symbol'),
                    trade_data.get('side'),
                    trade_data.get('entry_price'),
                    trade_data.get('exit_price'),
                    trade_data.get('quantity'),
                    trade_data.get('status'),
                    trade_data.get('entry_time'),
                    trade_data.get('exit_time'),
                    trade_data.get('stop_loss'),
                    trade_data.get('take_profit'),
                    trade_data.get('pnl'),
                    trade_data.get('pnl_percent'),
                    trade_data.get('commission'),
                    json.dumps(trade_data.get('metadata', {}))
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            return False
    
    def get_trades(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get trades from database"""
        with self._get_connection() as conn:
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            query += " ORDER BY entry_time DESC LIMIT ?"
            params.append(limit)
            
            df = pd.read_sql_query(query, conn, params=params)
            return df.to_dict('records')
    
    def save_market_data(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame
    ) -> bool:
        """Save market data (OHLCV)"""
        try:
            with self._get_connection() as conn:
                for idx, row in data.iterrows():
                    conn.execute("""
                        INSERT OR REPLACE INTO market_data
                        (symbol, timeframe, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, timeframe, idx,
                        row['open'], row['high'], row['low'],
                        row['close'], row['volume']
                    ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error saving market data: {e}")
            return False
    
    def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get market data from database"""
        with self._get_connection() as conn:
            query = """
                SELECT * FROM market_data
                WHERE symbol = ? AND timeframe = ?
            """
            params = [symbol, timeframe]
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', format='mixed')
                df.set_index('timestamp', inplace=True)
            
            return df
    
    def save_portfolio_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        """Save portfolio snapshot"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO portfolio_snapshots
                    (total_value, cash, invested_value, total_return,
                     total_return_percent, unrealized_pnl, num_positions)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    snapshot.get('total_value'),
                    snapshot.get('cash'),
                    snapshot.get('invested_value'),
                    snapshot.get('total_return'),
                    snapshot.get('total_return_percent'),
                    snapshot.get('unrealized_pnl'),
                    snapshot.get('num_positions')
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error saving portfolio snapshot: {e}")
            return False
    
    def save_prediction(self, prediction: Dict[str, Any]) -> bool:
        """Save agent prediction"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO predictions
                    (agent_name, symbol, action, confidence, price, quantity, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction.get('agent_name'),
                    prediction.get('symbol'),
                    prediction.get('action'),
                    prediction.get('confidence'),
                    prediction.get('price'),
                    prediction.get('quantity'),
                    json.dumps(prediction.get('metadata', {}))
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            return False
    
    def save_risk_metrics(self, metrics: Dict[str, float]) -> bool:
        """Save risk metrics"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO risk_metrics
                    (current_drawdown, max_drawdown, var_95, var_99, sharpe_ratio, volatility)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    metrics.get('current_drawdown'),
                    metrics.get('max_drawdown'),
                    metrics.get('var_95'),
                    metrics.get('var_99'),
                    metrics.get('sharpe_ratio'),
                    metrics.get('volatility')
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error saving risk metrics: {e}")
            return False
    
    def save_performance_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Save performance metrics"""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO performance_metrics
                    (total_trades, winning_trades, losing_trades, win_rate,
                     avg_profit, avg_loss, profit_factor, total_pnl)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.get('total_trades'),
                    metrics.get('winning_trades'),
                    metrics.get('losing_trades'),
                    metrics.get('win_rate'),
                    metrics.get('avg_profit'),
                    metrics.get('avg_loss'),
                    metrics.get('profit_factor'),
                    metrics.get('total_pnl')
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")
            return False
    
    def get_performance_history(
        self,
        days: int = 30
    ) -> pd.DataFrame:
        """Get performance history"""
        with self._get_connection() as conn:
            query = """
                SELECT * FROM performance_metrics
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp
            """.format(days)
            
            return pd.read_sql_query(query, conn)
    
    def cleanup_old_data(self, days: int = 90):
        """Clean up old data"""
        with self._get_connection() as conn:
            tables = ['market_data', 'predictions', 'risk_metrics', 'portfolio_snapshots']
            
            for table in tables:
                conn.execute(f"""
                    DELETE FROM {table}
                    WHERE timestamp < datetime('now', '-{days} days')
                """)
            
            conn.commit()
            logger.info(f"Cleaned up data older than {days} days")
