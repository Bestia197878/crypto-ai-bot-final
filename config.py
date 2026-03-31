"""
Configuration module for Crypto Trading AI - Pydantic V2 Compatible
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List, Optional, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


class AppConfig(BaseSettings):
    """Application configuration"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    app_name: str = Field(default="CryptoTradingAI", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")


class DatabaseConfig(BaseSettings):
    """Database configuration"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    database_url: str = Field(default=f"sqlite:///{DATA_DIR}/trading.db")
    redis_url: str = Field(default="redis://localhost:6379/0")


class ExchangeConfig(BaseSettings):
    """Exchange API configuration"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    # Binance
    binance_api_key: str = Field(default="")
    binance_secret_key: str = Field(default="")
    binance_testnet: bool = Field(default=True)
    
    # Bybit
    bybit_api_key: str = Field(default="")
    bybit_secret_key: str = Field(default="")
    bybit_testnet: bool = Field(default=True)
    
    # KuCoin
    kucoin_api_key: str = Field(default="")
    kucoin_secret_key: str = Field(default="")
    kucoin_passphrase: str = Field(default="")
    kucoin_sandbox: bool = Field(default=True)


# High performance symbols (defined before TradingConfig)
HIGH_PERFORMANCE_SYMBOLS = ["BTCUSDT", "SOLUSDT"]

# Symbol weight multipliers (for position sizing)
SYMBOL_WEIGHTS = {
    "BTCUSDT": 2.0,    # Double position size for BTC
    "SOLUSDT": 1.5,    # 50% larger positions for SOL
    "default": 1.0     # Standard size for others
}


class TradingConfig(BaseSettings):
    """Trading configuration"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    default_exchange: str = Field(default="binance")
    default_symbol: str = Field(default="BTCUSDT")
    default_timeframe: str = Field(default="1h")
    initial_balance: float = Field(default=10000.0)
    max_position_size: float = Field(default=0.1)
    stop_loss_percent: float = Field(default=2.0)
    take_profit_percent: float = Field(default=4.0)
    
    # Symbol preferences based on performance analysis
    enabled_symbols: List[str] = Field(default=HIGH_PERFORMANCE_SYMBOLS)
    symbol_weights: Dict[str, float] = Field(default=SYMBOL_WEIGHTS)


class AIConfig(BaseSettings):
    """AI Model configuration"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    dqn_model_path: str = Field(default=str(MODELS_DIR / "dqn_model.pt"))
    transformer_model_path: str = Field(default=str(MODELS_DIR / "transformer_model.pt"))
    lstm_model_path: str = Field(default=str(MODELS_DIR / "lstm_model.pt"))
    ensemble_model_path: str = Field(default=str(MODELS_DIR / "ensemble_model.pt"))
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 100000
    target_update: int = 1000


class RiskConfig(BaseSettings):
    """Risk management configuration"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    max_daily_loss: float = Field(default=5.0)
    max_positions: int = Field(default=3)
    risk_per_trade: float = Field(default=1.0)


class WebSocketConfig(BaseSettings):
    """WebSocket configuration"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    reconnect_interval: int = Field(default=5)
    ping_interval: int = Field(default=30)


class NotificationConfig(BaseSettings):
    """Notification configuration"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    telegram_bot_token: str = Field(default="")
    telegram_chat_id: str = Field(default="")
    discord_webhook_url: str = Field(default="")


# Initialize configurations
app_config = AppConfig()
database_config = DatabaseConfig()
exchange_config = ExchangeConfig()
trading_config = TradingConfig()
ai_config = AIConfig()
risk_config = RiskConfig()
ws_config = WebSocketConfig()
notification_config = NotificationConfig()

# Ensure HIGH_PERFORMANCE_SYMBOLS is defined for imports
try:
    _ = HIGH_PERFORMANCE_SYMBOLS
except NameError:
    HIGH_PERFORMANCE_SYMBOLS = ["BTCUSDT", "SOLUSDT"]

# Timeframes mapping
TIMEFRAMES = {
    "1m": "1 minute",
    "5m": "5 minutes",
    "15m": "15 minutes",
    "30m": "30 minutes",
    "1h": "1 hour",
    "4h": "4 hours",
    "1d": "1 day",
    "1w": "1 week"
}

# Supported symbols - ETHUSDT removed due to poor performance
SUPPORTED_SYMBOLS = [
    "BTCUSDT", "BNBUSDT", "ADAUSDT", "DOGEUSDT",
    "XRPUSDT", "DOTUSDT", "UNIUSDT", "LTCUSDT", "LINKUSDT",
    "BCHUSDT", "SOLUSDT", "MATICUSDT", "AVAXUSDT"
]

# High performance symbols (prioritized for trading)
HIGH_PERFORMANCE_SYMBOLS = ["BTCUSDT", "SOLUSDT"]

# Symbol weight multipliers (for position sizing)
SYMBOL_WEIGHTS = {
    "BTCUSDT": 2.0,    # Double position size for BTC
    "SOLUSDT": 1.5,    # 50% larger positions for SOL
    "default": 1.0     # Standard size for others
}

# Trading pairs by exchange
EXCHANGE_SYMBOLS = {
    "binance": SUPPORTED_SYMBOLS,
    "bybit": SUPPORTED_SYMBOLS,
    "kucoin": [s.replace("USDT", "-USDT") for s in SUPPORTED_SYMBOLS]
}
