# Crypto Trading AI

A comprehensive AI-powered cryptocurrency trading system with multiple machine learning agents, multi-exchange support, and advanced risk management.

## Features

### AI Agents
- **Super DQN Agent** - Deep Q-Network with dueling architecture and prioritized experience replay
- **Super Transformer Agent** - Transformer-based model for sequential market analysis
- **LSTM Agent** - LSTM with attention mechanism for time series prediction
- **Super Ensemble Agent** - Combines multiple agents with meta-learning
- **Super Self-Learning Agent** - Adapts to market regimes and continuously learns

### Exchange Support
- **Binance** - Full API integration with testnet support
- **Bybit** - Spot and derivatives trading
- **KuCoin** - Complete trading functionality

### Risk Management
- Value at Risk (VaR) calculation
- Position sizing based on risk
- Stop-loss and take-profit automation
- Drawdown monitoring
- Portfolio risk limits

### Additional Features
- Real-time WebSocket data streaming
- Backtesting engine
- Web dashboard (Streamlit)
- Mobile app (Kivy)
- Comprehensive audit logging

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/crypto-trading-ai.git
cd crypto-trading-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

## Quick Start

### 1. Configure API Keys

Edit `.env` file with your exchange API keys:

```env
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
BINANCE_TESTNET=true
```

### 2. Run Tests

```bash
python -m pytest tests/
```

### 3. Start Trading

```bash
# Start with default settings
python main.py trade

# Start with custom settings
python main.py trade --agent ensemble --exchange binance --symbol BTCUSDT --timeframe 1h
```

### 4. Run Backtest

```bash
python main.py backtest --data data/btc_usdt_1h.csv --agent ensemble
```

### 5. Train Agent

```bash
python main.py train --data data/btc_usdt_1h.csv --agent dqn --episodes 1000
```

### 6. Launch Dashboard

```bash
python main.py dashboard
```

## Usage

### Command Line Interface

```bash
# Start trading
python main.py trade [options]

# Run backtest
python main.py backtest --data <path> [options]

# Train agent
python main.py train --data <path> [options]

# Launch web dashboard
python main.py dashboard

# Launch mobile app
python main.py mobile

# Get system status
python main.py status
```

### Options

- `--agent` - Agent type (dqn, transformer, lstm, ensemble, self_learning)
- `--exchange` - Exchange (binance, bybit, kucoin)
- `--symbol` - Trading pair (e.g., BTCUSDT)
- `--timeframe` - Candle timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)
- `--data` - Path to historical data CSV
- `--episodes` - Number of training episodes
- `--balance` - Initial balance for backtesting

## Project Structure

```
crypto-trading-ai/
├── agents/                 # AI trading agents
│   ├── base_agent.py
│   ├── super_dqn_agent.py
│   ├── super_transformer_agent.py
│   ├── lstm_agent.py
│   ├── super_ensemble_agent.py
│   └── super_self_learning_agent.py
├── exchanges/             # Exchange integrations
│   ├── binance.py
│   ├── bybit.py
│   └── kucoin.py
├── trading/               # Trading engine & risk management
│   ├── engine.py
│   ├── portfolio.py
│   └── super_risk_manager.py
├── websocket/             # Real-time data streaming
│   ├── stream_manager.py
│   └── multi_exchange_stream.py
├── backtest/              # Backtesting engine
│   └── backtest_engine.py
├── training/              # Training scripts
│   ├── train_super_dqn.py
│   ├── train_super_transformer.py
│   └── train_super_ensemble.py
├── utils/                 # Utilities
│   ├── database.py
│   ├── indicators.py
│   ├── logger.py
│   └── audit.py
├── gui/                   # Web dashboard
│   └── dashboard.py
├── mobile/                # Mobile app
│   └── app.py
├── tests/                 # Unit tests
│   └── test_engine.py
├── main.py               # Main application
├── config.py             # Configuration
├── requirements.txt      # Dependencies
└── .env                  # Environment variables
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BINANCE_API_KEY` | Binance API key | - |
| `BINANCE_SECRET_KEY` | Binance secret key | - |
| `BYBIT_API_KEY` | Bybit API key | - |
| `KUCOIN_API_KEY` | KuCoin API key | - |
| `DEFAULT_SYMBOL` | Default trading pair | BTCUSDT |
| `DEFAULT_TIMEFRAME` | Default timeframe | 1h |
| `INITIAL_BALANCE` | Starting balance | 10000.0 |
| `MAX_POSITION_SIZE` | Max position size | 0.1 |
| `STOP_LOSS_PERCENT` | Default stop loss % | 2.0 |
| `TAKE_PROFIT_PERCENT` | Default take profit % | 4.0 |

### Risk Settings

Edit `config.py` or set environment variables:

```python
risk_config = RiskConfig(
    max_daily_loss=5.0,      # Max 5% daily loss
    max_positions=3,          # Max 3 open positions
    risk_per_trade=1.0        # Risk 1% per trade
)
```

## AI Agents

### Super DQN Agent

Deep Q-Network with advanced features:
- Dueling architecture
- Double DQN learning
- Prioritized experience replay
- Adaptive epsilon-greedy exploration

```python
from agents import SuperDQNAgent

agent = SuperDQNAgent(
    state_size=64,
    action_size=3,
    learning_rate=0.001,
    gamma=0.99
)
```

### Super Transformer Agent

Transformer-based agent for sequential analysis:
- Multi-head attention
- Positional encoding
- Attention pooling

```python
from agents import SuperTransformerAgent

agent = SuperTransformerAgent(
    state_size=64,
    action_size=3,
    sequence_length=60,
    d_model=128
)
```

### Super Ensemble Agent

Combines multiple agents with meta-learning:
- Dynamic weight adjustment
- Performance tracking
- Adaptive strategy selection

```python
from agents import SuperEnsembleAgent

agent = SuperEnsembleAgent(
    state_size=64,
    action_size=3
)
```

## Backtesting

```python
from backtest import BacktestEngine
from agents import SuperDQNAgent
import pandas as pd

# Load data
data = pd.read_csv('btc_usdt_1h.csv', parse_dates=['timestamp'], index_col='timestamp')

# Create agent
agent = SuperDQNAgent()
agent.load_model()

# Run backtest
backtest = BacktestEngine(agent=agent, initial_balance=10000.0)
result = backtest.run(data, symbol='BTCUSDT')

# Print report
print(backtest.get_report(result))
```

## Risk Management

```python
from trading import SuperRiskManager

risk_manager = SuperRiskManager(
    max_portfolio_risk=5.0,
    max_position_risk=2.0,
    max_daily_loss=5.0,
    max_drawdown=20.0
)

# Calculate position size
size, risk = risk_manager.calculate_position_size(
    account_value=10000.0,
    entry_price=50000.0,
    stop_loss=49000.0,
    risk_percent=2.0
)

# Check if trade is allowed
allowed, reason = risk_manager.check_trade_allowed(
    symbol='BTCUSDT',
    action='buy',
    position_size=1000.0,
    account_value=10000.0
)
```

## WebSocket Streaming

```python
from websocket import MultiExchangeStream, BinanceStreamManager

# Create streams
binance_stream = BinanceStreamManager(symbols=['BTCUSDT', 'ETHUSDT'])

# Create multi-exchange stream
multi_stream = MultiExchangeStream()
multi_stream.add_stream('binance', binance_stream)

# Add handler
def on_ticker(aggregated_ticker):
    print(f"Best bid: {aggregated_ticker.best_bid} on {aggregated_ticker.bid_exchange}")
    print(f"Best ask: {aggregated_ticker.best_ask} on {aggregated_ticker.ask_exchange}")

multi_stream.add_aggregated_handler(on_ticker)

# Start streaming
import asyncio
asyncio.run(multi_stream.start())
```

## Database

```python
from utils import DatabaseManager

db = DatabaseManager('data/trading.db')

# Save trade
db.save_trade({
    'id': '12345',
    'symbol': 'BTCUSDT',
    'side': 'buy',
    'entry_price': 50000.0,
    'quantity': 0.1,
    'status': 'open'
})

# Get trades
trades = db.get_trades(symbol='BTCUSDT', limit=100)

# Save market data
db.save_market_data('BTCUSDT', '1h', ohlcv_dataframe)
```

## Technical Indicators

```python
from utils import TechnicalIndicators
import pandas as pd

# Load data
df = pd.read_csv('btc_usdt_1h.csv')

# Calculate all indicators
df = TechnicalIndicators.calculate_all(df)

# Individual indicators
sma_20 = TechnicalIndicators.sma(df['close'], 20)
rsi = TechnicalIndicators.rsi(df['close'], 14)
macd, signal, hist = TechnicalIndicators.macd(df['close'])
upper, middle, lower = TechnicalIndicators.bollinger_bands(df['close'])
atr = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_engine.py::TestSuperDQNAgent

# Run with coverage
pytest --cov=agents --cov=trading tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Disclaimer

**Trading cryptocurrencies carries significant risk. This software is for educational purposes only. Always test thoroughly with paper trading before using real funds. The authors are not responsible for any financial losses incurred through the use of this software.**

## Support

For support, email support@cryptotradingai.com or join our Discord community.

## Roadmap

- [ ] Add more exchanges (Coinbase, Kraken)
- [ ] Implement options trading
- [ ] Add social sentiment analysis
- [ ] Machine learning model optimization
- [ ] Mobile app improvements
- [ ] Cloud deployment options

---

**Happy Trading! 📈🤖**
