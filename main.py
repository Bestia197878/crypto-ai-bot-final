"""
Crypto Trading AI - Main Application

A comprehensive AI-powered cryptocurrency trading system with:
- Multiple AI agents (DQN, Transformer, LSTM, Ensemble, Self-Learning)
- Multi-exchange support (Binance, Bybit, KuCoin)
- Advanced risk management
- Real-time WebSocket streaming
- Backtesting engine
- Web dashboard and mobile app
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Optional
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    app_config, exchange_config, trading_config,
    ai_config, risk_config
)
from utils.logger import setup_logger
from utils.database import DatabaseManager
from utils.audit import AuditLogger

from agents import (
    SuperDQNAgent,
    SuperTransformerAgent,
    LSTMAgent,
    SuperEnsembleAgent,
    SuperSelfLearningAgent
)

from exchanges import BinanceExchange, BybitExchange, KuCoinExchange

from trading import TradingEngine, Portfolio, SuperRiskManager

from websocket import StreamManager, MultiExchangeStream

from backtest import BacktestEngine


class CryptoTradingAI:
    """Main application class"""
    
    def __init__(self):
        self.logger = setup_logger()
        self.db = DatabaseManager()
        self.audit = AuditLogger()
        
        # Components
        self.agent: Optional[object] = None
        self.exchange: Optional[object] = None
        self.trading_engine: Optional[TradingEngine] = None
        self.stream_manager: Optional[StreamManager] = None
        
        self.logger.info("Crypto Trading AI initialized")
    
    def create_agent(self, agent_type: str = "ensemble"):
        """Create AI agent"""
        agent_map = {
            "dqn": SuperDQNAgent,
            "transformer": SuperTransformerAgent,
            "lstm": LSTMAgent,
            "ensemble": SuperEnsembleAgent,
            "self_learning": SuperSelfLearningAgent
        }
        
        agent_class = agent_map.get(agent_type.lower(), SuperEnsembleAgent)
        
        if agent_type.lower() == "dqn":
            self.agent = agent_class(
                state_size=64,
                action_size=3,
                model_path=ai_config.dqn_model_path
            )
        elif agent_type.lower() == "transformer":
            self.agent = agent_class(
                state_size=64,
                action_size=3,
                model_path=ai_config.transformer_model_path
            )
        elif agent_type.lower() == "lstm":
            self.agent = agent_class(
                state_size=64,
                action_size=3,
                model_path=ai_config.lstm_model_path
            )
        elif agent_type.lower() == "ensemble":
            self.agent = agent_class(
                state_size=64,
                action_size=3,
                model_path=ai_config.ensemble_model_path
            )
        else:
            self.agent = agent_class(
                state_size=64,
                action_size=3,
                model_path=ai_config.ensemble_model_path
            )
        
        self.logger.info(f"Created {agent_type} agent")
        return self.agent
    
    def create_exchange(self, exchange_name: str = "binance"):
        """Create exchange connection"""
        exchange_map = {
            "binance": (BinanceExchange, exchange_config.binance_api_key, 
                       exchange_config.binance_secret_key, exchange_config.binance_testnet),
            "bybit": (BybitExchange, exchange_config.bybit_api_key,
                     exchange_config.bybit_secret_key, exchange_config.bybit_testnet),
            "kucoin": (KuCoinExchange, exchange_config.kucoin_api_key,
                      exchange_config.kucoin_secret_key, exchange_config.kucoin_sandbox)
        }
        
        exchange_info = exchange_map.get(exchange_name.lower())
        
        if exchange_info:
            exchange_class, api_key, secret_key, testnet = exchange_info
            
            if exchange_name.lower() == "kucoin":
                self.exchange = exchange_class(
                    api_key=api_key,
                    secret_key=secret_key,
                    passphrase=exchange_config.kucoin_passphrase,
                    sandbox=testnet
                )
            else:
                self.exchange = exchange_class(
                    api_key=api_key,
                    secret_key=secret_key,
                    testnet=testnet
                )
            
            self.logger.info(f"Created {exchange_name} exchange connection")
        else:
            raise ValueError(f"Unknown exchange: {exchange_name}")
        
        return self.exchange
    
    async def start_trading(
        self,
        symbol: str = None,
        timeframe: str = None,
        agent_type: str = "ensemble",
        exchange_name: str = "binance"
    ):
        """Start trading"""
        symbol = symbol or trading_config.default_symbol
        timeframe = timeframe or trading_config.default_timeframe
        
        self.logger.info(f"Starting trading for {symbol} on {exchange_name}")
        
        # Create components
        if not self.agent:
            self.create_agent(agent_type)
        
        if not self.exchange:
            self.create_exchange(exchange_name)
        
        # Load model if exists
        self.agent.load_model()
        
        # Create portfolio and risk manager
        portfolio = Portfolio(trading_config.initial_balance)
        risk_manager = SuperRiskManager(
            max_portfolio_risk=risk_config.max_daily_loss,
            max_position_risk=risk_config.risk_per_trade
        )
        
        # Create trading engine
        self.trading_engine = TradingEngine(
            exchange=self.exchange,
            agent=self.agent,
            portfolio=portfolio,
            risk_manager=risk_manager,
            symbol=symbol,
            timeframe=timeframe
        )
        
        # Set callbacks
        self.trading_engine.on_trade_opened = self._on_trade_opened
        self.trading_engine.on_trade_closed = self._on_trade_closed
        self.trading_engine.on_error = self._on_error
        
        # Start trading
        await self.trading_engine.start()
    
    async def stop_trading(self):
        """Stop trading"""
        if self.trading_engine:
            await self.trading_engine.stop()
            self.logger.info("Trading stopped")
    
    def _on_trade_opened(self, trade):
        """Handle trade opened event"""
        self.logger.info(f"Trade opened: {trade}")
        self.db.save_trade(trade.__dict__)
        self.audit.log_trade(
            user_id="system",
            trade_action="OPEN",
            symbol=trade.symbol,
            quantity=trade.quantity,
            price=trade.entry_price
        )
    
    def _on_trade_closed(self, trade):
        """Handle trade closed event"""
        self.logger.info(f"Trade closed: {trade}")
        self.db.save_trade(trade.__dict__)
        self.audit.log_trade(
            user_id="system",
            trade_action="CLOSE",
            symbol=trade.symbol,
            quantity=trade.quantity,
            price=trade.exit_price or 0,
            pnl=trade.pnl
        )
    
    def _on_error(self, error):
        """Handle error event"""
        self.logger.error(f"Trading error: {error}")
    
    async def run_backtest(
        self,
        data_path: str,
        agent_type: str = "ensemble",
        initial_balance: float = 10000.0
    ):
        """Run backtest"""
        import pandas as pd
        
        self.logger.info(f"Starting backtest with {agent_type} agent")
        
        # Load data
        data = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
        
        # Create agent
        agent = self.create_agent(agent_type)
        agent.load_model()
        
        # Create backtest engine
        backtest = BacktestEngine(
            agent=agent,
            initial_balance=initial_balance
        )
        
        # Run backtest
        result = backtest.run(data)
        
        # Print report
        print(backtest.get_report(result))
        
        return result
    
    def train_agent(
        self,
        data_path: str,
        agent_type: str = "dqn",
        episodes: int = 1000
    ):
        """Train agent"""
        import pandas as pd
        from training import train_super_dqn, train_super_transformer, train_super_ensemble
        
        self.logger.info(f"Training {agent_type} agent for {episodes} episodes")
        
        # Load data
        data = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
        
        # Create agent
        agent = self.create_agent(agent_type)
        
        # Train
        if agent_type == "dqn":
            history = train_super_dqn(agent, data, episodes)
        elif agent_type == "transformer":
            history = train_super_transformer(agent, data, episodes)
        elif agent_type == "ensemble":
            history = train_super_ensemble(agent, data, episodes)
        else:
            raise ValueError(f"Training not implemented for {agent_type}")
        
        # Save model
        agent.save_model()
        
        self.logger.info("Training completed")
        return history
    
    def start_dashboard(self):
        """Start web dashboard using streamlit run"""
        import subprocess
        import sys
        
        # Get the absolute path to dashboard.py
        dashboard_path = Path(__file__).parent / "gui" / "dashboard.py"
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_path),
            "--server.port=8501",
            "--server.address=localhost",
            "--server.headless=true"
        ])
    
    def start_mobile_app(self):
        """Start mobile app"""
        from mobile.app import run_mobile_app
        run_mobile_app()
    
    def get_status(self) -> dict:
        """Get system status"""
        return {
            "agent": self.agent.name if self.agent else None,
            "exchange": self.exchange.name if self.exchange else None,
            "trading": self.trading_engine.is_running if self.trading_engine else False,
            "database": "connected",
            "version": app_config.app_version
        }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Crypto Trading AI')
    parser.add_argument(
        'command',
        choices=['trade', 'backtest', 'train', 'dashboard', 'mobile', 'status'],
        help='Command to run'
    )
    parser.add_argument('--agent', default='ensemble', help='Agent type')
    parser.add_argument('--exchange', default='binance', help='Exchange')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--timeframe', default='1h', help='Timeframe')
    parser.add_argument('--data', help='Data file path')
    parser.add_argument('--episodes', type=int, default=1000, help='Training episodes')
    parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
    
    args = parser.parse_args()
    
    app = CryptoTradingAI()
    
    if args.command == 'trade':
        asyncio.run(app.start_trading(
            symbol=args.symbol,
            timeframe=args.timeframe,
            agent_type=args.agent,
            exchange_name=args.exchange
        ))
    
    elif args.command == 'backtest':
        if not args.data:
            print("Error: --data required for backtest")
            sys.exit(1)
        
        result = asyncio.run(app.run_backtest(
            data_path=args.data,
            agent_type=args.agent,
            initial_balance=args.balance
        ))
    
    elif args.command == 'train':
        if not args.data:
            print("Error: --data required for training")
            sys.exit(1)
        
        app.train_agent(
            data_path=args.data,
            agent_type=args.agent,
            episodes=args.episodes
        )
    
    elif args.command == 'dashboard':
        app.start_dashboard()
    
    elif args.command == 'mobile':
        app.start_mobile_app()
    
    elif args.command == 'status':
        status = app.get_status()
        print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
