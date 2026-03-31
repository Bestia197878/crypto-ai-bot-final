"""
Unit tests for Crypto Trading AI
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import MarketState, Action
from agents.super_dqn_agent import SuperDQNAgent
from agents.lstm_agent import LSTMAgent
from agents.super_transformer_agent import SuperTransformerAgent
from agents.super_ensemble_agent import SuperEnsembleAgent
from trading.portfolio import Portfolio
from trading.super_risk_manager import SuperRiskManager
from utils.indicators import TechnicalIndicators
from utils.database import DatabaseManager
from utils.audit import AuditLogger


class TestMarketState:
    """Test MarketState class"""
    
    def test_creation(self):
        state = MarketState(
            price=50000.0,
            volume=1000.0,
            timestamp=1234567890,
            indicators={'rsi': 50.0}
        )
        
        assert state.price == 50000.0
        assert state.volume == 1000.0
        assert state.indicators['rsi'] == 50.0
    
    def test_to_vector(self):
        state = MarketState(
            price=50000.0,
            volume=1000.0,
            timestamp=1234567890,
            indicators={'rsi': 50.0, 'atr': 500.0}
        )
        
        vector = state.to_vector()
        assert len(vector) == 4  # price, volume, rsi, atr
        assert vector[0] == 50000.0
        assert vector[1] == 1000.0


class TestSuperDQNAgent:
    """Test SuperDQNAgent"""
    
    @pytest.fixture
    def agent(self):
        return SuperDQNAgent(
            state_size=64,
            action_size=3,
            device='cpu'
        )
    
    def test_initialization(self, agent):
        assert agent.name == "SuperDQNAgent"
        assert agent.state_size == 64
        assert agent.action_size == 3
        assert agent.epsilon > 0
    
    def test_predict(self, agent):
        # Create state with 64 features (state_size=64)
        # price + volume + 62 indicators
        indicators = {f'ind_{i}': 50.0 for i in range(62)}
        state = MarketState(
            price=50000.0,
            volume=1000.0,
            timestamp=1234567890,
            indicators=indicators
        )
        
        action = agent.predict(state)
        
        assert isinstance(action, Action)
        assert action.action_type in ['buy', 'sell', 'hold']
        assert 0 <= action.confidence <= 1
        assert action.price == 50000.0
    
    def test_store_experience(self, agent):
        state = MarketState(
            price=50000.0,
            volume=1000.0,
            timestamp=1234567890,
            indicators={}
        )
        
        next_state = MarketState(
            price=51000.0,
            volume=1100.0,
            timestamp=1234567891,
            indicators={}
        )
        
        agent.store_experience(state, 0, 100.0, next_state, False)
        
        assert len(agent.replay_buffer) == 1


class TestLSTMAgent:
    """Test LSTMAgent"""
    
    @pytest.fixture
    def agent(self):
        return LSTMAgent(
            state_size=64,
            action_size=3,
            device='cpu'
        )
    
    def test_initialization(self, agent):
        assert agent.name == "LSTMAgent"
        assert agent.sequence_length == 50
    
    def test_predict_with_sequence(self, agent):
        # Add multiple states to build sequence with 64 features
        for i in range(60):
            indicators = {f'ind_{j}': 50.0 + j for j in range(62)}
            state = MarketState(
                price=50000.0 + i * 100,
                volume=1000.0,
                timestamp=1234567890 + i,
                indicators=indicators
            )
            action = agent.predict(state)
        
        assert isinstance(action, Action)
        assert action.action_type in ['buy', 'sell', 'hold']


class TestPortfolio:
    """Test Portfolio class"""
    
    @pytest.fixture
    def portfolio(self):
        return Portfolio(initial_balance=10000.0)
    
    def test_initialization(self, portfolio):
        assert portfolio.initial_balance == 10000.0
        assert portfolio.cash == 10000.0
        assert portfolio.total_value == 10000.0
    
    def test_buy(self, portfolio):
        result = portfolio.buy('BTCUSDT', 0.1, 50000.0)
        
        assert result is True
        assert 'BTCUSDT' in portfolio.assets
        assert portfolio.assets['BTCUSDT'].quantity == 0.1
        assert portfolio.cash < 10000.0
    
    def test_sell(self, portfolio):
        portfolio.buy('BTCUSDT', 0.1, 50000.0)
        result = portfolio.sell('BTCUSDT', 0.05, 51000.0)
        
        assert result is True
        assert portfolio.assets['BTCUSDT'].quantity == 0.05
    
    def test_insufficient_funds(self, portfolio):
        result = portfolio.buy('BTCUSDT', 1.0, 50000.0)
        
        assert result is False
    
    def test_update_price(self, portfolio):
        portfolio.buy('BTCUSDT', 0.1, 50000.0)
        portfolio.update_price('BTCUSDT', 51000.0)
        
        assert portfolio.assets['BTCUSDT'].current_price == 51000.0
        assert portfolio.assets['BTCUSDT'].unrealized_pnl > 0


class TestSuperRiskManager:
    """Test SuperRiskManager"""
    
    @pytest.fixture
    def risk_manager(self):
        return SuperRiskManager(
            max_portfolio_risk=5.0,
            max_position_risk=2.0
        )
    
    def test_initialization(self, risk_manager):
        assert risk_manager.max_portfolio_risk == 5.0
        assert risk_manager.max_position_risk == 2.0
    
    def test_calculate_position_size(self, risk_manager):
        size, risk = risk_manager.calculate_position_size(
            account_value=10000.0,
            entry_price=50000.0,
            stop_loss=49000.0,
            risk_percent=2.0
        )
        
        assert size > 0
        assert risk > 0
    
    def test_check_trade_allowed(self, risk_manager):
        allowed, reason = risk_manager.check_trade_allowed(
            symbol='BTCUSDT',
            action='buy',
            position_size=1000.0,
            account_value=10000.0
        )
        
        assert allowed is True
        assert reason == "Trade allowed"
    
    def test_update_drawdown(self, risk_manager):
        risk_manager.update_drawdown(10000.0)
        risk_manager.update_drawdown(9000.0)
        
        assert risk_manager.current_drawdown == 10.0
        assert risk_manager.max_observed_drawdown == 10.0


class TestTechnicalIndicators:
    """Test TechnicalIndicators"""
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
    
    def test_sma(self, sample_data):
        sma = TechnicalIndicators.sma(sample_data['close'], 20)
        assert len(sma) == len(sample_data)
        assert not pd.isna(sma.iloc[-1])
    
    def test_rsi(self, sample_data):
        rsi = TechnicalIndicators.rsi(sample_data['close'], 14)
        assert len(rsi) == len(sample_data)
        assert 0 <= rsi.iloc[-1] <= 100 or pd.isna(rsi.iloc[-1])
    
    def test_macd(self, sample_data):
        macd, signal, hist = TechnicalIndicators.macd(sample_data['close'])
        assert len(macd) == len(sample_data)
        assert len(signal) == len(sample_data)
        assert len(hist) == len(sample_data)
    
    def test_bollinger_bands(self, sample_data):
        upper, middle, lower = TechnicalIndicators.bollinger_bands(sample_data['close'])
        assert len(upper) == len(sample_data)
        assert upper.iloc[-1] >= middle.iloc[-1] >= lower.iloc[-1]
    
    def test_atr(self, sample_data):
        atr = TechnicalIndicators.atr(
            sample_data['high'],
            sample_data['low'],
            sample_data['close']
        )
        assert len(atr) == len(sample_data)
        assert atr.iloc[-1] > 0 or pd.isna(atr.iloc[-1])
    
    def test_calculate_all(self, sample_data):
        result = TechnicalIndicators.calculate_all(sample_data)
        
        assert 'sma_20' in result.columns
        assert 'rsi' in result.columns
        assert 'macd' in result.columns
        assert 'bb_upper' in result.columns
        assert 'atr' in result.columns


class TestSuperTransformerAgent:
    """Test SuperTransformerAgent"""
    
    @pytest.fixture
    def agent(self):
        return SuperTransformerAgent(
            state_size=64,
            action_size=3,
            device='cpu'
        )
    
    def test_initialization(self, agent):
        assert agent.name == "SuperTransformerAgent"
        assert agent.sequence_length == 60
        assert agent.d_model == 128
    
    def test_sequence_buffer(self, agent):
        from agents.base_agent import MarketState
        
        # Add states to build sequence with 64 features
        for i in range(70):
            indicators = {f'ind_{j}': 50.0 + j for j in range(62)}
            state = MarketState(
                price=50000.0 + i * 100,
                volume=1000.0,
                timestamp=1234567890 + i,
                indicators=indicators
            )
            action = agent.predict(state)
        
        assert len(agent.sequence_buffer) == 60
        assert isinstance(action.action_type, str)


class TestSuperEnsembleAgent:
    """Test SuperEnsembleAgent"""
    
    @pytest.fixture
    def agent(self):
        return SuperEnsembleAgent(
            state_size=64,
            action_size=3,
            device='cpu'
        )
    
    def test_initialization(self, agent):
        assert agent.name == "SuperEnsembleAgent"
        assert len(agent.agents) == 3
        assert 'dqn' in agent.agents
        assert 'transformer' in agent.agents
        assert 'lstm' in agent.agents
    
    def test_agent_predictions(self, agent):
        from agents.base_agent import MarketState
        
        state = MarketState(
            price=50000.0,
            volume=1000.0,
            timestamp=1234567890,
            indicators={'rsi': 50.0}
        )
        
        action = agent.predict(state)
        
        assert action.action_type in ['buy', 'sell', 'hold']
        assert 'individual_predictions' in action.metadata
        assert 'dynamic_weights' in action.metadata


class TestDatabaseManager:
    """Test DatabaseManager"""
    
    @pytest.fixture
    def db(self):
        import tempfile
        import os
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield DatabaseManager(path)
        os.unlink(path)
    
    def test_initialization(self, db):
        assert db.db_path.exists()
    
    def test_save_and_get_trade(self, db):
        trade = {
            'id': 'test_123',
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'entry_price': 50000.0,
            'quantity': 0.1,
            'status': 'open'
        }
        
        result = db.save_trade(trade)
        assert result is True
        
        trades = db.get_trades(symbol='BTCUSDT')
        assert len(trades) == 1
        assert trades[0]['symbol'] == 'BTCUSDT'


class TestAuditLogger:
    """Test AuditLogger"""
    
    @pytest.fixture
    def audit(self):
        import tempfile
        import os
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield AuditLogger(path)
        os.unlink(path)
    
    def test_log_event(self, audit):
        event = audit.log(
            event_type='TEST',
            user_id='user1',
            action='TEST_ACTION',
            resource='test_resource',
            details={'key': 'value'}
        )
        
        assert event.hash != ''
        assert event.event_type == 'TEST'
    
    def test_log_trade(self, audit):
        event = audit.log_trade(
            user_id='user1',
            trade_action='BUY',
            symbol='BTCUSDT',
            quantity=0.1,
            price=50000.0
        )
        
        assert event.event_type == 'TRADE'
    
    def test_query(self, audit):
        audit.log('TEST', 'user1', 'ACTION', 'res', {})
        events = audit.query(user_id='user1')
        assert len(events) >= 1


class TestBacktestEngine:
    """Test BacktestEngine"""
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=200, freq='H')
        data = pd.DataFrame({
            'open': np.random.randn(200).cumsum() + 50000,
            'high': np.random.randn(200).cumsum() + 50100,
            'low': np.random.randn(200).cumsum() + 49900,
            'close': np.random.randn(200).cumsum() + 50000,
            'volume': np.random.randint(1000, 10000, 200)
        }, index=dates)
        return data
    
    @pytest.fixture
    def engine(self, sample_data):
        from backtest import BacktestEngine
        from agents import SuperDQNAgent
        agent = SuperDQNAgent(state_size=64, action_size=3, device='cpu')
        return BacktestEngine(agent=agent, initial_balance=10000.0)
    
    def test_initialization(self, engine):
        assert engine.initial_balance == 10000.0
        assert engine.commission == 0.001
    
    def test_calculate_drawdown(self, engine):
        equity = [10000, 11000, 10500, 9000, 9500, 12000]
        max_dd, max_dd_pct = engine._calculate_drawdown(equity)
        assert max_dd > 0
        assert max_dd_pct > 0


def run_tests():
    """Run all tests"""
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    run_tests()
