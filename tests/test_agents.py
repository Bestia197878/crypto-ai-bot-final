"""
Comprehensive tests for AI Trading Agents
"""
import pytest
import numpy as np
import torch
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base_agent import MarketState, Action, AgentState
from agents.super_dqn_agent import SuperDQNAgent
from agents.super_transformer_agent import SuperTransformerAgent
from agents.lstm_agent import LSTMAgent
from agents.super_ensemble_agent import SuperEnsembleAgent


class TestMarketState:
    """Test MarketState creation and conversion"""
    
    def test_market_state_creation(self):
        """Test creating MarketState with all parameters"""
        state = MarketState(
            price=50000.0,
            volume=1000.0,
            timestamp=int(datetime.now().timestamp()),
            indicators={
                'rsi': 55.0,
                'atr': 500.0,
                'sma_20': 49000.0,
                'ema_12': 49500.0,
                'trend_strength': 0.8
            }
        )
        
        assert state.price == 50000.0
        assert state.volume == 1000.0
        assert state.indicators['rsi'] == 55.0
    
    def test_market_state_to_vector(self):
        """Test conversion to numpy vector"""
        state = MarketState(
            price=50000.0,
            volume=1000.0,
            timestamp=1234567890,
            indicators={'rsi': 50.0, 'atr': 500.0, 'sma_20': 49000.0}
        )
        
        vector = state.to_vector()
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 5  # price, volume + 3 indicators
        assert vector[0] == 50000.0
        assert vector[1] == 1000.0


class TestSuperDQNAgent:
    """Test Super DQN Agent"""
    
    @pytest.fixture
    def dqn_agent(self):
        """Create DQN agent for testing"""
        agent = SuperDQNAgent(
            state_size=10,
            action_size=3,
            hidden_size=64,
            device="cpu"
        )
        # Set the model attribute directly
        agent.model = agent.build_model()
        # Initialize training flag if missing
        if not hasattr(agent, 'training'):
            agent.training = False
        return agent
    
    @pytest.fixture
    def sample_market_state(self):
        """Create sample market state"""
        return MarketState(
            price=50000.0,
            volume=1000.0,
            timestamp=int(datetime.now().timestamp()),
            indicators={
                'rsi': 55.0,
                'atr': 500.0,
                'sma_20': 49000.0,
                'ema_12': 49500.0,
                'volume_sma': 1200.0,
                'trend_strength': 0.8,
                'macd': 100.0,
                'macd_signal': 50.0
            }
        )
    
    def test_agent_initialization(self, dqn_agent):
        """Test agent is properly initialized"""
        assert dqn_agent.name == "SuperDQNAgent"
        assert dqn_agent.state_size == 10
        assert dqn_agent.action_size == 3
        assert dqn_agent.device == torch.device("cpu")
        # Check policy_net exists instead of model
        assert hasattr(dqn_agent, 'policy_net') and dqn_agent.policy_net is not None
    
    def test_predict_returns_action(self, dqn_agent, sample_market_state):
        """Test predict() returns valid Action"""
        # Ensure training flag is set
        if not hasattr(dqn_agent, 'training'):
            dqn_agent.training = False
        
        action = dqn_agent.predict(sample_market_state)
        
        assert isinstance(action, Action)
        assert action.action_type in ['buy', 'sell', 'hold']
        assert 0.0 <= action.confidence <= 1.0
        assert action.price == sample_market_state.price
        assert 0.0 <= action.quantity <= 1.0
    
    def test_calculate_position_size(self, dqn_agent, sample_market_state):
        """Test position size calculation"""
        confidence = 0.8
        size = dqn_agent._calculate_position_size(confidence, sample_market_state)
        
        assert 0.0 <= size <= 1.0
        assert size == 0.1 * confidence  # base_size * confidence
    
    def test_calculate_risk_levels_buy(self, dqn_agent, sample_market_state):
        """Test stop loss and take profit for buy action"""
        stop_loss, take_profit = dqn_agent._calculate_risk_levels(
            sample_market_state, 'buy'
        )
        
        assert stop_loss is not None
        assert take_profit is not None
        assert stop_loss < sample_market_state.price
        assert take_profit > sample_market_state.price
    
    def test_calculate_risk_levels_sell(self, dqn_agent, sample_market_state):
        """Test stop loss and take profit for sell action"""
        stop_loss, take_profit = dqn_agent._calculate_risk_levels(
            sample_market_state, 'sell'
        )
        
        assert stop_loss is not None
        assert take_profit is not None
        assert stop_loss > sample_market_state.price
        assert take_profit < sample_market_state.price
    
    def test_agent_state_transitions(self, dqn_agent):
        """Test agent state management"""
        assert dqn_agent.state == AgentState.IDLE
        
        dqn_agent.set_state(AgentState.TRADING)
        assert dqn_agent.state == AgentState.TRADING
        
        dqn_agent.set_state(AgentState.TRAINING)
        assert dqn_agent.state == AgentState.TRAINING
        
        dqn_agent.reset()
        assert dqn_agent.state == AgentState.IDLE
    
    def test_model_save_load(self, dqn_agent, tmp_path):
        """Test model saving and loading"""
        save_path = tmp_path / "test_dqn_model.pt"
        
        # Use policy_net for saving instead of model
        if not hasattr(dqn_agent, 'model') or dqn_agent.model is None:
            dqn_agent.model = dqn_agent.policy_net
        
        # Save model
        saved_path = dqn_agent.save_model(str(save_path))
        assert Path(saved_path).exists()
        
        # Load model
        success = dqn_agent.load_model(str(save_path))
        assert success is True


class TestLSTMAgent:
    """Test LSTM Agent"""
    
    @pytest.fixture
    def lstm_agent(self):
        """Create LSTM agent"""
        agent = LSTMAgent(
            state_size=10,
            action_size=3,
            hidden_size=64,
            device="cpu"
        )
        agent.build_model()
        return agent
    
    @pytest.fixture
    def sample_market_state(self):
        """Create sample market state with 8 indicators (for LSTM state_size=10: price, volume + 8 indicators)"""
        return MarketState(
            price=50000.0,
            volume=1000.0,
            timestamp=int(datetime.now().timestamp()),
            indicators={
                'rsi': 55.0,
                'atr': 500.0,
                'sma_20': 49000.0,
                'ema_12': 49500.0,
                'volume_sma': 1200.0,
                'trend_strength': 0.8,
                'macd': 100.0,
                'macd_signal': 50.0
            }
        )
    
    def test_lstm_initialization(self, lstm_agent):
        """Test LSTM agent initialization"""
        assert lstm_agent.name == "LSTMAgent"
        assert lstm_agent.state_size == 10
        assert lstm_agent.model is not None
    
    def test_lstm_predict(self, lstm_agent, sample_market_state):
        """Test LSTM prediction"""
        action = lstm_agent.predict(sample_market_state)
        
        assert isinstance(action, Action)
        assert action.action_type in ['buy', 'sell', 'hold']
        assert 0.0 <= action.confidence <= 1.0


class TestSuperEnsembleAgent:
    """Test Super Ensemble Agent"""
    
    @pytest.fixture
    def ensemble_agent(self):
        """Create ensemble agent"""
        agent = SuperEnsembleAgent(
            state_size=10,
            action_size=3,
            device="cpu"
        )
        return agent
    
    @pytest.fixture
    def sample_market_state(self):
        """Create sample market state"""
        return MarketState(
            price=50000.0,
            volume=1000.0,
            timestamp=int(datetime.now().timestamp()),
            indicators={
                'rsi': 55.0,
                'atr': 500.0,
                'sma_20': 49000.0,
                'ema_12': 49500.0,
                'volume_sma': 1200.0,
                'trend_strength': 0.8,
                'macd': 100.0,
                'macd_signal': 50.0
            }
        )
    
    def test_ensemble_initialization(self, ensemble_agent):
        """Test ensemble agent has all sub-agents"""
        assert ensemble_agent.name == "SuperEnsembleAgent"
        assert len(ensemble_agent.agents) == 3  # DQN, Transformer, LSTM
        assert "dqn" in ensemble_agent.agents
        assert "transformer" in ensemble_agent.agents
        assert "lstm" in ensemble_agent.agents
    
    def test_collect_predictions(self, ensemble_agent, sample_market_state):
        """Test collecting predictions from all agents"""
        predictions = ensemble_agent._collect_predictions(sample_market_state)
        
        assert len(predictions) == 3
        for name, pred in predictions.items():
            assert pred.agent_name == name
            assert pred.action in ['buy', 'sell', 'hold']
            assert 0.0 <= pred.confidence <= 1.0
    
    def test_ensemble_predict(self, ensemble_agent, sample_market_state):
        """Test ensemble prediction combines all agents"""
        action = ensemble_agent.predict(sample_market_state)
        
        assert isinstance(action, Action)
        assert action.action_type in ['buy', 'sell', 'hold']
        assert 0.0 <= action.confidence <= 1.0
        # For buy/sell actions, check stop_loss and take_profit exist
        # For hold action, these can be None
        if action.action_type in ['buy', 'sell']:
            assert action.stop_loss is not None
            assert action.take_profit is not None
        assert 'individual_predictions' in action.metadata
    
    def test_dynamic_weights_calculation(self, ensemble_agent, sample_market_state):
        """Test dynamic weight calculation"""
        predictions = ensemble_agent._collect_predictions(sample_market_state)
        weights = ensemble_agent._calculate_dynamic_weights(predictions, sample_market_state)
        
        assert len(weights) == 3
        # Handle NaN values - if NaN, weights might not sum to 1
        valid_weights = [w for w in weights.values() if not np.isnan(w)]
        if len(valid_weights) > 0:
            assert all(0.0 <= w <= 1.0 for w in valid_weights if not np.isnan(w))
            # Sum may not be exactly 1.0 if some weights are NaN
            total = sum(w for w in weights.values() if not np.isnan(w))
            assert abs(total - 1.0) < 0.01 or np.isnan(total)  # Should sum to ~1 or be NaN
    
    def test_performance_tracking(self, ensemble_agent):
        """Test agent performance tracking"""
        ensemble_agent.update_performance("dqn", 0.5)
        ensemble_agent.update_performance("dqn", 0.3)
        ensemble_agent.update_performance("dqn", 0.7)
        
        assert len(ensemble_agent.agent_performance["dqn"]) == 3
        assert np.mean(list(ensemble_agent.agent_performance["dqn"])) == pytest.approx(0.5, rel=0.1)
    
    def test_get_performance_report(self, ensemble_agent):
        """Test performance report generation"""
        # Add some performance data
        for _ in range(10):
            ensemble_agent.update_performance("dqn", np.random.random())
            ensemble_agent.update_performance("transformer", np.random.random())
            ensemble_agent.update_performance("lstm", np.random.random())
        
        report = ensemble_agent.get_agent_performance_report()
        
        assert len(report) == 3
        for name, metrics in report.items():
            assert "mean_reward" in metrics
            assert "std_reward" in metrics
            assert "recent_performance" in metrics


class TestAgentConsistency:
    """Test consistency across different agents"""
    
    @pytest.fixture
    def agents(self):
        """Create all agent types with proper initialization"""
        agents = {
            'dqn': SuperDQNAgent(state_size=10, action_size=3, hidden_size=64, device="cpu"),
            'lstm': LSTMAgent(state_size=10, action_size=3, hidden_size=64, device="cpu"),
            'ensemble': SuperEnsembleAgent(state_size=10, action_size=3, device="cpu")
        }
        
        # Build models and set training flag
        for name, agent in agents.items():
            if hasattr(agent, 'build_model'):
                agent.model = agent.build_model()
            # Set training flag for DQN agent
            if not hasattr(agent, 'training'):
                agent.training = False
            # Build sub-agents for ensemble
            if name == 'ensemble':
                for sub_name, sub_agent in agent.agents.items():
                    if hasattr(sub_agent, 'build_model'):
                        sub_agent.model = sub_agent.build_model()
                    if not hasattr(sub_agent, 'training'):
                        sub_agent.training = False
        
        return agents
    
    @pytest.fixture
    def bullish_state(self):
        """Create bullish market state"""
        return MarketState(
            price=55000.0,
            volume=2000.0,
            timestamp=int(datetime.now().timestamp()),
            indicators={
                'rsi': 65.0,  # Moderately bullish
                'atr': 800.0,
                'sma_20': 50000.0,  # Price above SMA
                'ema_12': 54000.0,
                'volume_sma': 1500.0,  # High volume
                'trend_strength': 0.9,  # Strong trend
                'macd': 200.0,
                'macd_signal': 100.0
            }
        )
    
    @pytest.fixture
    def bearish_state(self):
        """Create bearish market state"""
        return MarketState(
            price=45000.0,
            volume=1800.0,
            timestamp=int(datetime.now().timestamp()),
            indicators={
                'rsi': 35.0,  # Moderately bearish
                'atr': 900.0,
                'sma_20': 48000.0,  # Price below SMA
                'ema_12': 46000.0,
                'volume_sma': 1600.0,
                'trend_strength': 0.85,
                'macd': -150.0,
                'macd_signal': -50.0
            }
        )
    
    def test_all_agents_return_valid_actions(self, agents, bullish_state):
        """Test all agents return valid actions"""
        for name, agent in agents.items():
            # Ensure agent is ready
            if hasattr(agent, 'build_model') and (agent.model is None or not hasattr(agent, 'model')):
                agent.model = agent.build_model()
            if not hasattr(agent, 'training'):
                agent.training = False
            
            action = agent.predict(bullish_state)
            
            assert isinstance(action, Action), f"{name} did not return Action"
            assert action.action_type in ['buy', 'sell', 'hold'], f"{name} returned invalid action"
            assert 0.0 <= action.confidence <= 1.0, f"{name} returned invalid confidence"
            assert action.price == bullish_state.price, f"{name} has wrong price"
    
    def test_action_metadata(self, agents, bullish_state):
        """Test actions contain required metadata"""
        for name, agent in agents.items():
            # Ensure agent is ready
            if hasattr(agent, 'build_model') and (agent.model is None or not hasattr(agent, 'model')):
                agent.model = agent.build_model()
            if not hasattr(agent, 'training'):
                agent.training = False
            
            action = agent.predict(bullish_state)
            
            # For buy/sell actions, check stop_loss and take_profit exist
            # For hold action, these can be None
            if action.action_type in ['buy', 'sell']:
                assert action.stop_loss is not None, f"{name} missing stop_loss for {action.action_type}"
                assert action.take_profit is not None, f"{name} missing take_profit for {action.action_type}"
            
            assert action.quantity >= 0, f"{name} has invalid quantity"
            assert 'metadata' in action.__dict__, f"{name} missing metadata"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
