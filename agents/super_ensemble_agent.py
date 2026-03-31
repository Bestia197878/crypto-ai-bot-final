"""
Super Ensemble Agent - Combines multiple AI agents for superior trading decisions
"""
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from collections import deque

from .base_agent import BaseAgent, MarketState, Action, AgentState
from .super_dqn_agent import SuperDQNAgent
from .super_transformer_agent import SuperTransformerAgent
from .lstm_agent import LSTMAgent


@dataclass
class AgentPrediction:
    """Prediction from a single agent"""
    agent_name: str
    action: str
    confidence: float
    weight: float


class MetaLearner(nn.Module):
    """Meta-learner network that combines agent predictions"""
    
    def __init__(self, num_agents: int, hidden_size: int = 64):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(num_agents * 4, hidden_size),  # 4 features per agent
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_agents)  # Weight for each agent
        )
    
    def forward(self, agent_features: torch.Tensor) -> torch.Tensor:
        """Generate dynamic weights for agents"""
        weights = self.network(agent_features)
        weights = torch.softmax(weights, dim=-1)
        return weights


class SuperEnsembleAgent(BaseAgent):
    """
    Super Ensemble Agent that combines multiple AI agents:
    - Super DQN Agent
    - Super Transformer Agent
    - LSTM Agent
    
    Uses a meta-learner to dynamically weight agent predictions
    based on recent performance.
    """
    
    def __init__(
        self,
        state_size: int = 64,
        action_size: int = 3,
        device: str = "auto",
        model_path: Optional[str] = None,
        agent_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__(
            name="SuperEnsembleAgent",
            state_size=state_size,
            action_size=action_size,
            device=device,
            model_path=model_path
        )
        
        # Initialize component agents
        self.dqn_agent = SuperDQNAgent(
            state_size=state_size,
            action_size=action_size,
            device=device
        )
        
        self.transformer_agent = SuperTransformerAgent(
            state_size=state_size,
            action_size=action_size,
            device=device
        )
        
        self.lstm_agent = LSTMAgent(
            state_size=state_size,
            action_size=action_size,
            device=device
        )
        
        self.agents = {
            "dqn": self.dqn_agent,
            "transformer": self.transformer_agent,
            "lstm": self.lstm_agent
        }
        
        # Agent weights (can be fixed or learned)
        self.agent_weights = agent_weights or {
            "dqn": 0.33,
            "transformer": 0.33,
            "lstm": 0.34
        }
        
        # Performance tracking for adaptive weighting
        self.performance_window = 100
        self.agent_performance = {
            name: deque(maxlen=self.performance_window) 
            for name in self.agents.keys()
        }
        
        # Meta-learner for dynamic weighting
        self.meta_learner = MetaLearner(num_agents=len(self.agents)).to(self.device)
        self.meta_optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=0.001)
        
        # Action mapping
        self.action_map = {0: "buy", 1: "sell", 2: "hold"}
        self.reverse_action_map = {v: k for k, v in self.action_map.items()}
        
        logger.info(f"SuperEnsembleAgent initialized with {len(self.agents)} agents")
    
    def build_model(self) -> nn.Module:
        """Build meta-learner model"""
        return self.meta_learner
    
    def predict(self, state: MarketState) -> Action:
        """Make ensemble prediction"""
        predictions = self._collect_predictions(state)
        
        # Get dynamic weights from meta-learner
        dynamic_weights = self._calculate_dynamic_weights(predictions, state)
        
        # Weighted voting
        action_scores = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        
        for agent_name, pred in predictions.items():
            weight = dynamic_weights.get(agent_name, self.agent_weights.get(agent_name, 0.33))
            action_scores[pred.action] += pred.confidence * weight
        
        # Select best action
        best_action = max(action_scores, key=action_scores.get)
        total_confidence = action_scores[best_action]
        
        # Normalize confidence
        total_score = sum(action_scores.values())
        normalized_confidence = total_confidence / total_score if total_score > 0 else 0.33
        
        # Calculate position size and risk levels
        quantity = self._calculate_position_size(normalized_confidence, state)
        stop_loss, take_profit = self._calculate_risk_levels(state, best_action)
        
        return Action(
            action_type=best_action,
            confidence=normalized_confidence,
            price=state.price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "individual_predictions": {
                    name: {"action": p.action, "confidence": p.confidence}
                    for name, p in predictions.items()
                },
                "dynamic_weights": dynamic_weights,
                "action_scores": action_scores
            }
        )
    
    def _collect_predictions(self, state: MarketState) -> Dict[str, AgentPrediction]:
        """Collect predictions from all agents"""
        predictions = {}
        
        for name, agent in self.agents.items():
            try:
                action = agent.predict(state)
                predictions[name] = AgentPrediction(
                    agent_name=name,
                    action=action.action_type,
                    confidence=action.confidence,
                    weight=self.agent_weights.get(name, 0.33)
                )
            except Exception as e:
                logger.error(f"Error getting prediction from {name}: {e}")
                predictions[name] = AgentPrediction(
                    agent_name=name,
                    action="hold",
                    confidence=0.33,
                    weight=self.agent_weights.get(name, 0.33)
                )
        
        return predictions
    
    def _calculate_dynamic_weights(
        self,
        predictions: Dict[str, AgentPrediction],
        state: MarketState
    ) -> Dict[str, float]:
        """Calculate dynamic weights using meta-learner"""
        # Prepare features for meta-learner
        features = []
        for name in ["dqn", "transformer", "lstm"]:
            pred = predictions.get(name)
            if pred:
                action_idx = self.reverse_action_map.get(pred.action, 2)
                # Fix: check if performance history exists
                perf_history = self.agent_performance.get(name, [])
                avg_perf = np.mean(list(perf_history)) if perf_history else 0.5
                features.extend([
                    action_idx / 2.0,  # Normalized action
                    pred.confidence,
                    pred.weight,
                    avg_perf
                ])
            else:
                features.extend([1.0, 0.33, 0.33, 0.5])
        
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            weights = self.meta_learner(features_tensor).squeeze(0)
        
        weight_dict = {
            "dqn": weights[0].item(),
            "transformer": weights[1].item(),
            "lstm": weights[2].item()
        }
        
        return weight_dict
    
    def update_performance(self, agent_name: str, reward: float):
        """Update agent performance history"""
        if agent_name in self.agent_performance:
            self.agent_performance[agent_name].append(reward)
    
    def train_step(self, batch: List[Tuple]) -> float:
        """Train meta-learner"""
        self.set_state(AgentState.TRAINING)
        
        if len(batch) < 10:
            return 0.0
        
        # Prepare training data
        features = []
        targets = []
        
        for state, agent_preds, best_agent_idx, reward in batch:
            # Create feature vector
            feat = []
            for name in ["dqn", "transformer", "lstm"]:
                pred = agent_preds.get(name, {})
                # Fix: check if performance history exists
                perf_history = self.agent_performance.get(name, [])
                avg_perf = np.mean(list(perf_history)) if perf_history else 0.5
                feat.extend([
                    self.reverse_action_map.get(pred.get("action", "hold"), 2) / 2.0,
                    pred.get("confidence", 0.33),
                    self.agent_weights.get(name, 0.33),
                    avg_perf
                ])
            
            features.append(feat)
            
            # Target is one-hot encoding of best performing agent
            target = [0.0, 0.0, 0.0]
            target[best_agent_idx] = 1.0
            targets.append(target)
        
        features_tensor = torch.FloatTensor(features).to(self.device)
        targets_tensor = torch.FloatTensor(targets).to(self.device)
        
        # Forward pass
        predictions = self.meta_learner(features_tensor)
        
        # Loss
        loss = nn.functional.kl_div(
            torch.log(predictions + 1e-8),
            targets_tensor,
            reduction='batchmean'
        )
        
        # Backward pass
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()
        
        return loss.item()
    
    def _calculate_position_size(self, confidence: float, state: MarketState) -> float:
        """Calculate position size"""
        base_size = 0.1
        return base_size * confidence
    
    def _calculate_risk_levels(
        self,
        state: MarketState,
        action_type: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit"""
        price = state.price
        atr = state.indicators.get("atr", price * 0.02)
        
        if action_type == "buy":
            stop_loss = price - 2 * atr
            take_profit = price + 3 * atr
        elif action_type == "sell":
            stop_loss = price + 2 * atr
            take_profit = price - 3 * atr
        else:
            return None, None
            
        return stop_loss, take_profit
    
    def get_agent_performance_report(self) -> Dict:
        """Get performance report for all agents"""
        report = {}
        for name, history in self.agent_performance.items():
            if len(history) > 0:
                report[name] = {
                    "mean_reward": np.mean(history),
                    "std_reward": np.std(history),
                    "recent_performance": np.mean(list(history)[-10:]) if len(history) >= 10 else np.mean(history)
                }
            else:
                report[name] = {"mean_reward": 0, "std_reward": 0, "recent_performance": 0}
        return report
    
    def save_model(self, path: Optional[str] = None) -> str:
        """Save ensemble model and all agent models"""
        save_path = super().save_model(path)
        
        # Save individual agent models
        for name, agent in self.agents.items():
            agent_path = str(Path(save_path).parent / f"{name}_model.pt")
            agent.save_model(agent_path)
        
        return save_path
    
    def load_model(self, path: Optional[str] = None) -> bool:
        """Load ensemble model and all agent models"""
        success = super().load_model(path)
        
        # Load individual agent models
        if path:
            for name, agent in self.agents.items():
                agent_path = str(Path(path).parent / f"{name}_model.pt")
                agent.load_model(agent_path)
        
        return success
    
    def reset(self):
        """Reset ensemble and all agents"""
        super().reset()
        for agent in self.agents.values():
            agent.reset()
        for history in self.agent_performance.values():
            history.clear()
