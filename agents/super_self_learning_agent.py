"""
Super Self-Learning Agent - Continuously learns and adapts to market conditions
"""
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Deque
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
from loguru import logger
import json
from pathlib import Path

from .base_agent import BaseAgent, MarketState, Action, AgentState
from .super_ensemble_agent import SuperEnsembleAgent


@dataclass
class MarketRegime:
    """Market regime classification"""
    regime_type: str  # 'trending_up', 'trending_down', 'ranging', 'volatile'
    confidence: float
    features: Dict[str, float]
    timestamp: datetime


@dataclass
class LearningExperience:
    """Experience for self-learning"""
    state: np.ndarray
    action: str
    reward: float
    next_state: np.ndarray
    market_regime: str
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


class RegimeDetector(nn.Module):
    """Detects current market regime"""
    
    def __init__(self, input_size: int, num_regimes: int = 4):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_regimes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.network(x), dim=-1)


class SuperSelfLearningAgent(BaseAgent):
    """
    Super Self-Learning Agent that:
    1. Detects market regimes
    2. Adapts strategy based on regime
    3. Continuously learns from experience
    4. Maintains separate models for different regimes
    """
    
    REGIME_TYPES = ["trending_up", "trending_down", "ranging", "volatile"]
    
    def __init__(
        self,
        state_size: int = 64,
        action_size: int = 3,
        device: str = "auto",
        model_path: Optional[str] = None,
        memory_size: int = 10000,
        learning_interval: int = 100
    ):
        super().__init__(
            name="SuperSelfLearningAgent",
            state_size=state_size,
            action_size=action_size,
            device=device,
            model_path=model_path
        )
        
        self.learning_interval = learning_interval
        self.step_counter = 0
        
        # Base ensemble agent
        self.base_agent = SuperEnsembleAgent(
            state_size=state_size,
            action_size=action_size,
            device=device
        )
        
        # Regime detector
        self.regime_detector = RegimeDetector(state_size, len(self.REGIME_TYPES)).to(self.device)
        self.regime_optimizer = torch.optim.Adam(self.regime_detector.parameters(), lr=0.001)
        
        # Current regime
        self.current_regime: Optional[MarketRegime] = None
        
        # Experience memory organized by regime
        self.regime_memories: Dict[str, Deque[LearningExperience]] = {
            regime: deque(maxlen=memory_size)
            for regime in self.REGIME_TYPES
        }
        
        # Performance tracking by regime
        self.regime_performance: Dict[str, Dict] = {
            regime: {"wins": 0, "losses": 0, "total_reward": 0.0}
            for regime in self.REGIME_TYPES
        }
        
        # Strategy parameters by regime
        self.regime_strategies: Dict[str, Dict] = self._initialize_strategies()
        
        # Learning history
        self.learning_history: List[Dict] = []
        
        logger.info("SuperSelfLearningAgent initialized with regime detection")
    
    def _initialize_strategies(self) -> Dict[str, Dict]:
        """Initialize strategy parameters for each regime"""
        return {
            "trending_up": {
                "position_size_multiplier": 1.2,
                "stop_loss_factor": 1.5,
                "take_profit_factor": 3.0,
                "preferred_action": "buy"
            },
            "trending_down": {
                "position_size_multiplier": 1.2,
                "stop_loss_factor": 1.5,
                "take_profit_factor": 3.0,
                "preferred_action": "sell"
            },
            "ranging": {
                "position_size_multiplier": 0.8,
                "stop_loss_factor": 1.0,
                "take_profit_factor": 2.0,
                "preferred_action": "hold"
            },
            "volatile": {
                "position_size_multiplier": 0.5,
                "stop_loss_factor": 2.0,
                "take_profit_factor": 4.0,
                "preferred_action": "hold"
            }
        }
    
    def build_model(self) -> nn.Module:
        """Build regime detector model"""
        return self.regime_detector
    
    def detect_regime(self, state: MarketState) -> MarketRegime:
        """Detect current market regime"""
        state_tensor = self.preprocess_state(state)
        
        with torch.no_grad():
            regime_probs = self.regime_detector(state_tensor).squeeze(0)
            regime_idx = regime_probs.argmax().item()
            confidence = regime_probs[regime_idx].item()
        
        regime_type = self.REGIME_TYPES[regime_idx]
        
        # Extract regime-related features
        features = {
            "adx": state.indicators.get("adx", 25),
            "volatility": state.indicators.get("atr", 0) / state.price * 100,
            "trend_strength": state.indicators.get("trend_strength", 0),
            "rsi": state.indicators.get("rsi", 50)
        }
        
        return MarketRegime(
            regime_type=regime_type,
            confidence=confidence,
            features=features,
            timestamp=datetime.now()
        )
    
    def predict(self, state: MarketState) -> Action:
        """Make prediction with regime-aware adaptation"""
        # Detect current regime
        self.current_regime = self.detect_regime(state)
        
        # Get base prediction from ensemble
        base_action = self.base_agent.predict(state)
        
        # Adapt based on regime
        strategy = self.regime_strategies.get(self.current_regime.regime_type, {})
        
        # Adjust position size
        position_multiplier = strategy.get("position_size_multiplier", 1.0)
        adjusted_quantity = base_action.quantity * position_multiplier
        
        # Adjust stop loss and take profit
        stop_loss_factor = strategy.get("stop_loss_factor", 1.0)
        take_profit_factor = strategy.get("take_profit_factor", 1.0)
        
        adjusted_stop_loss = None
        adjusted_take_profit = None
        
        if base_action.stop_loss:
            price_diff_sl = abs(state.price - base_action.stop_loss)
            adjusted_stop_loss = state.price - price_diff_sl * stop_loss_factor if base_action.action_type == "buy" else state.price + price_diff_sl * stop_loss_factor
        
        if base_action.take_profit:
            price_diff_tp = abs(base_action.take_profit - state.price)
            adjusted_take_profit = state.price + price_diff_tp * take_profit_factor if base_action.action_type == "buy" else state.price - price_diff_tp * take_profit_factor
        
        # Store experience for learning
        self._store_experience(state, base_action)
        
        # Trigger learning if needed
        self.step_counter += 1
        if self.step_counter % self.learning_interval == 0:
            self._self_learn()
        
        return Action(
            action_type=base_action.action_type,
            confidence=base_action.confidence * self.current_regime.confidence,
            price=base_action.price,
            quantity=adjusted_quantity,
            stop_loss=adjusted_stop_loss,
            take_profit=adjusted_take_profit,
            metadata={
                **base_action.metadata,
                "regime": self.current_regime.regime_type,
                "regime_confidence": self.current_regime.confidence,
                "regime_features": self.current_regime.features,
                "adaptations": {
                    "position_multiplier": position_multiplier,
                    "stop_loss_factor": stop_loss_factor,
                    "take_profit_factor": take_profit_factor
                }
            }
        )
    
    def _store_experience(self, state: MarketState, action: Action):
        """Store experience for learning"""
        if self.current_regime is None:
            return
        
        experience = LearningExperience(
            state=state.to_vector(),
            action=action.action_type,
            reward=0.0,  # Will be updated later
            next_state=np.zeros_like(state.to_vector()),  # Will be filled later
            market_regime=self.current_regime.regime_type,
            timestamp=datetime.now()
        )
        
        self.regime_memories[self.current_regime.regime_type].append(experience)
    
    def update_reward(self, reward: float):
        """Update reward for last experience"""
        if self.current_regime:
            memory = self.regime_memories[self.current_regime.regime_type]
            if len(memory) > 0:
                last_exp = memory[-1]
                last_exp.reward = reward
                
                # Update performance tracking
                if reward > 0:
                    self.regime_performance[self.current_regime.regime_type]["wins"] += 1
                else:
                    self.regime_performance[self.current_regime.regime_type]["losses"] += 1
                self.regime_performance[self.current_regime.regime_type]["total_reward"] += reward
    
    def _self_learn(self):
        """Perform self-learning from accumulated experiences"""
        logger.info("Starting self-learning cycle")
        
        for regime in self.REGIME_TYPES:
            memory = self.regime_memories[regime]
            if len(memory) < 100:
                continue
            
            # Analyze performance in this regime
            recent_experiences = list(memory)[-100:]
            avg_reward = np.mean([e.reward for e in recent_experiences])
            
            # Adjust strategy parameters based on performance
            if avg_reward > 0:
                # Strategy is working, can be more aggressive
                self.regime_strategies[regime]["position_size_multiplier"] = min(
                    2.0, self.regime_strategies[regime]["position_size_multiplier"] * 1.05
                )
            else:
                # Strategy not working, be more conservative
                self.regime_strategies[regime]["position_size_multiplier"] = max(
                    0.3, self.regime_strategies[regime]["position_size_multiplier"] * 0.95
                )
            
            # Train regime detector
            self._train_regime_detector(regime, recent_experiences)
            
            logger.info(f"Updated strategy for {regime}: avg_reward={avg_reward:.4f}")
        
        # Train base agent
        self.base_agent.train_step([
            (e.state, e.action, e.reward) 
            for memory in self.regime_memories.values() 
            for e in list(memory)[-1000:]
        ])
        
        self.learning_history.append({
            "timestamp": datetime.now().isoformat(),
            "total_experiences": sum(len(m) for m in self.regime_memories.values()),
            "strategies": self.regime_strategies.copy()
        })
    
    def _train_regime_detector(self, regime: str, experiences: List[LearningExperience]):
        """Train regime detector"""
        if len(experiences) < 10:
            return
        
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        regime_idx = self.REGIME_TYPES.index(regime)
        targets = torch.full((len(experiences),), regime_idx, dtype=torch.long).to(self.device)
        
        self.regime_detector.train()
        
        predictions = self.regime_detector(states)
        loss = nn.functional.cross_entropy(predictions, targets)
        
        self.regime_optimizer.zero_grad()
        loss.backward()
        self.regime_optimizer.step()
    
    def train_step(self, batch: List[Tuple]) -> float:
        """Train agent"""
        return self.base_agent.train_step(batch)
    
    def get_regime_report(self) -> Dict:
        """Get report on regime detection and performance"""
        return {
            "current_regime": self.current_regime.regime_type if self.current_regime else None,
            "regime_confidence": self.current_regime.confidence if self.current_regime else 0,
            "regime_performance": self.regime_performance,
            "regime_strategies": self.regime_strategies,
            "experience_counts": {
                regime: len(memory) 
                for regime, memory in self.regime_memories.items()
            }
        }
    
    def save_model(self, path: Optional[str] = None) -> str:
        """Save model with all learning data"""
        save_path = super().save_model(path)
        
        # Save learning data
        learning_data = {
            "regime_strategies": self.regime_strategies,
            "regime_performance": self.regime_performance,
            "learning_history": self.learning_history[-100:]
        }
        
        learning_path = Path(save_path).parent / "learning_data.json"
        with open(learning_path, 'w') as f:
            json.dump(learning_data, f, indent=2, default=str)
        
        # Save base agent
        self.base_agent.save_model(str(Path(save_path).parent / "base_ensemble.pt"))
        
        return save_path
    
    def load_model(self, path: Optional[str] = None) -> bool:
        """Load model with learning data"""
        success = super().load_model(path)
        
        if path:
            learning_path = Path(path).parent / "learning_data.json"
            if learning_path.exists():
                with open(learning_path, 'r') as f:
                    learning_data = json.load(f)
                self.regime_strategies = learning_data.get("regime_strategies", self._initialize_strategies())
                self.regime_performance = learning_data.get("regime_performance", {})
                self.learning_history = learning_data.get("learning_history", [])
            
            # Load base agent
            self.base_agent.load_model(str(Path(path).parent / "base_ensemble.pt"))
        
        return success
    
    def reset(self):
        """Reset agent"""
        super().reset()
        self.base_agent.reset()
        for memory in self.regime_memories.values():
            memory.clear()
        self.current_regime = None
        self.step_counter = 0
