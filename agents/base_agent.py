"""
Base Agent class for all AI trading agents
"""
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import json
from loguru import logger


class AgentState(Enum):
    """Agent states"""
    IDLE = "idle"
    TRAINING = "training"
    TRADING = "trading"
    BACKTESTING = "backtesting"
    ERROR = "error"


@dataclass
class Action:
    """Trading action"""
    action_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    price: float
    quantity: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MarketState:
    """Market state representation"""
    price: float
    volume: float
    timestamp: int
    indicators: Dict[str, float]
    order_book: Optional[Dict] = None
    
    def to_vector(self) -> np.ndarray:
        """Convert market state to feature vector"""
        features = [self.price, self.volume]
        features.extend(self.indicators.values())
        return np.array(features, dtype=np.float32)


class BaseAgent(ABC):
    """Base class for all trading agents"""
    
    def __init__(
        self,
        name: str,
        state_size: int,
        action_size: int,
        device: str = "auto",
        model_path: Optional[str] = None
    ):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.model_path = model_path
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.state = AgentState.IDLE
        self.training = False  # Training mode flag
        self.training_history: List[Dict] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        logger.info(f"Initialized {self.name} on {self.device}")
    
    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build the neural network model"""
        pass
    
    @abstractmethod
    def predict(self, state: MarketState) -> Action:
        """Make a trading prediction"""
        pass
    
    @abstractmethod
    def train_step(self, batch: List[Tuple]) -> float:
        """Perform one training step"""
        pass
    
    def save_model(self, path: Optional[str] = None) -> str:
        """Save model to disk"""
        save_path = path or self.model_path
        if save_path is None:
            raise ValueError("No model path specified")
            
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "performance_metrics": self.performance_metrics,
            "training_history": self.training_history[-1000:]  # Last 1000 entries
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")
        return save_path
    
    def load_model(self, path: Optional[str] = None) -> bool:
        """Load model from disk"""
        load_path = path or self.model_path
        if load_path is None or not Path(load_path).exists():
            logger.warning(f"Model file not found: {load_path}")
            return False
            
        try:
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if self.optimizer and checkpoint.get("optimizer_state_dict"):
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.performance_metrics = checkpoint.get("performance_metrics", {})
            self.training_history = checkpoint.get("training_history", [])
            logger.info(f"Model loaded from {load_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_state(self) -> AgentState:
        """Get current agent state"""
        return self.state
    
    def set_state(self, state: AgentState):
        """Set agent state"""
        self.state = state
        logger.debug(f"Agent {self.name} state changed to {state.value}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get agent performance report"""
        return {
            "name": self.name,
            "state": self.state.value,
            "device": str(self.device),
            "metrics": self.performance_metrics,
            "training_episodes": len(self.training_history)
        }
    
    def reset(self):
        """Reset agent state"""
        self.state = AgentState.IDLE
        logger.info(f"Agent {self.name} reset")
    
    def preprocess_state(self, state: MarketState) -> torch.Tensor:
        """Preprocess market state for model input"""
        vector = state.to_vector()
        tensor = torch.FloatTensor(vector).unsqueeze(0).to(self.device)
        return tensor
    
    def calculate_confidence(self, predictions: torch.Tensor) -> float:
        """Calculate confidence score from predictions"""
        probs = torch.softmax(predictions, dim=-1)
        max_prob = torch.max(probs).item()
        return max_prob
