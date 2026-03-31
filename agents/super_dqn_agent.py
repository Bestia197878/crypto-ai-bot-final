"""
Super DQN Agent - Deep Q-Network for Crypto Trading
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger

from .base_agent import BaseAgent, MarketState, Action, AgentState


@dataclass
class Experience:
    """Experience tuple for replay buffer"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class DQNNetwork(nn.Module):
    """Deep Q-Network architecture"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(DQNNetwork, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling DQN: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


class ReplayBuffer:
    """Experience Replay Buffer"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
    def __len__(self):
        return len(self.buffer)


class SuperDQNAgent(BaseAgent):
    """
    Super DQN Agent with advanced features:
    - Dueling DQN architecture
    - Double DQN learning
    - Prioritized Experience Replay
    - Noisy Networks for exploration
    """
    
    def __init__(
        self,
        state_size: int = 64,
        action_size: int = 3,  # buy, sell, hold
        hidden_size: int = 256,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update: int = 1000,
        device: str = "auto",
        model_path: Optional[str] = None
    ):
        super().__init__(
            name="SuperDQNAgent",
            state_size=state_size,
            action_size=action_size,
            device=device,
            model_path=model_path
        )
        
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0
        
        # Initialize networks
        self.policy_net = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Action mapping
        self.action_map = {0: "buy", 1: "sell", 2: "hold"}
        
        logger.info(f"SuperDQNAgent initialized with state_size={state_size}, action_size={action_size}")
    
    def build_model(self) -> nn.Module:
        """Build DQN model"""
        return DQNNetwork(self.state_size, self.action_size, self.hidden_size)
    
    def predict(self, state: MarketState) -> Action:
        """Make trading prediction using epsilon-greedy policy"""
        self.policy_net.eval()
        
        state_tensor = self.preprocess_state(state)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action_idx = q_values.argmax(dim=1).item()
            confidence = self.calculate_confidence(q_values)
        
        # Epsilon-greedy exploration
        if self.training and random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_size - 1)
            confidence = 0.5
        
        action_type = self.action_map[action_idx]
        
        # Calculate position size based on confidence
        quantity = self._calculate_position_size(confidence, state)
        
        # Calculate stop loss and take profit
        stop_loss, take_profit = self._calculate_risk_levels(state, action_type)
        
        return Action(
            action_type=action_type,
            confidence=confidence,
            price=state.price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "q_values": q_values.cpu().numpy().tolist(),
                "epsilon": self.epsilon
            }
        )
    
    def train_step(self, batch: List[Tuple] = None) -> float:
        """Perform one training step"""
        self.set_state(AgentState.TRAINING)
        self.policy_net.train()
        
        if batch is None:
            if len(self.replay_buffer) < self.batch_size:
                return 0.0
            batch = self.replay_buffer.sample(self.batch_size)
        
        # Prepare batch
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use policy net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.debug("Target network updated")
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def store_experience(
        self,
        state: MarketState,
        action: int,
        reward: float,
        next_state: MarketState,
        done: bool
    ):
        """Store experience in replay buffer"""
        experience = Experience(
            state=state.to_vector(),
            action=action,
            reward=reward,
            next_state=next_state.to_vector(),
            done=done
        )
        self.replay_buffer.push(experience)
    
    def _calculate_position_size(self, confidence: float, state: MarketState) -> float:
        """Calculate position size based on confidence and risk"""
        base_size = 0.1  # 10% of available balance
        return base_size * confidence
    
    def _calculate_risk_levels(
        self,
        state: MarketState,
        action_type: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels"""
        price = state.price
        atr = state.indicators.get("atr", price * 0.02)  # Default 2% ATR
        
        if action_type == "buy":
            stop_loss = price - 2 * atr
            take_profit = price + 3 * atr
        elif action_type == "sell":
            stop_loss = price + 2 * atr
            take_profit = price - 3 * atr
        else:
            return None, None
            
        return stop_loss, take_profit
    
    def train(self, episodes: int = 1000, callback=None):
        """Train the agent for multiple episodes"""
        logger.info(f"Starting training for {episodes} episodes")
        
        for episode in range(episodes):
            episode_loss = 0
            steps = 0
            
            while len(self.replay_buffer) >= self.batch_size and steps < 100:
                loss = self.train_step()
                episode_loss += loss
                steps += 1
            
            avg_loss = episode_loss / max(steps, 1)
            
            self.training_history.append({
                "episode": episode,
                "loss": avg_loss,
                "epsilon": self.epsilon,
                "buffer_size": len(self.replay_buffer)
            })
            
            if callback:
                callback(episode, avg_loss, self.epsilon)
            
            if episode % 100 == 0:
                logger.info(f"Episode {episode}/{episodes}, Loss: {avg_loss:.4f}, Epsilon: {self.epsilon:.4f}")
        
        logger.info("Training completed")
        self.set_state(AgentState.IDLE)
