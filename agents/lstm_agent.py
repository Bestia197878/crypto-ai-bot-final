"""
LSTM Agent - LSTM-based AI for Crypto Trading
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional
from loguru import logger

from .base_agent import BaseAgent, MarketState, Action, AgentState


class LSTMTradingModel(nn.Module):
    """LSTM model for trading decisions"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        action_size: int = 3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output layers
        fc_input_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 2, action_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def attention_weights(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """Calculate attention weights"""
        attn_scores = self.attention(lstm_output).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        return attn_weights
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Apply attention
        attn_weights = self.attention_weights(lstm_out)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        
        # Fully connected layers
        output = self.fc(context)
        
        return output, hidden


class LSTMAgent(BaseAgent):
    """
    LSTM Agent for crypto trading.
    Uses LSTM with attention mechanism to analyze market sequences.
    """
    
    def __init__(
        self,
        state_size: int = 64,
        action_size: int = 3,
        sequence_length: int = 50,
        hidden_size: int = 128,
        num_layers: int = 3,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        device: str = "auto",
        model_path: Optional[str] = None
    ):
        super().__init__(
            name="LSTMAgent",
            state_size=state_size,
            action_size=action_size,
            device=device,
            model_path=model_path
        )
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gamma = gamma
        
        # Initialize model
        self.model = LSTMTradingModel(
            input_size=state_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            action_size=action_size
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Sequence buffer
        self.sequence_buffer: List[np.ndarray] = []
        self.hidden_state = None
        
        # Action mapping
        self.action_map = {0: "buy", 1: "sell", 2: "hold"}
        
        logger.info(f"LSTMAgent initialized with hidden_size={hidden_size}, num_layers={num_layers}")
    
    def build_model(self) -> nn.Module:
        """Build LSTM model"""
        return LSTMTradingModel(
            input_size=self.state_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            action_size=self.action_size
        )
    
    def _update_sequence(self, state: MarketState):
        """Update sequence buffer"""
        self.sequence_buffer.append(state.to_vector())
        if len(self.sequence_buffer) > self.sequence_length:
            self.sequence_buffer.pop(0)
    
    def _get_sequence_tensor(self) -> Optional[torch.Tensor]:
        """Get sequence as tensor"""
        if len(self.sequence_buffer) == 0:
            return None
        
        if len(self.sequence_buffer) < self.sequence_length:
            padding = [np.zeros(self.state_size)] * (self.sequence_length - len(self.sequence_buffer))
            sequence = padding + self.sequence_buffer
        else:
            sequence = self.sequence_buffer
        
        tensor = torch.FloatTensor(np.array(sequence)).unsqueeze(0).to(self.device)
        return tensor
    
    def predict(self, state: MarketState) -> Action:
        """Make trading prediction"""
        self._update_sequence(state)
        
        sequence_tensor = self._get_sequence_tensor()
        if sequence_tensor is None:
            return Action("hold", 0.33, state.price, 0)
        
        self.model.eval()
        
        with torch.no_grad():
            output, self.hidden_state = self.model(sequence_tensor, self.hidden_state)
            
            # Detach hidden state to prevent backprop through entire history
            if self.hidden_state is not None:
                self.hidden_state = tuple(h.detach() for h in self.hidden_state)
            
            probabilities = torch.softmax(output, dim=-1)
            action_idx = output.argmax(dim=1).item()
            confidence = probabilities[0][action_idx].item()
        
        action_type = self.action_map[action_idx]
        quantity = self._calculate_position_size(confidence, state)
        stop_loss, take_profit = self._calculate_risk_levels(state, action_type)
        
        return Action(
            action_type=action_type,
            confidence=confidence,
            price=state.price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "probabilities": probabilities.cpu().numpy().tolist(),
                "hidden_state_norm": torch.norm(self.hidden_state[0]).item() if self.hidden_state else 0
            }
        )
    
    def train_step(self, batch: List[Tuple]) -> float:
        """Perform one training step"""
        self.set_state(AgentState.TRAINING)
        self.model.train()
        
        sequences = []
        actions = []
        
        for seq, action, _ in batch:
            sequences.append(seq)
            actions.append(action)
        
        sequences = torch.FloatTensor(sequences).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        
        # Reset hidden state for training
        hidden = None
        
        # Forward pass
        outputs, _ = self.model(sequences, hidden)
        
        # Calculate loss
        loss = self.criterion(outputs, actions)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
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
    
    def reset(self):
        """Reset agent"""
        super().reset()
        self.sequence_buffer.clear()
        self.hidden_state = None
    
    def train(self, episodes: int = 1000, sequences: List = None, callback=None):
        """Train the agent"""
        logger.info(f"Starting LSTM training for {episodes} episodes")
        
        if sequences is None or len(sequences) < 32:
            logger.warning("Insufficient training data")
            return
        
        for episode in range(episodes):
            batch = np.random.choice(sequences, min(32, len(sequences)), replace=False)
            loss = self.train_step(batch)
            
            self.training_history.append({
                "episode": episode,
                "loss": loss
            })
            
            self.scheduler.step(loss)
            
            if callback:
                callback(episode, loss)
            
            if episode % 100 == 0:
                logger.info(f"Episode {episode}/{episodes}, Loss: {loss:.4f}")
        
        logger.info("Training completed")
        self.set_state(AgentState.IDLE)
