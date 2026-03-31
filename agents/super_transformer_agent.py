"""
Super Transformer Agent - Transformer-based AI for Crypto Trading
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional, Dict
from loguru import logger
import math

from .base_agent import BaseAgent, MarketState, Action, AgentState


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerTradingModel(nn.Module):
    """Transformer model for trading decisions"""
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        action_size: int = 3
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.attention_pool = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.LayerNorm(dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(dim_feedforward // 2, action_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Attention pooling
        attn_output, _ = self.attention_pool(encoded, encoded, encoded)
        
        # Global average pooling
        pooled = attn_output.mean(dim=1)
        
        # Decode to actions
        output = self.decoder(pooled)
        return output


class SuperTransformerAgent(BaseAgent):
    """
    Super Transformer Agent for crypto trading.
    Uses Transformer architecture to analyze sequential market data.
    """
    
    def __init__(
        self,
        state_size: int = 64,
        action_size: int = 3,
        sequence_length: int = 60,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        device: str = "auto",
        model_path: Optional[str] = None
    ):
        super().__init__(
            name="SuperTransformerAgent",
            state_size=state_size,
            action_size=action_size,
            device=device,
            model_path=model_path
        )
        
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.gamma = gamma
        
        # Initialize model
        self.model = TransformerTradingModel(
            input_size=state_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            action_size=action_size
        ).to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10)
        self.criterion = nn.CrossEntropyLoss()
        
        # Sequence buffer for temporal data
        self.sequence_buffer: List[np.ndarray] = []
        
        # Action mapping
        self.action_map = {0: "buy", 1: "sell", 2: "hold"}
        
        logger.info(f"SuperTransformerAgent initialized with d_model={d_model}, nhead={nhead}")
    
    def build_model(self) -> nn.Module:
        """Build transformer model"""
        return TransformerTradingModel(
            input_size=self.state_size,
            d_model=self.d_model,
            action_size=self.action_size
        )
    
    def _update_sequence(self, state: MarketState):
        """Update sequence buffer with new state"""
        self.sequence_buffer.append(state.to_vector())
        if len(self.sequence_buffer) > self.sequence_length:
            self.sequence_buffer.pop(0)
    
    def _get_sequence_tensor(self) -> Optional[torch.Tensor]:
        """Get sequence as tensor"""
        if len(self.sequence_buffer) < self.sequence_length:
            # Pad sequence if needed with zeros
            padding = [np.zeros(self.state_size, dtype=np.float32)] * (self.sequence_length - len(self.sequence_buffer))
            sequence = padding + list(self.sequence_buffer)
        else:
            sequence = list(self.sequence_buffer[-self.sequence_length:])
        
        # Ensure all elements have the same size
        sequence_array = np.array(sequence, dtype=np.float32)
        tensor = torch.FloatTensor(sequence_array).unsqueeze(0).to(self.device)
        return tensor
    
    def predict(self, state: MarketState) -> Action:
        """Make trading prediction"""
        self._update_sequence(state)
        
        sequence_tensor = self._get_sequence_tensor()
        if sequence_tensor is None:
            return Action("hold", 0.33, state.price, 0)
        
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(sequence_tensor)
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
                "sequence_length": len(self.sequence_buffer)
            }
        )
    
    def train_step(self, batch: List[Tuple]) -> float:
        """Perform one training step"""
        self.set_state(AgentState.TRAINING)
        self.model.train()
        
        sequences = []
        actions = []
        rewards = []
        
        for seq, action, reward in batch:
            sequences.append(seq)
            actions.append(action)
            rewards.append(reward)
        
        sequences = torch.FloatTensor(sequences).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        
        # Forward pass
        outputs = self.model(sequences)
        
        # Weighted cross-entropy with rewards
        log_probs = torch.log_softmax(outputs, dim=-1)
        selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = -(selected_log_probs * rewards).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
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
    
    def train(self, episodes: int = 1000, sequences: List = None, callback=None):
        """Train the agent"""
        logger.info(f"Starting transformer training for {episodes} episodes")
        
        if sequences is None or len(sequences) < self.batch_size:
            logger.warning("Insufficient training data")
            return
        
        for episode in range(episodes):
            # Sample batch
            batch = random.sample(sequences, min(self.batch_size, len(sequences)))
            loss = self.train_step(batch)
            
            self.training_history.append({
                "episode": episode,
                "loss": loss,
                "lr": self.scheduler.get_last_lr()[0]
            })
            
            if callback:
                callback(episode, loss)
            
            if episode % 100 == 0:
                logger.info(f"Episode {episode}/{episodes}, Loss: {loss:.4f}")
        
        logger.info("Training completed")
        self.set_state(AgentState.IDLE)
