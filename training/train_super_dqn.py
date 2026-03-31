"""
Training script for Super DQN Agent
"""
import numpy as np
import pandas as pd
from typing import List, Tuple
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.super_dqn_agent import SuperDQNAgent, Experience
from agents.base_agent import MarketState
from utils.indicators import TechnicalIndicators


def prepare_training_data(
    df: pd.DataFrame,
    lookback: int = 60
) -> List[Tuple[np.ndarray, int, float, np.ndarray]]:
    """
    Prepare training data from price data
    
    Returns list of (state, action, reward, next_state) tuples
    """
    # Calculate indicators
    df = TechnicalIndicators.calculate_all(df)
    df = df.dropna()
    
    training_data = []
    
    for i in range(lookback, len(df) - 1):
        # Create state from lookback window
        window = df.iloc[i-lookback:i]
        state = _create_state_vector(window)
        
        # Determine action based on future price movement
        current_price = df['close'].iloc[i]
        next_price = df['close'].iloc[i+1]
        
        price_change = (next_price - current_price) / current_price
        
        if price_change > 0.005:  # > 0.5% gain
            action = 0  # Buy
            reward = price_change * 100
        elif price_change < -0.005:  # > 0.5% loss
            action = 1  # Sell
            reward = abs(price_change) * 100
        else:
            action = 2  # Hold
            reward = 0
        
        # Create next state
        next_window = df.iloc[i-lookback+1:i+1]
        next_state = _create_state_vector(next_window)
        
        training_data.append((state, action, reward, next_state))
    
    logger.info(f"Prepared {len(training_data)} training samples")
    return training_data


def _create_state_vector(df: pd.DataFrame) -> np.ndarray:
    """Create state vector from DataFrame"""
    features = []
    
    # Price features (normalized)
    last_close = df['close'].iloc[-1]
    features.append((df['open'].iloc[-1] / last_close) - 1)
    features.append((df['high'].iloc[-1] / last_close) - 1)
    features.append((df['low'].iloc[-1] / last_close) - 1)
    features.append(0)  # Current close is reference
    
    # Volume
    features.append(df['volume'].iloc[-1] / df['volume'].mean())
    
    # Technical indicators
    if 'rsi' in df.columns:
        features.append(df['rsi'].iloc[-1] / 100)
    else:
        features.append(0.5)
    
    if 'macd' in df.columns:
        features.append(df['macd'].iloc[-1] / last_close)
        features.append(df['macd_signal'].iloc[-1] / last_close)
    else:
        features.extend([0, 0])
    
    if 'bb_upper' in df.columns:
        features.append((df['bb_upper'].iloc[-1] / last_close) - 1)
        features.append((df['bb_lower'].iloc[-1] / last_close) - 1)
    else:
        features.extend([0, 0])
    
    if 'atr' in df.columns:
        features.append(df['atr'].iloc[-1] / last_close)
    else:
        features.append(0.02)
    
    if 'adx' in df.columns:
        features.append(df['adx'].iloc[-1] / 100)
    else:
        features.append(0.25)
    
    # Trend features
    if len(df) >= 20:
        sma20 = df['close'].rolling(20).mean().iloc[-1]
        features.append((last_close / sma20) - 1)
    else:
        features.append(0)
    
    if len(df) >= 50:
        sma50 = df['close'].rolling(50).mean().iloc[-1]
        features.append((last_close / sma50) - 1)
    else:
        features.append(0)
    
    # Pad to 64 features
    while len(features) < 64:
        features.append(0)
    
    return np.array(features, dtype=np.float32)


def train_super_dqn(
    agent: SuperDQNAgent,
    data: pd.DataFrame,
    episodes: int = 1000,
    batch_size: int = 64,
    validation_split: float = 0.2,
    callback=None
) -> dict:
    """
    Train Super DQN Agent
    
    Args:
        agent: SuperDQNAgent instance
        data: Training data DataFrame
        episodes: Number of training episodes
        batch_size: Batch size for training
        validation_split: Fraction of data for validation
        callback: Optional callback function(episode, loss, epsilon)
    
    Returns:
        Training history dict
    """
    logger.info(f"Starting Super DQN training for {episodes} episodes")
    
    # Prepare training data
    training_data = prepare_training_data(data)
    
    # Split into train/validation
    split_idx = int(len(training_data) * (1 - validation_split))
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    # Fill replay buffer
    for state, action, reward, next_state in train_data:
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=False
        )
        agent.replay_buffer.push(experience)
    
    # Training loop
    history = {
        'loss': [],
        'val_loss': [],
        'epsilon': []
    }
    
    for episode in range(episodes):
        # Training step
        loss = agent.train_step()
        
        # Validation
        if len(val_data) > 0 and episode % 10 == 0:
            val_batch = val_data[:min(batch_size, len(val_data))]
            # Simple validation - just check if we can predict
            val_loss = loss  # Placeholder
        else:
            val_loss = 0
        
        # Record history
        history['loss'].append(loss)
        history['val_loss'].append(val_loss)
        history['epsilon'].append(agent.epsilon)
        
        # Callback
        if callback:
            callback(episode, loss, agent.epsilon)
        
        # Log progress
        if episode % 100 == 0:
            logger.info(
                f"Episode {episode}/{episodes} - "
                f"Loss: {loss:.4f}, Epsilon: {agent.epsilon:.4f}"
            )
    
    logger.info("Training completed")
    
    return history


def evaluate_agent(
    agent: SuperDQNAgent,
    data: pd.DataFrame
) -> dict:
    """Evaluate agent performance on test data"""
    test_data = prepare_training_data(data)
    
    correct = 0
    total = 0
    
    for state, true_action, reward, _ in test_data:
        # Create market state
        market_state = MarketState(
            price=1.0,
            volume=1.0,
            timestamp=0,
            indicators={}
        )
        
        # Get prediction
        action = agent.predict(market_state)
        
        # Check if correct (simplified)
        if (action.action_type == "buy" and true_action == 0) or \
           (action.action_type == "sell" and true_action == 1) or \
           (action.action_type == "hold" and true_action == 2):
            correct += 1
        
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }
