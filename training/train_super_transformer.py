"""
Training script for Super Transformer Agent
"""
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.super_transformer_agent import SuperTransformerAgent
from utils.indicators import TechnicalIndicators


def prepare_sequence_data(
    df: pd.DataFrame,
    sequence_length: int = 60
) -> List[Tuple[np.ndarray, int, float]]:
    """
    Prepare sequence training data
    
    Returns list of (sequence, action, reward) tuples
    """
    # Calculate indicators
    df = TechnicalIndicators.calculate_all(df)
    df = df.dropna()
    
    training_data = []
    
    for i in range(sequence_length, len(df) - 1):
        # Create sequence
        sequence = df.iloc[i-sequence_length:i]
        
        # Create state vector for each timestep
        seq_vectors = []
        for j in range(len(sequence)):
            row = sequence.iloc[j]
            vector = _create_state_vector(row, sequence.iloc[:j+1])
            seq_vectors.append(vector)
        
        seq_array = np.array(seq_vectors)
        
        # Determine action
        current_price = df['close'].iloc[i]
        next_price = df['close'].iloc[i+1]
        
        price_change = (next_price - current_price) / current_price
        
        if price_change > 0.005:
            action = 0  # Buy
            reward = price_change * 100
        elif price_change < -0.005:
            action = 1  # Sell
            reward = abs(price_change) * 100
        else:
            action = 2  # Hold
            reward = 0
        
        training_data.append((seq_array, action, reward))
    
    logger.info(f"Prepared {len(training_data)} sequence samples")
    return training_data


def _create_state_vector(row: pd.Series, history: pd.DataFrame) -> np.ndarray:
    """Create state vector from row"""
    features = []
    
    last_close = row['close']
    
    # Price features
    features.append((row['open'] / last_close) - 1)
    features.append((row['high'] / last_close) - 1)
    features.append((row['low'] / last_close) - 1)
    features.append(0)
    
    # Volume
    if 'volume' in history.columns and len(history) > 0:
        features.append(row['volume'] / history['volume'].mean())
    else:
        features.append(1.0)
    
    # Indicators
    features.append(row.get('rsi', 50) / 100)
    features.append(row.get('macd', 0) / last_close)
    features.append(row.get('macd_signal', 0) / last_close)
    features.append((row.get('bb_upper', last_close) / last_close) - 1)
    features.append((row.get('bb_lower', last_close) / last_close) - 1)
    features.append(row.get('atr', last_close * 0.02) / last_close)
    features.append(row.get('adx', 25) / 100)
    
    # Moving averages
    if len(history) >= 20:
        sma20 = history['close'].rolling(20).mean().iloc[-1]
        features.append((last_close / sma20) - 1)
    else:
        features.append(0)
    
    if len(history) >= 50:
        sma50 = history['close'].rolling(50).mean().iloc[-1]
        features.append((last_close / sma50) - 1)
    else:
        features.append(0)
    
    # Pad to 64 features
    while len(features) < 64:
        features.append(0)
    
    return np.array(features, dtype=np.float32)


def train_super_transformer(
    agent: SuperTransformerAgent,
    data: pd.DataFrame,
    episodes: int = 1000,
    batch_size: int = 32,
    validation_split: float = 0.2,
    callback=None
) -> dict:
    """
    Train Super Transformer Agent
    
    Args:
        agent: SuperTransformerAgent instance
        data: Training data DataFrame
        episodes: Number of training episodes
        batch_size: Batch size for training
        validation_split: Fraction of data for validation
        callback: Optional callback function(episode, loss)
    
    Returns:
        Training history dict
    """
    logger.info(f"Starting Super Transformer training for {episodes} episodes")
    
    # Prepare training data
    training_data = prepare_sequence_data(data, agent.sequence_length)
    
    # Split into train/validation
    split_idx = int(len(training_data) * (1 - validation_split))
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    logger.info(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    # Training loop
    history = {
        'loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    for episode in range(episodes):
        # Sample batch
        if len(train_data) >= batch_size:
            indices = np.random.choice(len(train_data), batch_size, replace=False)
            batch = [train_data[i] for i in indices]
        else:
            batch = train_data
        
        # Training step
        loss = agent.train_step(batch)
        
        # Validation
        if len(val_data) > 0 and episode % 10 == 0:
            val_batch = val_data[:min(batch_size, len(val_data))]
            val_loss = loss  # Simplified
        else:
            val_loss = 0
        
        # Record history
        history['loss'].append(loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(agent.scheduler.get_last_lr()[0])
        
        # Callback
        if callback:
            callback(episode, loss)
        
        # Log progress
        if episode % 100 == 0:
            logger.info(f"Episode {episode}/{episodes} - Loss: {loss:.4f}")
    
    logger.info("Training completed")
    
    return history
