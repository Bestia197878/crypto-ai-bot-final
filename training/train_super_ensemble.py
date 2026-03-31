"""
Training script for Super Ensemble Agent
"""
import numpy as np
import pandas as pd
from typing import List, Dict
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.super_ensemble_agent import SuperEnsembleAgent
from agents.super_dqn_agent import SuperDQNAgent
from agents.super_transformer_agent import SuperTransformerAgent
from agents.lstm_agent import LSTMAgent
from training.train_super_dqn import prepare_training_data


def train_super_ensemble(
    agent: SuperEnsembleAgent,
    data: pd.DataFrame,
    episodes: int = 500,
    pretrain_agents: bool = True,
    callback=None
) -> dict:
    """
    Train Super Ensemble Agent
    
    This involves:
    1. Pre-training individual agents (optional)
    2. Training the meta-learner to combine predictions
    
    Args:
        agent: SuperEnsembleAgent instance
        data: Training data DataFrame
        episodes: Number of training episodes for meta-learner
        pretrain_agents: Whether to pre-train individual agents
        callback: Optional callback function
    
    Returns:
        Training history dict
    """
    logger.info(f"Starting Super Ensemble training")
    
    # Prepare training data
    training_data = prepare_training_data(data)
    
    # Step 1: Pre-train individual agents (if enabled)
    if pretrain_agents:
        logger.info("Pre-training individual agents...")
        
        # Train DQN agent
        logger.info("Training DQN agent...")
        for state, action, reward, next_state in training_data[:1000]:
            agent.dqn_agent.store_experience(
                state=type('obj', (object,), {'to_vector': lambda: state})(),
                action=action,
                reward=reward,
                next_state=type('obj', (object,), {'to_vector': lambda: next_state})(),
                done=False
            )
        
        for _ in range(100):
            agent.dqn_agent.train_step()
        
        # Train Transformer agent
        logger.info("Training Transformer agent...")
        # Prepare sequence data for transformer
        seq_data = []
        for i in range(60, min(len(training_data), 1000)):
            seq = np.array([training_data[j][0] for j in range(i-60, i)])
            seq_data.append((seq, training_data[i][1], training_data[i][2]))
        
        for _ in range(100):
            if len(seq_data) >= 32:
                batch = seq_data[:32]
                agent.transformer_agent.train_step(batch)
        
        # Train LSTM agent
        logger.info("Training LSTM agent...")
        for _ in range(100):
            if len(seq_data) >= 32:
                batch = seq_data[:32]
                agent.lstm_agent.train_step(batch)
    
    # Step 2: Train meta-learner
    logger.info("Training meta-learner...")
    
    history = {
        'meta_loss': [],
        'agent_performances': []
    }
    
    for episode in range(episodes):
        # Sample batch of market states
        batch_size = min(32, len(training_data))
        indices = np.random.choice(len(training_data), batch_size, replace=False)
        
        meta_batch = []
        
        for idx in indices:
            state, true_action, reward, _ = training_data[idx]
            
            # Get predictions from all agents
            market_state = type('obj', (object,), {
                'to_vector': lambda: state,
                'price': 1.0,
                'volume': 1.0,
                'indicators': {}
            })()
            
            agent_preds = {}
            for name, sub_agent in agent.agents.items():
                try:
                    pred = sub_agent.predict(market_state)
                    agent_preds[name] = {
                        'action': pred.action_type,
                        'confidence': pred.confidence
                    }
                except Exception:
                    agent_preds[name] = {'action': 'hold', 'confidence': 0.33}
            
            # Determine best agent (simplified - the one that would have picked correctly)
            best_agent_idx = true_action % 3  # Simplified mapping
            
            meta_batch.append((state, agent_preds, best_agent_idx, reward))
        
        # Train meta-learner
        loss = agent.train_step(meta_batch)
        
        history['meta_loss'].append(loss)
        
        # Update agent performances
        for name in agent.agents.keys():
            agent.update_performance(name, reward)
        
        history['agent_performances'].append(agent.get_agent_performance_report())
        
        # Callback
        if callback:
            callback(episode, loss)
        
        # Log progress
        if episode % 100 == 0:
            logger.info(f"Meta-learner episode {episode}/{episodes} - Loss: {loss:.4f}")
    
    logger.info("Ensemble training completed")
    
    return history


def evaluate_ensemble(
    agent: SuperEnsembleAgent,
    data: pd.DataFrame
) -> Dict:
    """Evaluate ensemble agent performance"""
    test_data = prepare_training_data(data)
    
    correct = 0
    total = 0
    
    agent_correct = {'dqn': 0, 'transformer': 0, 'lstm': 0}
    agent_total = {'dqn': 0, 'transformer': 0, 'lstm': 0}
    
    for state, true_action, reward, _ in test_data[:1000]:
        market_state = type('obj', (object,), {
            'to_vector': lambda: state,
            'price': 1.0,
            'volume': 1.0,
            'indicators': {}
        })()
        
        # Get ensemble prediction
        action = agent.predict(market_state)
        
        # Check if correct
        pred_action = 0 if action.action_type == "buy" else (1 if action.action_type == "sell" else 2)
        
        if pred_action == true_action:
            correct += 1
        
        total += 1
        
        # Check individual agents
        for name, sub_agent in agent.agents.items():
            try:
                pred = sub_agent.predict(market_state)
                sub_action = 0 if pred.action_type == "buy" else (1 if pred.action_type == "sell" else 2)
                
                if sub_action == true_action:
                    agent_correct[name] += 1
                agent_total[name] += 1
            except Exception:
                pass
    
    results = {
        'ensemble_accuracy': correct / total if total > 0 else 0,
        'ensemble_correct': correct,
        'total': total
    }
    
    for name in agent.agents.keys():
        if agent_total[name] > 0:
            results[f'{name}_accuracy'] = agent_correct[name] / agent_total[name]
        else:
            results[f'{name}_accuracy'] = 0
    
    return results
