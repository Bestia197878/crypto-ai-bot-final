"""
Generate AI Agent Predictions and save to database
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils.database import DatabaseManager
from agents.super_dqn_agent import SuperDQNAgent
from agents.lstm_agent import LSTMAgent
from agents.super_transformer_agent import SuperTransformerAgent
from agents.super_ensemble_agent import SuperEnsembleAgent
from agents.base_agent import MarketState
import numpy as np
from datetime import datetime

def generate_agent_predictions(symbol="BTCUSDT"):
    """Generate predictions from all agents and save to database"""
    db = DatabaseManager("data/trading.db")
    
    print(f"🤖 Generating AI predictions for {symbol}...")
    
    # Create sample market state
    market_state = MarketState(
        price=45000.0 + np.random.uniform(-2000, 2000),
        volume=np.random.uniform(1000, 2000),
        timestamp=int(datetime.now().timestamp()),
        indicators={
            'rsi': np.random.uniform(40, 60),
            'atr': np.random.uniform(800, 1200),
            'sma_20': 44000.0,
            'ema_12': 44500.0,
            'volume_sma': 1500.0,
            'trend_strength': np.random.uniform(0.5, 0.9),
            'macd': np.random.uniform(-100, 100),
            'macd_signal': np.random.uniform(-50, 50)
        }
    )
    
    # Initialize agents
    agents = {
        'SuperDQN': SuperDQNAgent(state_size=10, action_size=3, device='cpu'),
        'LSTM': LSTMAgent(state_size=10, action_size=3, device='cpu'),
        'SuperTransformer': SuperTransformerAgent(state_size=10, action_size=3, device='cpu'),
        'Ensemble': SuperEnsembleAgent(state_size=10, action_size=3, device='cpu')
    }
    
    # Build models and generate predictions
    predictions = []
    
    for agent_name, agent in agents.items():
        try:
            # Build model if needed
            if hasattr(agent, 'build_model'):
                agent.model = agent.build_model()
            
            # Set training flag for DQN and Ensemble sub-agents
            if not hasattr(agent, 'training'):
                agent.training = False
            
            # For Ensemble, also set training on sub-agents
            if agent_name == 'Ensemble' and hasattr(agent, 'agents'):
                for sub_name, sub_agent in agent.agents.items():
                    if not hasattr(sub_agent, 'training'):
                        sub_agent.training = False
                    if hasattr(sub_agent, 'build_model') and (not hasattr(sub_agent, 'model') or sub_agent.model is None):
                        sub_agent.model = sub_agent.build_model()
            
            # Generate prediction
            action = agent.predict(market_state)
            
            prediction = {
                'agent_name': agent_name,
                'symbol': symbol,
                'action': action.action_type.upper(),
                'confidence': action.confidence,
                'price': action.price,
                'quantity': action.quantity,
                'metadata': {
                    'stop_loss': action.stop_loss,
                    'take_profit': action.take_profit,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            # Save to database
            db.save_prediction(prediction)
            predictions.append(prediction)
            
            print(f"  ✅ {agent_name}: {action.action_type.upper()} (confidence: {action.confidence:.2%})")
            
        except Exception as e:
            print(f"  ❌ {agent_name}: Error - {e}")
    
    print(f"\n💾 Saved {len(predictions)} predictions to database")
    return predictions

def generate_multiple_predictions(count=10, symbol="BTCUSDT"):
    """Generate multiple prediction cycles"""
    print(f"\n🔁 Generating {count} prediction cycles...\n")
    
    all_predictions = []
    for i in range(count):
        print(f"Cycle {i+1}/{count}")
        preds = generate_agent_predictions(symbol)
        all_predictions.extend(preds)
        print()
    
    print(f"✅ Total predictions generated: {len(all_predictions)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate AI Agent Predictions')
    parser.add_argument('--count', type=int, default=5, help='Number of prediction cycles')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')
    
    args = parser.parse_args()
    
    generate_multiple_predictions(count=args.count, symbol=args.symbol)
