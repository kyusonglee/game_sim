#!/usr/bin/env python3
"""
Quick test script to debug environment issues
"""

import logging
import numpy as np
from rl_environment import FarmRobotEnvironment

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_environment():
    """Test the environment with a few simple actions"""
    logger.info("🧪 Testing environment setup...")
    
    try:
        # Create environment
        env = FarmRobotEnvironment(game_url="http://localhost:8000", headless=True)
        logger.info("✅ Environment created successfully")
        
        # Reset environment
        logger.info("🔄 Resetting environment...")
        obs, info = env.reset()
        logger.info(f"✅ Environment reset. Obs shape: image={obs['image'].shape}, features={obs['features'].shape}")
        
        # Take 5 simple actions
        for i in range(5):
            logger.info(f"🎮 Taking action {i+1}/5...")
            
            # Simple action: center position + wait
            action = np.array([0.5, 0.5, 7])  # center x, center y, wait action
            
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                logger.info(f"✅ Action {i+1} completed. Reward: {reward:.3f}, Done: {terminated or truncated}")
                
                if terminated or truncated:
                    logger.info("🔄 Episode ended, resetting...")
                    obs, info = env.reset()
                    
            except Exception as step_error:
                logger.error(f"❌ Error in step {i+1}: {step_error}")
                break
        
        logger.info("🎉 Environment test completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            env.close()
            logger.info("🔒 Environment closed")
        except:
            pass

if __name__ == "__main__":
    test_environment() 