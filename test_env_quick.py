#!/usr/bin/env python3
"""
Quick test script to debug environment issues and GPU utilization
"""

import logging
import numpy as np
import urllib.request
import subprocess
import time
import os
import torch
from rl_environment import FarmRobotEnvironment
from ppo_trainer import PPOAgent, TrainingConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_memory():
    """Check GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_total = props.total_memory / 1024**3  # GB
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3  # GB
            
            logger.info(f"GPU {i} ({props.name}):")
            logger.info(f"  Total memory: {memory_total:.2f} GB")
            logger.info(f"  Allocated: {memory_allocated:.2f} GB ({memory_allocated/memory_total*100:.1f}%)")
            logger.info(f"  Cached: {memory_cached:.2f} GB ({memory_cached/memory_total*100:.1f}%)")
    else:
        logger.info("No CUDA GPUs available")

def test_gpu_utilization():
    """Test GPU utilization with large batch processing"""
    if not torch.cuda.is_available():
        logger.warning("No CUDA available, skipping GPU test")
        return
    
    logger.info("🚀 Testing GPU utilization with large batches...")
    
    try:
        # Create environment and agent with massive batch size
        env = FarmRobotEnvironment("http://localhost:8000", headless=True)
        
        config = TrainingConfig(
            batch_size=2048,  # Large batch
            update_frequency=32768,  # Large buffer
            device="cuda",
            enable_mixed_precision=True,
            gpu_memory_fraction=0.9
        )
        
        logger.info("📊 GPU memory before agent creation:")
        check_gpu_memory()
        
        # Create agent (this should allocate lots of GPU memory)
        agent = PPOAgent(env, config)
        
        logger.info("📊 GPU memory after agent creation:")
        check_gpu_memory()
        
        # Test forward pass with large batch
        logger.info("🔥 Testing large batch forward pass...")
        batch_size = 1024  # Large batch for testing
        
        dummy_obs = {
            'image': torch.randint(0, 255, (batch_size, 84, 84, 3), device='cuda', dtype=torch.uint8),
            'features': torch.randn(batch_size, 32, device='cuda')
        }
        
        with torch.no_grad():
            if config.enable_mixed_precision:
                with torch.cuda.amp.autocast():
                    movement_coords, action_logits, values = agent.network(dummy_obs)
            else:
                movement_coords, action_logits, values = agent.network(dummy_obs)
        
        logger.info(f"✅ Processed batch of {batch_size} samples")
        logger.info(f"   Movement coords shape: {movement_coords.shape}")
        logger.info(f"   Action logits shape: {action_logits.shape}")
        logger.info(f"   Values shape: {values.shape}")
        
        logger.info("📊 GPU memory after large batch processing:")
        check_gpu_memory()
        
        env.close()
        
    except Exception as e:
        logger.error(f"❌ GPU test failed: {e}")
        import traceback
        traceback.print_exc()

def check_server_running(url: str) -> bool:
    """Check if server is running at the given URL"""
    try:
        urllib.request.urlopen(url, timeout=2)
        return True
    except:
        return False

def start_server():
    """Start the HTTP server"""
    logger.info("🌐 Starting local HTTP server...")
    
    # Check if we have the game files
    if not os.path.exists("index.html") or not os.path.exists("game.js"):
        logger.error("❌ Game files (index.html, game.js) not found in current directory")
        return None
    
    try:
        # Start simple HTTP server
        process = subprocess.Popen(
            ["python", "-m", "http.server", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(3)
        
        if check_server_running("http://localhost:8000"):
            logger.info("✅ Server started successfully at http://localhost:8000")
            return process
        else:
            process.terminate()
            logger.error("❌ Server failed to start")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error starting server: {e}")
        return None

def test_environment():
    """Test the environment with a few simple actions"""
    logger.info("🧪 Testing environment setup...")
    
    server_process = None
    
    try:
        # Check if server is running, start if needed
        game_url = "http://localhost:8000"
        if not check_server_running(game_url):
            logger.info("🔄 Server not running, starting...")
            server_process = start_server()
            if not server_process:
                logger.error("❌ Cannot start server, aborting test")
                return
        else:
            logger.info("✅ Server already running")
        
        # Create environment
        env = FarmRobotEnvironment(game_url=game_url, headless=True)
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
        
        # Clean up server
        if server_process:
            try:
                server_process.terminate()
                logger.info("🛑 Server stopped")
            except:
                pass

if __name__ == "__main__":
    logger.info(f"📁 Current directory: {os.getcwd()}")
    logger.info(f"📂 Files in directory: {os.listdir('.')}")
    
    # Test basic environment
    test_environment()
    
    # Test GPU utilization if CUDA is available
    if torch.cuda.is_available():
        test_gpu_utilization()
    else:
        logger.warning("No CUDA available, skipping GPU utilization test") 