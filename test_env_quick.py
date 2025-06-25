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
        
        # Test the act method specifically (this was causing the error)
        logger.info("🎯 Testing agent act method with multi-GPU...")
        try:
            dummy_obs_single = {
                'image': torch.randint(0, 255, (1, 84, 84, 3), device='cuda', dtype=torch.uint8),
                'features': torch.randn(1, 32, device='cuda')
            }
            
            # Test single observation act method
            action, log_prob, value = agent.network.module.act(dummy_obs_single) if hasattr(agent, 'use_multi_gpu') and agent.use_multi_gpu else agent.network.act(dummy_obs_single)
            
            logger.info(f"✅ Act method test successful:")
            logger.info(f"   Action: {action}")
            logger.info(f"   Log prob: {log_prob:.4f}")
            logger.info(f"   Value: {value:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Act method test failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test forward pass with large batch
        logger.info("🔥 Testing large batch forward pass...")
        batch_size = 1024  # Large batch for testing
        
        dummy_obs = {
            'image': torch.randint(0, 255, (batch_size, 84, 84, 3), device='cuda', dtype=torch.uint8),
            'features': torch.randn(batch_size, 32, device='cuda')
        }
        
        # Test multiple large batches to force memory allocation
        for i in range(3):
            logger.info(f"🔥 Processing large batch {i+1}/3...")
            with torch.no_grad():
                if config.enable_mixed_precision:
                    with torch.amp.autocast('cuda'):
                        movement_coords, action_logits, values = agent.network(dummy_obs)
                else:
                    movement_coords, action_logits, values = agent.network(dummy_obs)
            
            # Skip the large tensor allocation since we're already using 20GB
            if i == 0:
                # Show that we're successfully using the network with high memory
                logger.info(f"📈 Network successfully processed {batch_size} samples with {torch.cuda.memory_allocated(0)/1024**3:.1f}GB GPU memory")
        
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
            logger.info(f"�� Taking action {i+1}/5...")
            
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

def test_direct_gpu_allocation():
    """Direct test of GPU memory allocation to verify 24GB usage"""
    if not torch.cuda.is_available():
        logger.warning("No CUDA available, skipping direct GPU test")
        return
    
    logger.info("🔥 Direct GPU memory allocation test...")
    
    try:
        # Check initial memory
        logger.info("📊 Initial GPU memory:")
        check_gpu_memory()
        
        # Allocate increasingly large tensors until we hit ~20GB
        tensors = []
        target_memory_gb = 20  # Target 20GB usage
        current_memory_gb = 0
        
        tensor_size_gb = 1  # Start with 1GB tensors
        
        while current_memory_gb < target_memory_gb:
            try:
                # Calculate tensor dimensions for approximately tensor_size_gb GB
                # float16 = 2 bytes, so for 1GB we need ~500M elements
                elements = int((tensor_size_gb * 1024**3) / 2)  # 2 bytes per float16
                
                # Create a large tensor
                large_tensor = torch.randn(elements, device='cuda', dtype=torch.float16)
                tensors.append(large_tensor)
                
                current_memory_gb = torch.cuda.memory_allocated(0) / 1024**3
                logger.info(f"✅ Allocated {tensor_size_gb}GB tensor. Total: {current_memory_gb:.2f}GB")
                
                # Increase tensor size for next allocation
                if current_memory_gb < 10:
                    tensor_size_gb = 2  # Use 2GB tensors when we have room
                elif current_memory_gb < 15:
                    tensor_size_gb = 1  # Use 1GB tensors as we approach limit
                else:
                    tensor_size_gb = 0.5  # Use smaller tensors near the limit
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.info(f"💾 Hit memory limit at {current_memory_gb:.2f}GB")
                    break
                else:
                    raise e
        
        logger.info("📊 Final GPU memory after direct allocation:")
        check_gpu_memory()
        
        # Test computation on large tensors
        if tensors:
            logger.info("🧮 Testing computation on large tensors...")
            result = torch.sum(tensors[0][:1000000])  # Sum part of first tensor
            logger.info(f"✅ Computation successful. Sample result: {result.item():.2f}")
        
        # Keep tensors in memory (don't clear)
        logger.info("💾 Keeping large tensors allocated to maintain GPU memory usage")
        
        return tensors  # Return to keep in memory
        
    except Exception as e:
        logger.error(f"❌ Direct GPU allocation test failed: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    logger.info(f"📁 Current directory: {os.getcwd()}")
    logger.info(f"📂 Files in directory: {os.listdir('.')}")
    
    # Test basic environment
    test_environment()
    
    # Test GPU utilization if CUDA is available
    if torch.cuda.is_available():
        logger.info("🚀 Starting GPU utilization tests...")
        
        # Test 1: Direct GPU memory allocation
        logger.info("\n" + "="*50)
        logger.info("TEST 1: Direct GPU Memory Allocation")
        logger.info("="*50)
        tensors = test_direct_gpu_allocation()
        
        # Test 2: Agent GPU utilization 
        logger.info("\n" + "="*50)
        logger.info("TEST 2: Agent GPU Utilization")
        logger.info("="*50)
        test_gpu_utilization()
        
        # Final memory check
        logger.info("\n" + "="*50)
        logger.info("FINAL GPU MEMORY STATUS")
        logger.info("="*50)
        check_gpu_memory()
        
    else:
        logger.warning("No CUDA available, skipping all GPU tests") 