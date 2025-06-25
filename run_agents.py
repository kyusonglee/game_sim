#!/usr/bin/env python3
"""
Main runner script for the Outdoor Robot Simulator AI agents.

This script provides two modes:
1. LLM Agent - Uses ChatGPT o3-mini to play the game via vision and state analysis
2. Deep RL Agent - Uses PPO reinforcement learning to train and play the game

Usage:
    python run_agents.py --mode llm --api-key YOUR_OPENAI_KEY
    python run_agents.py --mode rl --action train
    python run_agents.py --mode rl --action evaluate --model-path ppo_model_final.pth
"""

import argparse
import asyncio
import sys
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_llm_agent(api_key: str, game_url: str = "http://localhost:8000", 
                  max_actions: int = 500, action_delay: float = 3.0):
    """Run the LLM-based agent"""
    try:
        from llm_agent import FarmRobotLLMAgent
        
        print("🤖 Starting LLM Agent (ChatGPT o3-mini)")
        print(f"Game URL: {game_url}")
        print(f"Max actions: {max_actions}")
        print(f"Action delay: {action_delay}s")
        print("-" * 50)
        
        # Ensure server is running
        actual_game_url = ensure_server_running(game_url)
        
        agent = FarmRobotLLMAgent(api_key)
        
        async def run_agent():
            try:
                # Setup browser and navigate to game
                agent.setup_browser(actual_game_url)
                
                # Start playing
                await agent.play_game(max_actions=max_actions, action_delay=action_delay)
                
            except Exception as e:
                logger.error(f"Error running LLM agent: {e}")
            finally:
                agent.cleanup()
        
        # Run the async agent
        asyncio.run(run_agent())
        
    except ImportError as e:
        print(f"❌ Error importing LLM agent dependencies: {e}")
        print("Please install required packages:")
        print("pip install openai selenium pillow")
        sys.exit(1)

def run_rl_agent_training(game_url: str = "http://localhost:8000", 
                         headless: bool = True, config_overrides: dict = None):
    """Run the Deep RL agent training"""
    try:
        from ppo_trainer import PPOAgent, TrainingConfig
        from rl_environment import FarmRobotEnvironment
        
        print("🧠 Starting Deep RL Agent Training (PPO)")
        print(f"Game URL: {game_url}")
        print(f"Headless mode: {headless}")
        print("-" * 50)
        
        # Ensure server is running
        actual_game_url = ensure_server_running(game_url)
        
        # Create GPU-optimized training configuration for TITAN RTX
        config = TrainingConfig(
            learning_rate=3e-4,
            max_episodes=5000,
            max_steps_per_episode=2000,
            update_frequency=32768,      # GPU optimized - 4x larger buffer
            batch_size=2048,             # GPU optimized - 8x larger batch size
            save_frequency=50,
            enable_mixed_precision=True,  # GPU optimization
            gpu_memory_fraction=0.9      # Use 90% of GPU memory for TITAN RTX
        )
        
        # Apply any config overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    print(f"Config override: {key} = {value}")
        
        print(f"Using device: {config.device}")
        
        # Create environment
        env = FarmRobotEnvironment(game_url=actual_game_url, headless=headless)
        
        # Create agent
        agent = PPOAgent(env, config)
        
        try:
            # Train agent
            agent.train()
            
            # Evaluate final performance
            print("\n🎯 Evaluating final performance...")
            agent.evaluate(num_episodes=5)
            
        except KeyboardInterrupt:
            print("\n⏹️ Training interrupted by user")
            agent.save_model("ppo_model_interrupted.pth")
        finally:
            env.close()
            
    except ImportError as e:
        print(f"❌ Error importing RL agent dependencies: {e}")
        print("Please install required packages:")
        print("pip install torch torchvision gymnasium opencv-python selenium matplotlib")
        sys.exit(1)

def run_rl_agent_evaluation(model_path: str, game_url: str = "http://localhost:8000",
                           headless: bool = False, num_episodes: int = 10):
    """Run the Deep RL agent evaluation"""
    try:
        from ppo_trainer import PPOAgent, TrainingConfig
        from rl_environment import FarmRobotEnvironment
        
        print("🎮 Starting Deep RL Agent Evaluation")
        print(f"Model: {model_path}")
        print(f"Game URL: {game_url}")
        print(f"Episodes: {num_episodes}")
        print("-" * 50)
        
        # Ensure server is running
        actual_game_url = ensure_server_running(game_url)
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            sys.exit(1)
        
        # Create environment
        env = FarmRobotEnvironment(game_url=actual_game_url, headless=headless)
        
        # Create agent with dummy config (will be overridden when loading)
        config = TrainingConfig()
        agent = PPOAgent(env, config)
        
        try:
            # Load model
            agent.load_model(model_path)
            
            # Evaluate performance
            avg_reward, avg_score = agent.evaluate(num_episodes=num_episodes)
            
            print(f"\n📊 Evaluation Results:")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Average Score: {avg_score:.2f}")
            
        finally:
            env.close()
            
    except ImportError as e:
        print(f"❌ Error importing RL agent dependencies: {e}")
        print("Please install required packages:")
        print("pip install torch torchvision gymnasium opencv-python selenium matplotlib")
        sys.exit(1)

def check_server_running(url: str) -> bool:
    """Check if server is running at the given URL"""
    import urllib.request
    try:
        urllib.request.urlopen(url, timeout=2)
        return True
    except:
        return False

def ensure_server_running(game_url: str = "http://localhost:8000") -> str:
    """Ensure the game server is running, start it if needed"""
    if check_server_running(game_url):
        print(f"✅ Server already running at {game_url}")
        return game_url
    
    print(f"🔄 Server not running, attempting to start...")
    
    # Try to start the enhanced server first
    try:
        import subprocess
        import time
        
        # Start server.py in background
        server_process = subprocess.Popen(
            [sys.executable, "server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit for server to start
        time.sleep(3)
        
        if check_server_running(game_url):
            print(f"✅ Enhanced server started at {game_url}")
            return game_url
        else:
            server_process.terminate()
            
    except Exception as e:
        print(f"⚠️ Could not start enhanced server: {e}")
    
    # Fallback to simple server
    return start_local_server()

def start_local_server():
    """Start a simple local HTTP server for the game"""
    import http.server
    import socketserver
    import threading
    import time
    
    PORT = 8000
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory='.', **kwargs)
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"🌐 Starting simple server at http://localhost:{PORT}")
            
            def serve():
                httpd.serve_forever()
            
            server_thread = threading.Thread(target=serve, daemon=True)
            server_thread.start()
            
            time.sleep(2)  # Give server time to start
            return f"http://localhost:{PORT}"
            
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"⚠️ Port {PORT} already in use, assuming server is running")
            return f"http://localhost:{PORT}"
        else:
            raise

def main():
    parser = argparse.ArgumentParser(
        description="Run AI agents for the Outdoor Robot Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run LLM agent
  python run_agents.py --mode llm --api-key sk-your-openai-key
  
  # Train RL agent
  python run_agents.py --mode rl --action train
  
  # Evaluate trained RL agent
  python run_agents.py --mode rl --action evaluate --model-path ppo_model_final.pth
  
  # Train RL agent with custom settings
  python run_agents.py --mode rl --action train --headless --lr 1e-4 --episodes 1000
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['llm', 'rl'], 
        required=True,
        help='Choose agent mode: llm (LLM-based) or rl (Deep RL)'
    )
    
    parser.add_argument(
        '--action',
        choices=['train', 'evaluate'],
        default='train',
        help='Action for RL mode: train or evaluate (default: train)'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='OpenAI API key (required for LLM mode)'
    )
    
    parser.add_argument(
        '--game-url',
        type=str,
        default='auto',
        help='Game URL (default: auto-start local server)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to trained model file (for RL evaluate mode)'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run browser in headless mode (faster for training)'
    )
    
    parser.add_argument(
        '--max-actions',
        type=int,
        default=500,
        help='Maximum actions for LLM agent (default: 500)'
    )
    
    parser.add_argument(
        '--action-delay',
        type=float,
        default=3.0,
        help='Delay between actions for LLM agent in seconds (default: 3.0)'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of evaluation episodes (default: 10)'
    )
    
    # RL training hyperparameters
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--gamma', type=float, help='Discount factor')
    parser.add_argument('--max-episodes', type=int, help='Maximum training episodes')
    
    args = parser.parse_args()
    
    # Setup game URL
    if args.game_url == 'auto':
        game_url = start_local_server()
    else:
        game_url = args.game_url
    
    # Check if game files exist
    required_files = ['index.html', 'game.js']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Missing game files: {missing_files}")
        print("Please ensure the game files are in the current directory.")
        sys.exit(1)
    
    # Run appropriate agent
    if args.mode == 'llm':
        if not args.api_key:
            print("❌ OpenAI API key required for LLM mode")
            print("Use: --api-key YOUR_OPENAI_KEY")
            sys.exit(1)
        
        run_llm_agent(
            api_key=args.api_key,
            game_url=game_url,
            max_actions=args.max_actions,
            action_delay=args.action_delay
        )
        
    elif args.mode == 'rl':
        # Prepare config overrides
        config_overrides = {}
        if args.lr is not None:
            config_overrides['learning_rate'] = args.lr
        if args.gamma is not None:
            config_overrides['gamma'] = args.gamma
        if args.max_episodes is not None:
            config_overrides['max_episodes'] = args.max_episodes
        
        if args.action == 'train':
            run_rl_agent_training(
                game_url=game_url,
                headless=args.headless,
                config_overrides=config_overrides
            )
        elif args.action == 'evaluate':
            if not args.model_path:
                print("❌ Model path required for evaluation")
                print("Use: --model-path path/to/model.pth")
                sys.exit(1)
            
            run_rl_agent_evaluation(
                model_path=args.model_path,
                game_url=game_url,
                headless=args.headless,
                num_episodes=args.episodes
            )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1) 