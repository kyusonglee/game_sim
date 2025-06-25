#!/usr/bin/env python3
"""
Easy runner script for the Robot Simulator AI Agent
"""

import sys
import subprocess
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['pygame', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please install manually:")
            print(f"   pip install {' '.join(missing_packages)}")
            return False
    
    return True

def run_simple_game():
    """Run the original human-playable game"""
    print("🎮 Starting the human-playable Robot Simulator...")
    print("Controls:")
    print("  Arrow keys: Move robot")
    print("  P: Pick up object")
    print("  D: Drop object")
    print("  R: Restart level")
    print("\nPress Ctrl+C to exit\n")
    
    try:
        import simple
        simple.main()
    except KeyboardInterrupt:
        print("\n👋 Game stopped by user")
    except Exception as e:
        print(f"❌ Error running game: {e}")

def run_web_game():
    """Start the web server for the browser-based game"""
    print("🌐 Starting web server for browser-based game...")
    
    try:
        import server
        server.main()
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error running server: {e}")

def train_agent(episodes=50, no_render=False):
    """Train the AI agent"""
    print(f"🤖 Training AI agent for {episodes} episodes...")
    
    if not check_dependencies():
        return
    
    try:
        from simple_agent import train_agent
        train_agent(episodes=episodes, render=not no_render)
    except Exception as e:
        print(f"❌ Error training agent: {e}")

def watch_agent(episodes=5):
    """Watch the trained agent play"""
    print(f"👀 Watching trained agent play for {episodes} episodes...")
    
    if not check_dependencies():
        return
    
    try:
        from simple_agent import play_with_agent
        play_with_agent(episodes=episodes)
    except Exception as e:
        print(f"❌ Error running agent: {e}")

def run_web_agent():
    """Run the advanced web-based agent"""
    print("🤖 Starting advanced web-based AI agent...")
    print("Note: This requires Chrome/Chromium and chromedriver to be installed")
    
    try:
        import agent
        agent.main()
    except ImportError as e:
        print(f"❌ Missing dependencies for web agent: {e}")
        print("Install with: pip install selenium opencv-python")
    except Exception as e:
        print(f"❌ Error running web agent: {e}")

def show_help():
    """Show help information"""
    print("""
🤖 Robot House Simulator - AI Agent Runner

Available modes:

1. play-human     - Play the game manually (Python/Pygame version)
2. play-web       - Start web server for browser game
3. train-agent    - Train the AI agent (Q-learning)
4. watch-agent    - Watch trained agent play
5. web-agent      - Run advanced web-based agent (requires Chrome)

Examples:
  python run_agent.py play-human              # Play the game yourself
  python run_agent.py train-agent --episodes 100  # Train for 100 episodes
  python run_agent.py watch-agent --episodes 3    # Watch agent play 3 games
  python run_agent.py train-agent --no-render     # Train without graphics (faster)

For the web version:
  python run_agent.py play-web               # Start server, then visit http://localhost:8000
  python run_agent.py web-agent              # Run web-based AI agent

The game involves:
- Controlling a robot in a house with multiple rooms
- Picking up objects and moving them to target locations
- Learning optimal strategies through trial and error

The AI agent uses Q-learning to gradually improve its performance.
    """)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Robot Simulator AI Agent Runner")
    
    subparsers = parser.add_subparsers(dest='mode', help='Available modes')
    
    # Human play mode
    subparsers.add_parser('play-human', help='Play the game manually')
    
    # Web game mode
    subparsers.add_parser('play-web', help='Start web server for browser game')
    
    # Training mode
    train_parser = subparsers.add_parser('train-agent', help='Train the AI agent')
    train_parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    train_parser.add_argument('--no-render', action='store_true', help='Disable graphics for faster training')
    
    # Watch mode
    watch_parser = subparsers.add_parser('watch-agent', help='Watch trained agent play')
    watch_parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to watch')
    
    # Web agent mode
    subparsers.add_parser('web-agent', help='Run advanced web-based agent')
    
    # Help
    subparsers.add_parser('help', help='Show detailed help')
    
    args = parser.parse_args()
    
    if not args.mode:
        show_help()
        return
    
    if args.mode == 'play-human':
        run_simple_game()
    elif args.mode == 'play-web':
        run_web_game()
    elif args.mode == 'train-agent':
        train_agent(episodes=args.episodes, no_render=args.no_render)
    elif args.mode == 'watch-agent':
        watch_agent(episodes=args.episodes)
    elif args.mode == 'web-agent':
        run_web_agent()
    elif args.mode == 'help':
        show_help()
    else:
        print(f"❌ Unknown mode: {args.mode}")
        show_help()

if __name__ == "__main__":
    main() 