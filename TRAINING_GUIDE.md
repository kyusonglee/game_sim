# 🚜 Farm Robot Simulator - Training Guide

This guide explains how to train AI agents for the Farm Robot Simulator using both LLM-based and Reinforcement Learning approaches.

## 🎯 Overview

The simulator supports two types of AI agents:
1. **LLM Agent** - Uses ChatGPT-4o-mini for real-time decision making
2. **RL Agent** - Uses PPO (Proximal Policy Optimization) for deep reinforcement learning

## 🔧 Setup Requirements

### Basic Requirements
```bash
# Install basic dependencies
pip install openai selenium pillow

# For RL training, install additional packages
pip install torch torchvision gymnasium opencv-python matplotlib
```

### WebDriver Setup
The agents use Selenium to control the browser. Make sure you have Chrome installed and optionally download ChromeDriver.

## 🤖 LLM Agent Training

### Quick Start
```bash
# Start the server
python server.py

# In another terminal, run the LLM agent
python run_agents.py --mode llm --api-key YOUR_OPENAI_API_KEY
```

### LLM Agent Features
- **Fixed Coordinate Bug**: Corrected canvas click calculations for proper navigation
- **Stuck Detection**: Automatically detects when robot repeats actions and suggests alternatives
- **Force Recovery**: After 5 repeated actions, forces different actions (warn animals, charge, explore)
- **Enhanced Prompts**: Better task prioritization and stuck recovery guidance

### LLM Training Options
```bash
# Standard training session
python run_agents.py --mode llm --api-key YOUR_KEY --max-actions 500 --action-delay 3.0

# Quick test run
python run_agents.py --mode llm --api-key YOUR_KEY --max-actions 50 --action-delay 1.0

# Slower, more deliberate training
python run_agents.py --mode llm --api-key YOUR_KEY --max-actions 1000 --action-delay 5.0
```

### LLM Agent Improvements Made
1. **Navigation Fix**: Fixed coordinate calculation bug that caused infinite movement loops
2. **Stuck Detection**: Tracks robot position and repeated actions
3. **Recovery Actions**: Forces alternative actions when stuck (warn, charge, explore)
4. **Better Context**: Provides stuck warnings to help LLM make better decisions

## 🧠 RL Agent Training (Deep Learning)

### Basic RL Training
```bash
# Train a new model (headless for speed)
python run_agents.py --mode rl --action train --headless --episodes 1000

# Train with custom parameters
python run_agents.py --mode rl --action train --headless --lr 3e-4 --episodes 2000

# Train with visual feedback (slower but easier to debug)
python run_agents.py --mode rl --action train --episodes 500
```

### RL Training Features
- **PPO Algorithm**: Uses Proximal Policy Optimization for stable training
- **Hybrid Action Space**: Combines continuous movement with discrete actions
- **Vision + State**: Uses both RGB images (84x84) and game state vector (32D)
- **Reward Engineering**: Rewards efficient farm management and animal protection

### RL Evaluation
```bash
# Evaluate a trained model
python run_agents.py --mode rl --action evaluate --model-path ppo_model_final.pth --episodes 10

# Evaluate with visual feedback
python run_agents.py --mode rl --action evaluate --model-path ppo_model_final.pth --episodes 5
```

## 🎮 Training Data Collection

### Automatic Data Collection
Both agents automatically collect training data:
- **Screenshots**: Captured for every action
- **Game State**: Complete environment state logged
- **Actions**: All robot actions and their outcomes
- **Performance**: Scores, battery usage, task completion

### Data Export
```bash
# Export training data for analysis
python analyze_training_data.py --export-visual
python analyze_training_data.py --export-ml
python analyze_training_data.py --generate-report
```

## 📊 Training Configuration

### RL Training Parameters
```python
TrainingConfig:
- learning_rate: 3e-4
- max_episodes: 5000
- max_steps_per_episode: 2000
- update_frequency: 2048
- save_frequency: 50
- batch_size: 64
- gamma: 0.99
- epsilon: 0.2
```

### Environment Parameters
```python
FarmRobotEnvironment:
- observation_space: RGB(84,84,3) + features(32)
- action_space: Hybrid (movement + discrete actions)
- reward_function: Farm management + animal protection
- episode_length: 2000 steps max
```

## 🎯 Training Tips

### For LLM Agent
1. **API Key**: Get OpenAI API key from https://platform.openai.com/
2. **Rate Limits**: Use `--action-delay` to avoid hitting API rate limits
3. **Cost Control**: Start with `--max-actions 50` for testing
4. **Monitoring**: Watch the console for stuck detection warnings

### For RL Agent
1. **Start Small**: Begin with 100-500 episodes for testing
2. **Use Headless**: `--headless` flag makes training much faster
3. **Monitor Rewards**: Training logs show reward progression
4. **Save Frequently**: Models are auto-saved every 50 episodes
5. **GPU**: Will automatically use CUDA if available

### Common Issues
1. **Port Conflicts**: Use `lsof -ti:8000 | xargs kill -9` to free port
2. **Browser Issues**: Chrome updates may require ChromeDriver updates
3. **Memory**: RL training uses significant memory, close other applications
4. **API Limits**: OpenAI has rate limits, adjust `--action-delay` accordingly

## 📈 Performance Metrics

### Key Metrics Tracked
- **Score**: Points earned from completing tasks
- **Efficiency**: Tasks completed per minute
- **Battery Management**: How well agent manages power
- **Animal Protection**: Speed of warning wild animals
- **Resource Usage**: Water and battery conservation

### Success Indicators
- **LLM Agent**: >80% action success rate, varied actions, task completion
- **RL Agent**: Increasing episode rewards, stable policy convergence
- **Both**: Improved farm management scores over time

## 🔬 Advanced Training

### Custom Reward Functions
Modify `rl_environment.py` to adjust reward signals:
```python
# Example: Increase animal warning importance
animal_protection_bonus = len(detected_animals) * 50
```

### Hyperparameter Tuning
```bash
# Test different learning rates
python run_agents.py --mode rl --action train --lr 1e-4 --episodes 1000
python run_agents.py --mode rl --action train --lr 1e-3 --episodes 1000

# Adjust exploration
python run_agents.py --mode rl --action train --gamma 0.95 --episodes 1000
```

### Multi-Agent Training
Train multiple models with different configurations and compare performance.

## 🚀 Next Steps

1. **Start Simple**: Begin with LLM agent to understand the environment
2. **Collect Data**: Let LLM agent generate training data
3. **Train RL Model**: Use collected data to improve RL training
4. **Compare Performance**: Test both approaches on the same tasks
5. **Iterate**: Adjust parameters based on results

## 📞 Troubleshooting

### Server Issues
```bash
# Check if server is running
curl http://localhost:8000

# Kill existing processes
lsof -ti:8000 | xargs kill -9

# Restart server
python server.py
```

### Agent Issues
```bash
# Test basic functionality
python -c "from llm_agent import FarmRobotLLMAgent; print('LLM OK')"
python -c "from rl_environment import FarmRobotEnvironment; print('RL OK')"
```

Happy training! 🌾🤖 