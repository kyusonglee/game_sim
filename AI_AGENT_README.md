# 🤖 AI Agent for Robot House Simulator

This repository contains multiple AI agents that can learn to play the Robot House Simulator game automatically. The game involves controlling a robot to navigate through rooms, pick up objects, and complete tasks.

## 🎮 Game Overview

The Robot House Simulator is a 2D game where:
- A robot navigates through multiple rooms (Living Room, Bedroom, Kitchen, Bathroom)
- Tasks involve picking up specific objects and moving them to target locations
- The robot has realistic movement (forward/backward, left/right rotation)
- The game collects training data for AI development

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Basic dependencies (automatically installed by run_agent.py)
pip install pygame numpy

# For advanced web agent (optional)
pip install selenium opencv-python torch torchvision
```

### 2. Run the Agent

The easiest way to get started is with the runner script:

```bash
# Train the AI agent (recommended first step)
python run_agent.py train-agent --episodes 100

# Watch the trained agent play
python run_agent.py watch-agent --episodes 5

# Play the game manually to understand it
python run_agent.py play-human

# Start web server for browser version
python run_agent.py play-web
```

### 3. See Results

After training, you'll see:
- Success rate improvements over time
- Saved model file (`robot_agent_model.json`)
- Training statistics and progress

## 📋 Available Agents

### 1. Simple Agent (`simple_agent.py`)
- **Type**: Q-Learning (Reinforcement Learning)
- **Interface**: Direct Python game integration
- **Best for**: Learning RL basics, fast training
- **Requirements**: pygame, numpy

```bash
# Train the simple agent
python run_agent.py train-agent --episodes 50

# Watch it play
python run_agent.py watch-agent
```

### 2. Web Agent (`agent.py`)
- **Type**: Computer Vision + Path Planning
- **Interface**: Browser automation with Selenium
- **Best for**: Web game interaction, computer vision
- **Requirements**: Chrome, chromedriver, selenium, opencv

```bash
# Run the web agent (requires Chrome)
python run_agent.py web-agent
```

## 🧠 How the AI Works

### Simple Agent (Q-Learning)

1. **State Representation**: Robot position, room, carried object, task info
2. **Actions**: move_forward, move_backward, rotate_left, rotate_right, pickup, drop
3. **Rewards**:
   - +100 for completing task
   - +20 for picking up correct object
   - +30 for dropping in correct room
   - +1 for moving closer to target
   - -10 for collisions
   - -0.1 per time step (encourages efficiency)

4. **Learning**: Q-table updated using Bellman equation
5. **Exploration**: Epsilon-greedy strategy (starts random, becomes more strategic)

### Web Agent (Computer Vision)

1. **Screenshot Analysis**: Uses OpenCV to detect objects by color
2. **Game State Extraction**: JavaScript execution to get precise game state
3. **Path Planning**: Calculates optimal routes to targets
4. **Action Execution**: Sends keyboard commands via Selenium

## 📊 Training Process

The agent learns through trial and error:

1. **Exploration Phase**: Agent tries random actions to discover the environment
2. **Learning Phase**: Agent updates its knowledge based on rewards received
3. **Exploitation Phase**: Agent uses learned knowledge to make better decisions
4. **Optimization**: Balance between exploration and exploitation gradually shifts

### Training Parameters

- **Learning Rate**: How fast the agent updates its knowledge (default: 0.1)
- **Discount Factor**: How much future rewards matter (default: 0.95)
- **Epsilon**: Exploration rate (starts high, decays over time)

## 🎯 Training Tips

### For Best Results:

1. **Start with 50-100 episodes** to see initial learning
2. **Use rendering every 5th episode** to monitor progress
3. **Train longer (200+ episodes)** for better performance
4. **Adjust parameters** if learning is too slow/fast

### Monitoring Progress:

```bash
# Train with visual feedback
python run_agent.py train-agent --episodes 100

# Train faster without graphics
python run_agent.py train-agent --episodes 200 --no-render
```

### Expected Learning Curve:

- **Episodes 1-20**: Random behavior, low success rate
- **Episodes 20-50**: Basic task understanding emerges
- **Episodes 50-100**: Consistent improvement in efficiency
- **Episodes 100+**: Near-optimal performance on simple tasks

## 📈 Performance Metrics

The agent tracks several metrics:

- **Success Rate**: Percentage of tasks completed successfully
- **Average Reward**: Higher is better (indicates efficient task completion)
- **Steps per Episode**: Lower is better (indicates efficiency)
- **Q-table Size**: Number of unique states learned

## 🔧 Customization

### Modify Rewards (`simple_agent.py`):

```python
def calculate_reward(self, prev_state, action, new_state, task_completed, collision):
    # Customize reward function here
    reward = 0
    if task_completed:
        reward += 100  # Increase for harder tasks
    # Add more reward shaping...
    return reward
```

### Adjust Learning Parameters:

```python
agent = SimpleAgent(
    learning_rate=0.05,    # Slower learning
    discount_factor=0.99,  # More future-focused
    epsilon=0.2           # More exploration
)
```

### Change Action Space:

```python
self.actions = [
    'move_forward', 'move_backward', 
    'rotate_left', 'rotate_right',
    'pickup', 'drop'
    # Add more actions if needed
]
```

## 🗃️ File Structure

```
simulator/
├── agent.py              # Advanced web-based agent
├── simple_agent.py       # Q-learning agent
├── run_agent.py          # Easy runner script
├── simple.py             # Python game implementation
├── game.js               # JavaScript web game
├── server.py             # Web server
├── analyze_training_data.py  # Data analysis tools
├── requirements.txt      # Python dependencies
└── robot_agent_model.json   # Saved agent model (after training)
```

## 🐛 Troubleshooting

### Common Issues:

1. **"No module named pygame"**
   ```bash
   pip install pygame numpy
   ```

2. **Agent not learning**
   - Increase episodes: `--episodes 200`
   - Check reward function
   - Verify game state extraction

3. **Web agent fails**
   - Install Chrome and chromedriver
   - Check if game server is running: `python run_agent.py play-web`

4. **Training too slow**
   - Use `--no-render` for faster training
   - Reduce episode length in code

5. **Agent gets stuck**
   - Add negative rewards for repeated actions
   - Implement better exploration strategy

## 🚀 Advanced Features

### Experiment with Different Algorithms:

1. **Deep Q-Learning (DQN)**: Use neural networks instead of Q-tables
2. **Policy Gradient**: Directly learn action policies
3. **Actor-Critic**: Combine value and policy learning
4. **Computer Vision**: Improve object detection accuracy

### Data Analysis:

```bash
# Analyze training data
python analyze_training_data.py
```

### Web Integration:

```bash
# Start web server
python run_agent.py play-web

# In another terminal, run web agent
python run_agent.py web-agent
```

## 🎓 Learning Resources

- **Reinforcement Learning**: Sutton & Barto's "Reinforcement Learning: An Introduction"
- **Q-Learning**: Understanding the temporal difference learning algorithm
- **Computer Vision**: OpenCV tutorials for object detection
- **Game AI**: Sebastian Thrun's AI for Robotics course

## 🤝 Contributing

Feel free to improve the agents by:

1. Adding new reward functions
2. Implementing different RL algorithms
3. Improving computer vision accuracy
4. Adding more sophisticated path planning
5. Creating better state representations

## 📝 License

This project is open source. Feel free to use and modify for educational purposes.

---

**Happy Learning! 🤖✨**

The agent will improve with time and practice, just like any good student. Watch it grow from random movements to strategic, efficient task completion! 