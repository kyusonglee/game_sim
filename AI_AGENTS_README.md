# 🤖 AI Agents for Outdoor Robot Simulator

This project implements two advanced AI approaches to play the Outdoor Robot Front Yard Management game:

1. **LLM Agent** - Uses ChatGPT o3-mini with vision capabilities for intelligent gameplay
2. **Deep RL Agent** - Uses Proximal Policy Optimization (PPO) for autonomous learning

## 🎯 Game Overview

The outdoor robot simulator is a front yard management game where an AI robot must:
- 📦 Deliver packages to the house through the front door
- 📸 Take daily photos of plants to monitor health
- 🐿️ Warn animals away from plants (time-sensitive for bonus points)
- 🌱 Mow lawn when grass gets too long
- 🗑️ Sort trash into appropriate bins (trash/recycling/compost)
- 🌿 Remove weeds and compost them
- 🔋 Manage battery life by returning to charging station

## 🚀 Quick Start

### Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements_agents.txt
```

2. **Install ChromeDriver for Selenium:**
```bash
# On macOS
brew install chromedriver

# On Ubuntu/Debian
sudo apt-get install chromium-chromedriver

# On Windows
# Download from https://chromedriver.chromium.org/
```

3. **Get OpenAI API key (for LLM agent):**
   - Visit [OpenAI API](https://platform.openai.com/api-keys)
   - Create an API key
   - Keep it secure - you'll need it to run the LLM agent

### Running the Agents

**🚀 Automatic Server Setup**: Both agents will automatically start the game server if it's not running. No need to manually start `server.py` first!

#### 🧠 LLM Agent (ChatGPT o3-mini)

```bash
# Basic usage - server starts automatically!
python run_agents.py --mode llm --api-key YOUR_OPENAI_API_KEY

# With custom settings
python run_agents.py --mode llm \
    --api-key YOUR_API_KEY \
    --max-actions 1000 \
    --action-delay 1.5
```

#### 🤖 Deep RL Agent (PPO)

```bash
# Train a new agent
python run_agents.py --mode rl --action train

# Train with custom hyperparameters
python run_agents.py --mode rl --action train \
    --headless \
    --lr 1e-4 \
    --max-episodes 1000

# Evaluate a trained agent
python run_agents.py --mode rl --action evaluate \
    --model-path ppo_model_final.pth \
    --episodes 10
```

## 🏗️ Architecture

### LLM Agent Architecture

```
Game State Extraction → Image Capture → OpenAI API → Action Parsing → Game Execution
                     ↗                              ↘
              Feature Vector                    JSON Response
```

**Key Components:**
- **GameStateExtractor**: Extracts robot position, objects, scores via JavaScript
- **Vision System**: Captures game canvas as images for GPT-4V
- **Action Executor**: Translates LLM decisions into game controls
- **State Description**: Converts game state to natural language

**Supported Actions:**
- `move` - Navigate to coordinates
- `pickup` - Pick up nearby items
- `drop` - Drop carried items
- `photo` - Take plant photos
- `mow` - Mow grass areas
- `warn` - Warn animals
- `charge` - Go to charging station

### Deep RL Agent Architecture

```
Environment Wrapper → CNN Feature Extractor → Actor-Critic Network → PPO Training
                   ↗                        ↘
              Game State                    Action Selection
```

**Key Components:**
- **OutdoorRobotEnvironment**: Gymnasium-compatible environment wrapper
- **CNNFeatureExtractor**: Processes 84x84 RGB game screenshots
- **ActorCriticNetwork**: Combined policy and value networks
- **PPOBuffer**: Stores rollout data for training
- **PPOAgent**: Main training and evaluation logic

**State Space:**
- **Visual**: 84x84x3 RGB images
- **Features**: 32-dimensional vector including:
  - Robot position and battery (features 0-5)
  - Game state (score, level, day, time) (features 6-9)
  - **Current task information** (features 10, 27-31):
    - Active task count
    - Task type flags (package, animal, mow, weed, trash)
  - Object counts and distances (features 11-18)
  - Plant health metrics (features 19-21)
  - Vision and time-based features (features 22-26)

**Action Space:**
- Mixed continuous-discrete actions: [move_x, move_y, action_type]
- Movement: Both point navigation AND arrow keys for maximum flexibility
  - Point navigation: Continuous (x, y) coordinates (0-1 normalized) 
  - Arrow keys: Up, Down, Left, Right for precise local movement
- Actions: point_nav, arrow_up, arrow_down, arrow_left, arrow_right, pickup, drop, photo, mow, warn, charge, wait

**Reward Function:**
- +Points for score improvements
- +Rewards for task completion
- +Battery management bonuses
- -Penalties for inefficiency and low battery
- -Penalties for unhandled objects

## 📊 Training Results

### LLM Agent Performance
- **Response Time**: ~3-5 seconds per action (includes rate limiting)
- **Success Rate**: Dependent on prompt engineering and game complexity
- **Rate Limiting**: Built-in 2s minimum interval + automatic retry with exponential backoff
- **Strengths**: 
  - Excellent at understanding complex game rules
  - Good at prioritizing time-sensitive tasks
  - Natural language reasoning
  - Robust error handling and rate limit management
- **Weaknesses**:
  - API costs and latency
  - Slower due to rate limiting (but more reliable)

### Deep RL Agent Performance
- **Training Time**: ~2-4 hours on GPU for good performance
- **Convergence**: Usually within 1000-2000 episodes
- **Strengths**:
  - Fast inference (milliseconds)
  - Learns optimal strategies through trial and error
  - No API costs after training
- **Weaknesses**:
  - Requires significant training time
  - Black box decision making

## 🔧 Advanced Configuration

### LLM Agent Configuration

```python
# In llm_agent.py
agent = OutdoorRobotLLMAgent(
    api_key="your_key",
    model="gpt-4o-mini"  # or "gpt-4o" for better performance
)
```

### RL Agent Configuration

```python
# In ppo_trainer.py
config = TrainingConfig(
    learning_rate=3e-4,      # Learning rate
    gamma=0.99,              # Discount factor
    gae_lambda=0.95,         # GAE lambda
    clip_epsilon=0.2,        # PPO clip parameter
    value_loss_coef=0.5,     # Value loss coefficient
    entropy_coef=0.01,       # Entropy bonus
    max_episodes=5000,       # Training episodes
    update_frequency=2048,   # Steps per update
    batch_size=64,          # Minibatch size
    num_epochs=10           # PPO epochs per update
)
```

## 📈 Performance Monitoring

### Training Metrics
- Episode rewards and lengths
- Policy and value losses
- Success rates for different tasks
- Battery efficiency metrics

### Evaluation Metrics
- Average game score
- Task completion rates
- Battery management efficiency
- Time to complete all tasks

## 🚨 Troubleshooting

### OpenAI Rate Limits

If you encounter rate limit errors:

1. **Built-in Protection**: The LLM agent includes automatic rate limiting and retry logic
2. **Increase Delays**: Use longer action delays:
   ```bash
   python run_agents.py --mode llm --api-key YOUR_KEY --action-delay 5.0
   ```
3. **Check Your Limits**: Visit [OpenAI Rate Limits](https://platform.openai.com/account/rate-limits)
4. **Upgrade Plan**: Consider upgrading your OpenAI plan for higher limits

### Browser/ChromeDriver Issues

1. **Install ChromeDriver**: Ensure ChromeDriver is in your PATH
2. **Chrome Version**: Make sure Chrome and ChromeDriver versions match
3. **Permissions**: On macOS, you may need to allow ChromeDriver in Security & Privacy

### Server Connection Issues

The agents automatically start the server, but if you encounter issues:

1. **Manual Server**: Start server manually: `python server.py`
2. **Port Conflicts**: Kill processes on port 8000: `lsof -ti:8000 | xargs kill -9`
3. **Firewall**: Ensure localhost connections are allowed

## 🛠️ Customization

### Adding New Actions

1. **Update action space** in `rl_environment.py`
2. **Add action execution** in `_execute_action()`
3. **Update LLM prompt** in `llm_agent.py`
4. **Add reward shaping** for the new action

### Modifying Reward Function

Edit the `_execute_action()` method in `rl_environment.py`:

```python
# Example: Add bonus for specific behaviors
if new_state.get('some_condition'):
    reward += bonus_amount
```

### Custom Neural Network Architecture

Modify `ppo_networks.py` to change the CNN or add new network components:

```python
# Example: Add attention mechanism
class AttentionNetwork(nn.Module):
    def __init__(self):
        # Your custom architecture
        pass
```

## 🐛 Troubleshooting

### Common Issues

1. **ChromeDriver not found**
   ```bash
   # Make sure ChromeDriver is in PATH or install via package manager
   which chromedriver
   ```

2. **GPU memory issues**
   ```bash
   # Reduce batch size or use CPU
   python run_agents.py --mode rl --action train --batch-size 32
   ```

3. **Game not loading**
   ```bash
   # Check if local server is running
   curl http://localhost:8000
   ```

4. **OpenAI API errors**
   - Check API key validity
   - Verify sufficient credits
   - Check rate limits

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Research Applications

### Potential Extensions

1. **Multi-Agent Systems**: Multiple robots working together
2. **Curriculum Learning**: Progressive difficulty in training
3. **Transfer Learning**: Apply to similar games or real robots
4. **Imitation Learning**: Learn from human demonstrations
5. **Meta-Learning**: Adapt quickly to new game variants

### Academic Use Cases

- **Reinforcement Learning Research**: Benchmark new algorithms
- **Computer Vision**: Object detection and scene understanding
- **NLP Integration**: Natural language command following
- **Human-AI Interaction**: Study how AI agents interpret instructions

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements_agents.txt
pip install pytest black flake8

# Run tests
python -m pytest tests/

# Format code
black *.py

# Lint code
flake8 *.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models and API
- PyTorch team for the deep learning framework
- Gymnasium for the RL environment interface
- Selenium for browser automation
- The open-source community for inspiration and tools

## 📞 Support

- 🐛 Report bugs: [GitHub Issues](https://github.com/your-repo/issues)
- 💡 Feature requests: [GitHub Discussions](https://github.com/your-repo/discussions)
- 📧 Email: your-email@domain.com

---

**Happy Robot Training! 🤖🌱** 