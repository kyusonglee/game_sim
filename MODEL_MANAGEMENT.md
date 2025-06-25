# 🗂️ Model Management Guide

## Where Models Are Saved

### RL Models (PPO Training)
All RL models are saved in the **current working directory** (`/Users/kyusonglee/Documents/proj/simulator`)

#### Model Files:
1. **Checkpoint Models:** `ppo_model_step_{N}.pth`
   - Saved every 50 training steps (configurable)
   - Examples: `ppo_model_step_50.pth`, `ppo_model_step_100.pth`
   - Contains: network weights, optimizer state, training progress

2. **Final Model:** `ppo_model_final.pth`
   - Saved when training completes or is interrupted (Ctrl+C)
   - Best model to use for evaluation

3. **Training Plots:** `training_progress.png`
   - Shows reward and loss curves
   - Generated automatically after training

### LLM Agent Data
The LLM agent doesn't save models (uses OpenAI API), but saves training data:
- **Session Logs:** `training_logs/` directory
- **Action History:** JSON files with game interactions

## Model Management Commands

### Check for Existing Models
```bash
# List all model files
ls -la *.pth *.png

# Show model details
ls -lah ppo_model_*.pth
```

### Training with Model Saves
```bash
# Train and save every 50 steps (default)
python run_agents.py --mode rl --action train --episodes 1000

# Train and save every 25 steps (more frequent saves)
python run_agents.py --mode rl --action train --episodes 1000 --save-freq 25
```

### Load and Evaluate Models
```bash
# Evaluate the final model
python run_agents.py --mode rl --action evaluate --model-path ppo_model_final.pth

# Evaluate a specific checkpoint
python run_agents.py --mode rl --action evaluate --model-path ppo_model_step_100.pth

# Evaluate with visual feedback
python run_agents.py --mode rl --action evaluate --model-path ppo_model_final.pth --episodes 5
```

### Resume Training
```bash
# Resume from a checkpoint (if implemented)
python run_agents.py --mode rl --action train --resume-from ppo_model_step_100.pth
```

## Model File Contents

Each `.pth` file contains:
```python
{
    'network_state_dict': torch.state_dict,     # Neural network weights
    'optimizer_state_dict': torch.state_dict,   # Optimizer state
    'training_step': int,                       # Current training step
    'config': TrainingConfig,                   # Training configuration
    'reward_history': List[float],              # Reward progression
    'loss_history': List[float]                 # Loss progression
}
```

## Best Practices

### During Training
1. **Monitor Progress:** Watch console logs for reward improvements
2. **Save Frequency:** Default 50 steps is good, increase for longer training
3. **Disk Space:** Models are ~10-50MB each, plan accordingly
4. **Backup Important Models:** Copy successful models to safe location

### Model Selection
1. **Latest != Best:** Use `training_progress.png` to find peak performance
2. **Evaluate Multiple:** Test different checkpoints to find best performer
3. **Stable Performance:** Look for models where rewards have stabilized

### File Organization
```
simulator/
├── ppo_model_final.pth          # Final trained model
├── ppo_model_step_50.pth        # Checkpoint models
├── ppo_model_step_100.pth
├── training_progress.png        # Training plots
├── training_logs/               # LLM agent logs
└── models/                      # Optional: organize models here
    ├── best_models/
    └── experiments/
```

## Troubleshooting

### Model Loading Errors
```bash
# Check if model file exists
ls -la ppo_model_final.pth

# Verify model is not corrupted
python -c "import torch; torch.load('ppo_model_final.pth'); print('Model OK')"
```

### Out of Disk Space
```bash
# Check disk usage
df -h

# Remove old checkpoints (keep final and recent ones)
rm ppo_model_step_[1-4]*.pth

# Compress old models
tar -czf old_models.tar.gz ppo_model_step_*.pth
```

### Training Interrupted
If training stops unexpectedly:
1. Check for `ppo_model_final.pth` (auto-saved on interruption)
2. Use latest checkpoint: `ppo_model_step_*.pth`
3. Restart training with same parameters

## Model Sharing

### Export Best Model
```bash
# Copy to results directory
mkdir -p results/trained_models
cp ppo_model_final.pth results/trained_models/farm_robot_v1.pth
cp training_progress.png results/trained_models/
```

### Import Model
```bash
# Use shared model
python run_agents.py --mode rl --action evaluate --model-path results/trained_models/farm_robot_v1.pth
```

Remember: Models are automatically saved during training, so you don't need to manually save them! 