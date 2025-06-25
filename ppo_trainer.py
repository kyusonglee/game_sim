import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import logging
from dataclasses import dataclass
from rl_environment import FarmRobotEnvironment
from ppo_networks import ActorCriticNetwork, PPOBuffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for RL training"""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_episodes: int = 10000
    max_steps_per_episode: int = 2000
    update_frequency: int = 2048
    num_epochs: int = 10
    batch_size: int = 64
    save_frequency: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class PPOAgent:
    """PPO Agent for training on the outdoor robot simulator"""
    
    def __init__(self, env, config: TrainingConfig):
        self.env = env
        self.config = config
        
        # Initialize network
        self.network = ActorCriticNetwork(
            env.observation_space, 
            env.action_space
        ).to(config.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)
        
        # Buffer
        self.buffer = PPOBuffer(
            config.update_frequency,
            env.observation_space,
            env.action_space,
            config.device
        )
        
        # Training tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.training_step = 0
        
        # Logging
        self.reward_history = []
        self.loss_history = []
        
    def collect_rollout(self):
        """Collect rollout data"""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(self.config.update_frequency):
            # Convert obs to tensors
            obs_tensor = {
                'image': torch.from_numpy(obs['image']).to(self.config.device),
                'features': torch.from_numpy(obs['features']).to(self.config.device)
            }
            
            # Get action
            action, log_prob, value = self.network.act(obs_tensor)
            
            # Take step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store in buffer
            self.buffer.store(obs, action, reward, value, log_prob, done)
            
            episode_reward += reward
            episode_length += 1
            
            obs = next_obs
            
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                logger.info(f"Episode finished - Reward: {episode_reward:.2f}, Length: {episode_length}, Score: {info.get('score', 0)}")
                
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
    
    def update_policy(self):
        """Update policy using PPO"""
        # Get buffer data
        buffer_data = self.buffer.get(self.config.gamma, self.config.gae_lambda)
        
        total_loss = 0
        num_updates = 0
        
        # Multiple epochs
        for epoch in range(self.config.num_epochs):
            # Mini-batch updates
            indices = torch.randperm(self.config.update_frequency)
            
            for start in range(0, self.config.update_frequency, self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_obs = {
                    'image': buffer_data['observations']['image'][batch_indices],
                    'features': buffer_data['observations']['features'][batch_indices]
                }
                batch_actions = buffer_data['actions'][batch_indices]
                batch_old_log_probs = buffer_data['old_log_probs'][batch_indices]
                batch_advantages = buffer_data['advantages'][batch_indices]
                batch_returns = buffer_data['returns'][batch_indices]
                
                # Forward pass
                movement_coords, action_type_logits, values = self.network(batch_obs)
                
                # Split actions
                batch_move_coords = batch_actions[:, :2]  # [batch, 2] - movement coordinates
                batch_action_types = batch_actions[:, 2].long()  # [batch] - action types
                
                # Compute losses for discrete actions (action type)
                dist = Categorical(logits=action_type_logits)
                new_log_probs = dist.log_prob(batch_action_types)
                entropy = dist.entropy().mean()
                
                # Movement loss (MSE between predicted and actual coordinates)
                movement_loss = F.mse_loss(movement_coords, batch_move_coords)
                
                # Policy loss (PPO clip) for discrete actions
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = batch_advantages * ratio
                surr2 = batch_advantages * torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss
                loss = (policy_loss + 
                       movement_loss * 0.1 +  # Weight for movement loss
                       self.config.value_loss_coef * value_loss - 
                       self.config.entropy_coef * entropy)
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_updates += 1
        
        avg_loss = total_loss / num_updates
        self.loss_history.append(avg_loss)
        
        logger.info(f"Policy updated - Average loss: {avg_loss:.4f}")
        
        # Reset buffer
        self.buffer.reset()
        self.training_step += 1
    
    def train(self):
        """Main training loop"""
        logger.info(f"Starting PPO training on {self.config.device}")
        
        episode_count = 0
        
        try:
            while episode_count < self.config.max_episodes:
                # Collect rollout
                self.collect_rollout()
                
                # Update policy
                self.update_policy()
                
                # Logging
                if len(self.episode_rewards) > 0:
                    avg_reward = np.mean(self.episode_rewards)
                    avg_length = np.mean(self.episode_lengths)
                    self.reward_history.append(avg_reward)
                    
                    logger.info(f"Training step {self.training_step}: "
                              f"Avg reward: {avg_reward:.2f}, "
                              f"Avg length: {avg_length:.1f}")
                
                # Save model
                if self.training_step % self.config.save_frequency == 0:
                    self.save_model(f"ppo_model_step_{self.training_step}.pth")
                
                episode_count += len(self.episode_rewards)
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        logger.info("Training completed")
        self.save_model("ppo_model_final.pth")
        self.plot_training_progress()
    
    def save_model(self, filename: str):
        """Save model and training state"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'config': self.config,
            'reward_history': self.reward_history,
            'loss_history': self.loss_history
        }, filename)
        logger.info(f"Model saved to {filename}")
    
    def load_model(self, filename: str):
        """Load model and training state"""
        checkpoint = torch.load(filename, map_location=self.config.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.reward_history = checkpoint.get('reward_history', [])
        self.loss_history = checkpoint.get('loss_history', [])
        logger.info(f"Model loaded from {filename}")
    
    def plot_training_progress(self):
        """Plot training progress"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Reward plot
        ax1.plot(self.reward_history)
        ax1.set_title('Average Episode Reward')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(self.loss_history)
        ax2.set_title('Training Loss')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.show()
    
    def evaluate(self, num_episodes: int = 10):
        """Evaluate trained agent"""
        logger.info(f"Evaluating agent for {num_episodes} episodes")
        
        episode_rewards = []
        episode_scores = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            
            while True:
                obs_tensor = {
                    'image': torch.from_numpy(obs['image']).to(self.config.device),
                    'features': torch.from_numpy(obs['features']).to(self.config.device)
                }
                
                action, _, _ = self.network.act(obs_tensor, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_scores.append(info.get('score', 0))
            logger.info(f"Evaluation episode {episode + 1}: Reward={episode_reward:.2f}, Score={info.get('score', 0)}")
        
        avg_reward = np.mean(episode_rewards)
        avg_score = np.mean(episode_scores)
        logger.info(f"Evaluation complete - Average reward: {avg_reward:.2f}, Average score: {avg_score:.2f}")
        
        return avg_reward, avg_score

def main():
    """Main function for training the RL agent"""
    print("=== Outdoor Robot Deep RL Agent ===")
    
    # Training configuration
    config = TrainingConfig(
        learning_rate=3e-4,
        max_episodes=5000,
        max_steps_per_episode=2000,
        update_frequency=2048,
        save_frequency=50
    )
    
    print(f"Using device: {config.device}")
    
    # Create environment
    env = FarmRobotEnvironment(headless=False)  # Set to True for faster training
    
    # Create agent
    agent = PPOAgent(env, config)
    
    try:
        # Train agent
        agent.train()
        
        # Evaluate final performance
        agent.evaluate(num_episodes=5)
        
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        env.close()

if __name__ == "__main__":
    main() 