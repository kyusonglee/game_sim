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
import os

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
    update_frequency: int = 32768  # 16x larger buffer for massive GPU utilization (32k steps)
    num_epochs: int = 10
    batch_size: int = 2048  # 16x larger batch size for TITAN RTX (2k samples per batch)
    save_frequency: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # GPU optimization settings - TITAN RTX specific
    gpu_memory_fraction: float = 0.9  # Use 90% of 24GB = ~21GB per GPU
    enable_mixed_precision: bool = True  # Use half precision for speed
    prefetch_factor: int = 8  # More aggressive data loading
    num_workers: int = 4  # More parallel data processing
    pin_memory: bool = True  # Keep data in GPU memory
    non_blocking: bool = True  # Async GPU transfers
    force_single_gpu: bool = False  # Force single GPU usage (disable DataParallel)
    gpu_id: int = 0  # Which GPU to use when force_single_gpu is True
    save_dir: str = "./checkpoints"  # Directory to save checkpoints

class PPOAgent:
    """PPO Agent for training on the outdoor robot simulator - GPU Optimized"""
    
    def __init__(self, env, config: TrainingConfig):
        self.env = env
        self.config = config
        
        # Initialize network with massive architecture for TITAN RTX
        self.network = ActorCriticNetwork(
            env.observation_space, 
            env.action_space,
            hidden_size=4096  # Massive network for TITAN RTX 24GB utilization
        ).to(config.device)
        
        # Enable multi-GPU if available and not forced to single GPU
        if (config.device == "cuda" and 
            torch.cuda.device_count() > 1 and 
            not config.force_single_gpu):
            logger.info(f"🚀 Using {torch.cuda.device_count()} GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
            self.network = torch.nn.DataParallel(self.network)
            self.use_multi_gpu = True
        else:
            if config.force_single_gpu:
                # Set specific GPU
                if torch.cuda.device_count() > config.gpu_id:
                    torch.cuda.set_device(config.gpu_id)
                    logger.info(f"🎯 Using single GPU {config.gpu_id}: {torch.cuda.get_device_name(config.gpu_id)}")
                else:
                    logger.warning(f"⚠️ Requested GPU {config.gpu_id} not available, using GPU 0")
                    torch.cuda.set_device(0)
            else:
                logger.info(f"🎯 Using single GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
            self.use_multi_gpu = False
        
        # Pre-allocate GPU memory for maximum utilization
        if config.device == "cuda":
            # Set memory fraction to use most of the 24GB
            torch.cuda.set_per_process_memory_fraction(config.gpu_memory_fraction)
            
            logger.info(f"🚀 Pre-allocating GPU memory for TITAN RTX utilization...")
            
            # Force allocation of large memory blocks to utilize GPU
            memory_hogs = []
            try:
                # Allocate fewer but still large tensors to utilize GPU without OOM
                for i in range(3):  # Reduced from 10 to 3 to avoid OOM
                    # Large image batch tensors (reduced size)
                    large_image_batch = torch.randn(
                        config.batch_size * 2, 84, 84, 3,  # Reduced multiplier from 4 to 2
                        device=config.device, 
                        dtype=torch.float16 if config.enable_mixed_precision else torch.float32,
                        requires_grad=True
                    )
                    
                    # Large feature batch tensors (reduced size)
                    large_feature_batch = torch.randn(
                        config.batch_size * 2, 2048,  # Reduced from 4096 to 2048
                        device=config.device, 
                        dtype=torch.float16 if config.enable_mixed_precision else torch.float32,
                        requires_grad=True
                    )
                    
                    # Large weight matrices (reduced size)
                    large_weights = torch.randn(
                        2048, 2048,  # Reduced from 4096x4096 to 2048x2048
                        device=config.device, 
                        dtype=torch.float16 if config.enable_mixed_precision else torch.float32,
                        requires_grad=True
                    )
                    
                    memory_hogs.extend([large_image_batch, large_feature_batch, large_weights])
                
                # Force multiple forward passes to allocate intermediate tensors
                dummy_obs = {
                    'image': torch.randint(0, 255, (config.batch_size, 84, 84, 3), device=config.device, dtype=torch.uint8),
                    'features': torch.randn(config.batch_size, 32, device=config.device)
                }
                
                for _ in range(5):  # Multiple passes to build up memory
                    if config.enable_mixed_precision:
                        with torch.amp.autocast('cuda'):
                            if self.use_multi_gpu:
                                _ = self.network.module.act(dummy_obs)
                            else:
                                _ = self.network.act(dummy_obs)
                    else:
                        if self.use_multi_gpu:
                            _ = self.network.module.act(dummy_obs)
                        else:
                            _ = self.network.act(dummy_obs)
                
                # Log current memory usage
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                memory_cached = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"📊 GPU memory after pre-allocation: {memory_allocated:.2f}GB allocated, {memory_cached:.2f}GB cached")
                
                # Keep some of the large tensors in memory (don't delete all)
                # This forces the GPU to maintain high memory usage
                self._memory_hogs = memory_hogs[:5]  # Keep 5 large tensors
                
            except Exception as e:
                logger.warning(f"⚠️ Memory pre-allocation partially failed: {e}")
                # Continue with whatever memory we could allocate
            
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Don't clear cache - we want to keep the allocated memory
            logger.info(f"✅ GPU memory pre-allocation completed")
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.Adam(
            self.network.parameters(), 
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        
        # Mixed precision training for faster GPU processing
        if config.enable_mixed_precision and config.device == "cuda":
            self.scaler = torch.amp.GradScaler('cuda')  # Updated API
            self.use_amp = True
            logger.info("Enabled mixed precision training for faster GPU processing")
        else:
            self.scaler = None
            self.use_amp = False
        
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
        self.total_episodes = 0  # Total episodes trained across all sessions
        
        # Logging
        self.reward_history = []
        self.loss_history = []
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
        logger.info(f"💾 Checkpoints will be saved to: {config.save_dir}")
        
    def collect_rollout(self):
        """Collect rollout data"""
        logger.info(f"🔄 STARTING ROLLOUT COLLECTION (Buffer size: {self.config.update_frequency})")
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(self.config.update_frequency):
            if step % 100 == 0:  # More frequent logging
                progress_pct = (step / self.config.update_frequency) * 100
                logger.info(f"Rollout progress: {step}/{self.config.update_frequency} steps ({progress_pct:.1f}%)")
            
            try:
                # Convert obs to tensors
                obs_tensor = {
                    'image': torch.from_numpy(obs['image']).to(self.config.device),
                    'features': torch.from_numpy(obs['features']).to(self.config.device)
                }
                
                # Get action
                if self.use_multi_gpu:
                    # For DataParallel, use the module's act method
                    action, log_prob, value = self.network.module.act(obs_tensor)
                else:
                    # For single GPU, use act method directly
                    action, log_prob, value = self.network.act(obs_tensor)
                logger.debug(f"Step {step}: Action generated: {action}")
                
                # Take step in environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                logger.debug(f"Step {step}: Reward: {reward:.3f}, Done: {done}")
                
                # Store in buffer
                self.buffer.store(obs, action, reward, value, log_prob, done)
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    logger.debug(f"Episode completed: Reward={episode_reward:.2f}, Length={episode_length}")
                    
                    # Reset environment
                    obs, _ = self.env.reset()
                    episode_reward = 0
                    episode_length = 0
                else:
                    obs = next_obs
                    
            except Exception as e:
                logger.error(f"Error in rollout step {step}: {e}")
                # Try to recover
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
        
        logger.info(f"✅ ROLLOUT COLLECTION COMPLETED!")
        logger.info(f"   Total steps collected: {self.config.update_frequency}")
        logger.info(f"   Episodes completed: {len(self.episode_rewards)}")
        if len(self.episode_rewards) > 0:
            logger.info(f"   Average episode reward: {np.mean(self.episode_rewards):.2f}")
            logger.info(f"   Average episode length: {np.mean(self.episode_lengths):.1f}")
        logger.info(f"🚀 STARTING POLICY UPDATE...")
    
    def update_policy(self):
        """Update policy using PPO"""
        logger.info(f"Starting policy update with batch size {self.config.batch_size}")
        
        # Get buffer data
        buffer_data = self.buffer.get(self.config.gamma, self.config.gae_lambda)
        
        # Log GPU memory before training
        if self.config.device == "cuda":
            torch.cuda.empty_cache()  # Clear cache before training
            initial_memory = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"GPU memory before policy update: {initial_memory:.2f} GB")
        
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
                
                # Forward pass with mixed precision
                if self.use_amp:
                    with torch.cuda.amp.autocast():
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
                    
                    # Mixed precision backward pass
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                else:
                    # Regular forward pass
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
                    
                    # Regular backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                    self.optimizer.step()
                
                total_loss += loss.item()
                num_updates += 1
                
                # Log detailed loss information every few batches
                if num_updates % 5 == 0:  # Every 5 batches
                    logger.info(f"  Batch {num_updates}: Loss={loss.item():.4f} "
                              f"(Policy={policy_loss.item():.4f}, "
                              f"Value={value_loss.item():.4f}, "
                              f"Movement={movement_loss.item():.4f}, "
                              f"Entropy={entropy.item():.4f})")
        
        avg_loss = total_loss / num_updates
        self.loss_history.append(avg_loss)
        
        # Enhanced policy update completion message
        logger.info(f"🔥 POLICY UPDATE COMPLETED 🔥")
        logger.info(f"   Total batches processed: {num_updates}")
        logger.info(f"   Average loss: {avg_loss:.4f}")
        logger.info(f"   Samples processed: {self.config.update_frequency}")
        logger.info(f"   Batch size: {self.config.batch_size}")
        
        # Log GPU memory after training
        if self.config.device == "cuda":
            final_memory = torch.cuda.memory_allocated(0) / 1024**3
            memory_cached = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"   GPU memory after update: {final_memory:.2f}GB allocated, {memory_cached:.2f}GB cached")
        
        # Reset buffer
        self.buffer.reset()
        self.training_step += 1
    
    def train(self):
        """Main training loop with GPU optimization"""
        logger.info(f"Starting PPO training on {self.config.device}")
        
        # GPU memory management
        if self.config.device == "cuda":
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(self.config.gpu_memory_fraction)
            
            # Clear cache and get initial memory info
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Total GPU memory: {total_memory:.1f} GB")
            logger.info(f"Using {self.config.gpu_memory_fraction*100:.0f}% of GPU memory")
            logger.info(f"Mixed precision: {'Enabled' if self.use_amp else 'Disabled'}")
            logger.info(f"Batch size: {self.config.batch_size} (16x larger for TITAN RTX)")
            logger.info(f"Buffer size: {self.config.update_frequency} (16x larger)")
        
        episode_count = 0
        
        try:
            while episode_count < self.config.max_episodes:
                # Collect rollout
                self.collect_rollout()
                
                # Update policy
                self.update_policy()
                
                # Logging with GPU memory monitoring
                if len(self.episode_rewards) > 0:
                    avg_reward = np.mean(self.episode_rewards)
                    avg_length = np.mean(self.episode_lengths)
                    self.reward_history.append(avg_reward)
                    
                    # GPU memory usage
                    gpu_memory_info = ""
                    if self.config.device == "cuda":
                        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
                        memory_cached = torch.cuda.memory_reserved(0) / 1024**3  # GB
                        gpu_memory_info = f", GPU: {memory_allocated:.1f}GB/{memory_cached:.1f}GB"
                    
                    logger.info(f"Training step {self.training_step}: "
                              f"Avg reward: {avg_reward:.2f}, "
                              f"Avg length: {avg_length:.1f}"
                              f"{gpu_memory_info}")
                
                # Save model
                if self.training_step % self.config.save_frequency == 0:
                    checkpoint_name = f"checkpoint_step_{self.training_step}_episodes_{self.total_episodes}.pth"
                    self.save_model(checkpoint_name)
                
                # Update total episode count
                self.total_episodes += len(self.episode_rewards)
                episode_count = self.total_episodes
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            # Save interrupted checkpoint
            interrupt_name = f"checkpoint_interrupted_step_{self.training_step}_episodes_{self.total_episodes}.pth"
            self.save_model(interrupt_name)
        
        logger.info("Training completed")
        final_name = f"checkpoint_final_step_{self.training_step}_episodes_{self.total_episodes}.pth"
        self.save_model(final_name)
        self.plot_training_progress()
    
    def save_model(self, filename: str):
        """Save model and training state"""
        # Ensure filename includes directory
        if not filename.startswith('/') and not filename.startswith('./'):
            filepath = os.path.join(self.config.save_dir, filename)
        else:
            filepath = filename
            
        checkpoint_data = {
            'network_state_dict': self.network.module.state_dict() if self.use_multi_gpu else self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'total_episodes': self.total_episodes,
            'config': self.config,
            'reward_history': self.reward_history,
            'loss_history': self.loss_history,
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths)
        }
        
        torch.save(checkpoint_data, filepath)
        logger.info(f"💾 Checkpoint saved to {filepath}")
        logger.info(f"   Training step: {self.training_step}")
        logger.info(f"   Total episodes: {self.total_episodes}")
        logger.info(f"   Average reward (last 100): {np.mean(self.episode_rewards) if self.episode_rewards else 0:.2f}")
        
        return filepath
    
    def load_model(self, filename: str):
        """Load model and training state"""
        logger.info(f"📂 Loading checkpoint from {filename}")
        checkpoint = torch.load(filename, map_location=self.config.device)
        
        # Load network state
        if self.use_multi_gpu:
            self.network.module.load_state_dict(checkpoint['network_state_dict'])
        else:
            self.network.load_state_dict(checkpoint['network_state_dict'])
            
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training progress
        self.training_step = checkpoint.get('training_step', 0)
        self.total_episodes = checkpoint.get('total_episodes', 0)
        self.reward_history = checkpoint.get('reward_history', [])
        self.loss_history = checkpoint.get('loss_history', [])
        
        # Restore episode tracking if available
        if 'episode_rewards' in checkpoint:
            self.episode_rewards = deque(checkpoint['episode_rewards'], maxlen=100)
        if 'episode_lengths' in checkpoint:
            self.episode_lengths = deque(checkpoint['episode_lengths'], maxlen=100)
            
        logger.info(f"✅ Checkpoint loaded successfully!")
        logger.info(f"   Resumed from training step: {self.training_step}")
        logger.info(f"   Total episodes completed: {self.total_episodes}")
        logger.info(f"   Reward history length: {len(self.reward_history)}")
        logger.info(f"   Loss history length: {len(self.loss_history)}")
        
        if self.episode_rewards:
            logger.info(f"   Recent average reward: {np.mean(self.episode_rewards):.2f}")
            
        return checkpoint
    
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
        update_frequency=32768,
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