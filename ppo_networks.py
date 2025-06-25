import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class CNNFeatureExtractor(nn.Module):
    """CNN for processing visual observations - Optimized for TITAN RTX 24GB"""
    
    def __init__(self, input_channels=3):
        super().__init__()
        
        # Much larger network for TITAN RTX utilization
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=8, stride=4),  # 4x larger
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),  # 4x larger
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=1),  # 4x larger
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),  # Additional layer
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1),  # Additional massive layer
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Flatten()
        )
        
        # Calculate conv output size
        self.conv_output_size = self._get_conv_output_size((3, 84, 84))
        
    def _get_conv_output_size(self, shape):
        """Calculate output size of conv layers"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            output = self.conv_layers(dummy_input)
            return output.shape[1]
    
    def forward(self, x):
        return self.conv_layers(x)

class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO with mixed continuous-discrete actions - TITAN RTX Optimized"""
    
    def __init__(self, observation_space, action_space, hidden_size=4096):  # Massive hidden size for TITAN RTX
        super().__init__()
        
        # Action space is Box([move_x, move_y, action_type])
        self.action_space_shape = action_space.shape[0]  # Should be 3
        
        # CNN for image processing
        self.cnn = CNNFeatureExtractor()
        
        # Feature processing
        feature_size = observation_space['features'].shape[0]
        
        # Combined feature processing
        combined_size = self.cnn.conv_output_size + feature_size
        
        # Massive shared network for TITAN RTX utilization
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),  # More regularization for larger network
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),  # Additional layer
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),  # Additional layer
            nn.ReLU()
        )
        
        # Actor heads - separate for continuous and discrete parts
        # Continuous movement coordinates (move_x, move_y)
        self.movement_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, 2),  # x, y coordinates
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
        # Discrete action type
        self.action_type_head = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, 12)  # 12 action types (point_nav + 4 arrows + 7 others)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.Linear(hidden_size // 8, 1)
        )
        
    def forward(self, observations):
        # Process image
        images = observations['image'].float() / 255.0
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)  # NHWC -> NCHW
        
        cnn_features = self.cnn(images)
        
        # Process other features
        other_features = observations['features']
        if len(other_features.shape) == 1:
            other_features = other_features.unsqueeze(0)
        
        # Combine features
        combined_features = torch.cat([cnn_features, other_features], dim=1)
        
        # Shared processing
        shared_output = self.shared_layers(combined_features)
        
        # Get outputs
        movement_coords = self.movement_head(shared_output)  # [batch, 2] - (x, y) coordinates
        action_type_logits = self.action_type_head(shared_output)  # [batch, 8] - discrete action logits
        value = self.critic(shared_output)
        
        return movement_coords, action_type_logits, value
    
    def act(self, observations, deterministic=False):
        """Select action given observations"""
        with torch.no_grad():
            movement_coords, action_type_logits, value = self.forward(observations)
            
            # Sample movement coordinates (continuous)
            if deterministic:
                # Use mean for deterministic action
                move_x, move_y = movement_coords[0]
            else:
                # Add small noise for exploration
                noise = torch.randn_like(movement_coords) * 0.1
                movement_coords_noisy = torch.clamp(movement_coords + noise, 0.0, 1.0)
                move_x, move_y = movement_coords_noisy[0]
            
            # Sample action type (discrete)
            if deterministic:
                action_type = torch.argmax(action_type_logits, dim=-1)
            else:
                dist = Categorical(logits=action_type_logits)
                action_type = dist.sample()
            
            # Combine into single action
            action = torch.tensor([move_x.item(), move_y.item(), action_type.item()], dtype=torch.float32)
            
            # Calculate log probability (simplified for now)
            action_type_log_prob = F.log_softmax(action_type_logits, dim=-1)[0, action_type]
            
            return action.numpy(), action_type_log_prob.item(), value.item()

class PPOBuffer:
    """Buffer for storing rollout data"""
    
    def __init__(self, size, observation_space, action_space, device):
        self.size = size
        self.device = device
        
        # Buffers
        self.observations = {
            'image': torch.zeros((size, 84, 84, 3), dtype=torch.uint8),
            'features': torch.zeros((size, observation_space['features'].shape[0]), dtype=torch.float32)
        }
        self.actions = torch.zeros((size, action_space.shape[0]), dtype=torch.float32)  # [move_x, move_y, action_type]
        self.rewards = torch.zeros(size, dtype=torch.float32)
        self.values = torch.zeros(size, dtype=torch.float32)
        self.log_probs = torch.zeros(size, dtype=torch.float32)
        self.dones = torch.zeros(size, dtype=torch.bool)
        
        self.ptr = 0
        self.max_size = size
        
    def store(self, obs, action, reward, value, log_prob, done):
        """Store transition in buffer"""
        assert self.ptr < self.max_size
        
        self.observations['image'][self.ptr] = torch.from_numpy(obs['image'])
        self.observations['features'][self.ptr] = torch.from_numpy(obs['features'])
        self.actions[self.ptr] = torch.from_numpy(action) if isinstance(action, np.ndarray) else action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr += 1
    
    def get(self, gamma=0.99, gae_lambda=0.95):
        """Get all buffer data with computed advantages"""
        assert self.ptr == self.max_size
        
        # Compute advantages using GAE
        advantages = torch.zeros_like(self.rewards)
        last_advantage = 0
        
        for t in reversed(range(self.max_size)):
            if t == self.max_size - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]
                
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t].float()) - self.values[t]
            advantages[t] = last_advantage = delta + gamma * gae_lambda * (1 - self.dones[t].float()) * last_advantage
        
        returns = advantages + self.values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            'observations': {
                'image': self.observations['image'].to(self.device),
                'features': self.observations['features'].to(self.device)
            },
            'actions': self.actions.to(self.device),
            'old_log_probs': self.log_probs.to(self.device),
            'advantages': advantages.to(self.device),
            'returns': returns.to(self.device)
        }
    
    def reset(self):
        """Reset buffer pointer"""
        self.ptr = 0 