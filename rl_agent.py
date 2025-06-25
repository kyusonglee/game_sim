import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import spaces
import cv2
import json
import time
import logging
from collections import deque
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import threading
from typing import Dict, List, Tuple, Any, Optional
import pickle
import os
from dataclasses import dataclass

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

class OutdoorRobotEnvironment(gym.Env):
    """Gymnasium environment wrapper for the outdoor robot simulator"""
    
    def __init__(self, game_url: str = "http://localhost:8000", headless: bool = True):
        super().__init__()
        
        self.game_url = game_url
        self.headless = headless
        self.driver = None
        
        # Action space: [move_x, move_y, pickup, drop, photo, mow, warn, charge]
        # Move actions are discretized to 8 directions + stay
        self.action_space = spaces.Discrete(16)  # 9 movement + 7 discrete actions
        
        # Observation space: RGB image + game state features
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
            'features': spaces.Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)
        })
        
        self.max_episode_steps = 2000
        self.current_step = 0
        self.last_score = 0
        self.last_battery = 100
        self.episode_reward = 0
        
        # Movement directions (x, y offsets)
        self.movement_actions = [
            (0, 0),      # Stay
            (50, 0),     # Right
            (-50, 0),    # Left  
            (0, -50),    # Up
            (0, 50),     # Down
            (35, -35),   # Up-Right
            (-35, -35),  # Up-Left
            (35, 35),    # Down-Right
            (-35, 35),   # Down-Left
        ]
        
        # Discrete actions
        self.discrete_actions = ['pickup', 'drop', 'photo', 'mow', 'warn', 'charge', 'wait']
        
    def setup_browser(self):
        """Initialize browser for game interaction"""
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1200,800")
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.get(self.game_url)
        
        # Wait for game to load
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, "gameCanvas"))
        )
        time.sleep(3)  # Wait for game initialization
        
    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        if seed is not None:
            np.random.seed(seed)
        
        if self.driver is None:
            self.setup_browser()
        else:
            # Restart game
            self.driver.find_element(By.TAG_NAME, "body").send_keys('r')
            time.sleep(1)
        
        self.current_step = 0
        self.last_score = 0
        self.last_battery = 100
        self.episode_reward = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute action and return next state"""
        self.current_step += 1
        
        # Execute action
        reward = self._execute_action(action)
        
        # Get new observation
        observation = self._get_observation()
        
        # Check if episode is done
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps
        
        info = self._get_info()
        
        self.episode_reward += reward
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(self, action: int) -> float:
        """Execute the given action and return reward"""
        reward = 0.0
        
        try:
            # Get game state before action
            prev_state = self._get_game_state()
            prev_score = prev_state.get('game', {}).get('score', 0) if prev_state else 0
            prev_battery = prev_state.get('robot', {}).get('batteryLevel', 0) if prev_state else 0
            
            if action < 9:  # Movement actions
                dx, dy = self.movement_actions[action]
                if dx != 0 or dy != 0:  # Only move if not staying
                    robot_pos = prev_state.get('robot', {}).get('position', {})
                    new_x = robot_pos.get('x', 500) + dx
                    new_y = robot_pos.get('y', 350) + dy
                    
                    # Clamp to canvas bounds
                    new_x = max(50, min(950, new_x))
                    new_y = max(50, min(650, new_y))
                    
                    canvas = self.driver.find_element(By.ID, "gameCanvas")
                    action_chains = ActionChains(self.driver)
                    action_chains.move_to_element_with_offset(canvas, new_x-500, new_y-350).click().perform()
                    
                    # Small negative reward for movement to encourage efficiency
                    reward -= 0.1
            
            else:  # Discrete actions
                discrete_action_index = action - 9
                if discrete_action_index < len(self.discrete_actions):
                    action_name = self.discrete_actions[discrete_action_index]
                    
                    if action_name == 'pickup':
                        self.driver.find_element(By.TAG_NAME, "body").send_keys('p')
                    elif action_name == 'drop':
                        self.driver.find_element(By.TAG_NAME, "body").send_keys('d')
                    elif action_name == 'photo':
                        self.driver.find_element(By.TAG_NAME, "body").send_keys('c')
                    elif action_name == 'mow':
                        self.driver.find_element(By.TAG_NAME, "body").send_keys('m')
                    elif action_name == 'warn':
                        self.driver.find_element(By.TAG_NAME, "body").send_keys('w')
                    elif action_name == 'charge':
                        self.driver.find_element(By.TAG_NAME, "body").send_keys('g')
                    elif action_name == 'wait':
                        time.sleep(0.1)
            
            # Wait for action to complete
            time.sleep(0.2)
            
            # Get game state after action
            new_state = self._get_game_state()
            new_score = new_state.get('game', {}).get('score', 0) if new_state else 0
            new_battery = new_state.get('robot', {}).get('batteryLevel', 0) if new_state else 0
            
            # Reward calculation
            # Score improvement reward
            score_diff = new_score - prev_score
            if score_diff > 0:
                reward += score_diff * 0.1  # Scale down score rewards
            
            # Battery management rewards
            if new_battery < 20:
                reward -= 5.0  # Heavy penalty for low battery
            elif new_battery > 80:
                reward += 0.5  # Small reward for maintaining good battery
            
            # Battery drain penalty (encourage efficiency)
            battery_diff = new_battery - prev_battery
            if battery_diff < -5:  # Significant battery drain
                reward -= 1.0
            
            # Task completion rewards
            if new_state:
                active_tasks = new_state.get('game', {}).get('activeTasks', [])
                if len(active_tasks) == 0:
                    reward += 10.0  # Big reward for completing all tasks
                
                # Specific object interaction rewards
                env = new_state.get('environment', {})
                
                # Penalize for having many unhandled objects
                trash_count = len(env.get('trash', []))
                weed_count = len(env.get('weeds', []))
                animal_count = len(env.get('animals', []))
                
                reward -= (trash_count * 0.1 + weed_count * 0.1 + animal_count * 0.5)
                
                # Reward for being near charging station when battery is low
                robot_pos = new_state.get('robot', {}).get('position', {})
                charging_station = env.get('chargingStation', {}).get('rect', {})
                if charging_station and robot_pos:
                    cs_center_x = charging_station.get('x', 0) + charging_station.get('width', 0) / 2
                    cs_center_y = charging_station.get('y', 0) + charging_station.get('height', 0) / 2
                    distance_to_charger = np.sqrt(
                        (robot_pos.get('x', 0) - cs_center_x) ** 2 + 
                        (robot_pos.get('y', 0) - cs_center_y) ** 2
                    )
                    
                    if new_battery < 30 and distance_to_charger < 50:
                        reward += 2.0  # Reward for going to charger when needed
            
            # Update tracking variables
            self.last_score = new_score
            self.last_battery = new_battery
            
        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
            reward = -1.0  # Penalty for action execution errors
        
        return reward
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation (image + features)"""
        try:
            # Capture canvas screenshot
            canvas = self.driver.find_element(By.ID, "gameCanvas")
            canvas_screenshot = canvas.screenshot_as_png
            
            # Convert to numpy array and resize
            img_array = np.frombuffer(canvas_screenshot, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img_resized = cv2.resize(img, (84, 84))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            # Get game state features
            game_state = self._get_game_state()
            features = self._extract_features(game_state)
            
            return {
                'image': img_rgb,
                'features': features
            }
            
        except Exception as e:
            logger.error(f"Error getting observation: {e}")
            # Return default observation
            return {
                'image': np.zeros((84, 84, 3), dtype=np.uint8),
                'features': np.zeros(32, dtype=np.float32)
            }
    
    def _extract_features(self, game_state: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from game state"""
        features = np.zeros(32, dtype=np.float32)
        
        if not game_state:
            return features
        
        robot = game_state.get('robot', {})
        env = game_state.get('environment', {})
        game = game_state.get('game', {})
        
        # Robot features (0-6)
        robot_pos = robot.get('position', {})
        features[0] = robot_pos.get('x', 0) / 1000.0  # Normalize position
        features[1] = robot_pos.get('y', 0) / 700.0
        features[2] = robot.get('batteryLevel', 0) / 100.0
        features[3] = 1.0 if robot.get('isCharging') else 0.0
        features[4] = 1.0 if robot.get('carriedItem') else 0.0
        features[5] = robot.get('angle', 0) / (2 * np.pi)  # Normalize angle
        
        # Game state features (6-11)
        features[6] = game.get('score', 0) / 1000.0  # Normalize score
        features[7] = game.get('level', 1) / 10.0
        features[8] = game.get('dayCount', 1) / 30.0
        features[9] = game.get('gameTime', 0) / 1440.0  # Normalize to day
        features[10] = len(game.get('activeTasks', []))
        
        # Environment object counts (11-18)
        features[11] = len(env.get('plants', [])) / 10.0
        features[12] = len(env.get('animals', [])) / 5.0
        features[13] = len([p for p in env.get('packages', []) if not p.get('isDelivered')]) / 5.0
        features[14] = len(env.get('trash', [])) / 10.0
        features[15] = len(env.get('weeds', [])) / 10.0
        features[16] = len([g for g in env.get('grassAreas', []) if g.get('needsMowing')]) / 5.0
        
        # Distance to key locations (18-22)
        if robot_pos:
            # Distance to charging station
            charging_station = env.get('chargingStation', {}).get('rect', {})
            if charging_station:
                cs_x = charging_station.get('x', 0) + charging_station.get('width', 0) / 2
                cs_y = charging_station.get('y', 0) + charging_station.get('height', 0) / 2
                dist_to_charger = np.sqrt((robot_pos.get('x', 0) - cs_x) ** 2 + 
                                        (robot_pos.get('y', 0) - cs_y) ** 2)
                features[17] = min(dist_to_charger / 1000.0, 1.0)
            
            # Distance to house
            house = env.get('house', {}).get('rect', {})
            if house:
                house_x = house.get('x', 0) + house.get('width', 0) / 2
                house_y = house.get('y', 0) + house.get('height', 0) / 2
                dist_to_house = np.sqrt((robot_pos.get('x', 0) - house_x) ** 2 + 
                                      (robot_pos.get('y', 0) - house_y) ** 2)
                features[18] = min(dist_to_house / 1000.0, 1.0)
        
        # Plant health features (19-24)
        plants = env.get('plants', [])
        if plants:
            avg_health = np.mean([p.get('health', 100) for p in plants]) / 100.0
            min_health = min([p.get('health', 100) for p in plants]) / 100.0
            needs_water_count = sum([1 for p in plants if p.get('needsWater', False)])
            
            features[19] = avg_health
            features[20] = min_health
            features[21] = needs_water_count / len(plants)
        
        # Vision system features (22-24)
        vision = game_state.get('vision', {})
        features[22] = vision.get('detectedObjects', 0) / 10.0
        features[23] = vision.get('visionRange', 100) / 100.0
        
        # Time-based features (24-27)
        features[24] = (game.get('gameTime', 0) % 60) / 60.0  # Hour of day
        features[25] = np.sin(2 * np.pi * game.get('gameTime', 0) / 1440.0)  # Day cycle sin
        features[26] = np.cos(2 * np.pi * game.get('gameTime', 0) / 1440.0)  # Day cycle cos
        
        # Task priority features (27-31)
        tasks = game.get('activeTasks', [])
        task_types = ['package_pickup', 'warn_animals', 'mow_lawn', 'remove_weeds', 'cleanup_trash']
        for i, task_type in enumerate(task_types):
            has_task = any(task_type in task.get('id', '') for task in tasks)
            features[27 + i] = 1.0 if has_task else 0.0
        
        return features
    
    def _get_game_state(self) -> Dict[str, Any]:
        """Extract game state from browser"""
        try:
            game_state = self.driver.execute_script("""
                if (window.game) {
                    return {
                        robot: {
                            position: window.game.robot ? {x: window.game.robot.pos.x, y: window.game.robot.pos.y} : null,
                            angle: window.game.robot ? window.game.robot.angle : null,
                            batteryLevel: window.game.batteryLevel,
                            carriedItem: window.game.robot ? window.game.robot.carriedItem : null,
                            isCharging: window.game.isCharging
                        },
                        environment: {
                            house: window.game.house,
                            chargingStation: window.game.chargingStation,
                            frontDoor: window.game.frontDoor,
                            bins: window.game.bins,
                            plants: window.game.plants || [],
                            animals: window.game.animals || [],
                            packages: window.game.packages || [],
                            trash: window.game.trash || [],
                            weeds: window.game.weeds || [],
                            grassAreas: window.game.grassAreas || []
                        },
                        game: {
                            score: window.game.score,
                            level: window.game.level,
                            dayCount: window.game.dayCount,
                            gameTime: window.game.gameTime,
                            activeTasks: window.game.activeTasks || [],
                            gameRunning: window.game.gameRunning
                        },
                        vision: {
                            detectedObjects: window.game.detectedObjects ? Array.from(window.game.detectedObjects).length : 0,
                            visionRange: window.game.visionRange,
                            visionAngle: window.game.visionAngle
                        }
                    };
                }
                return null;
            """)
            return game_state
        except Exception as e:
            logger.error(f"Error getting game state: {e}")
            return None
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate"""
        try:
            game_state = self._get_game_state()
            if not game_state:
                return True
                
            # Game over conditions
            if not game_state.get('game', {}).get('gameRunning', True):
                return True
                
            # Battery depleted
            if game_state.get('robot', {}).get('batteryLevel', 0) <= 0:
                return True
                
            return False
        except:
            return True
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about current state"""
        game_state = self._get_game_state()
        if game_state:
            return {
                'score': game_state.get('game', {}).get('score', 0),
                'battery': game_state.get('robot', {}).get('batteryLevel', 0),
                'level': game_state.get('game', {}).get('level', 1),
                'day': game_state.get('game', {}).get('dayCount', 1),
                'tasks': len(game_state.get('game', {}).get('activeTasks', [])),
                'episode_reward': self.episode_reward
            }
        return {'episode_reward': self.episode_reward}
    
    def close(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()

class CNNFeatureExtractor(nn.Module):
    """CNN for processing visual observations"""
    
    def __init__(self, input_channels=3):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
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
    """Actor-Critic network for PPO"""
    
    def __init__(self, observation_space, action_space, hidden_size=512):
        super().__init__()
        
        self.action_space_size = action_space.n
        
        # CNN for image processing
        self.cnn = CNNFeatureExtractor()
        
        # Feature processing
        feature_size = observation_space['features'].shape[0]
        
        # Combined feature processing
        combined_size = self.cnn.conv_output_size + feature_size
        
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, self.action_space_size)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
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
        
        # Actor and Critic outputs
        action_logits = self.actor(shared_output)
        value = self.critic(shared_output)
        
        return action_logits, value
    
    def act(self, observations, deterministic=False):
        """Select action given observations"""
        with torch.no_grad():
            action_logits, value = self.forward(observations)
            
            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
                log_prob = F.log_softmax(action_logits, dim=-1)[0, action]
            else:
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()

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
        self.actions = torch.zeros(size, dtype=torch.long)
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
        self.actions[self.ptr] = action
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
                
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            advantages[t] = last_advantage = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_advantage
        
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
                action_logits, values = self.network(batch_obs)
                
                # Compute losses
                dist = Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Policy loss (PPO clip)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = batch_advantages * ratio
                surr2 = batch_advantages * torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss
                loss = (policy_loss + 
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
    env = OutdoorRobotEnvironment(headless=False)  # Set to True for faster training
    
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