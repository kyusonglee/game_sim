import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
import time
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from typing import Dict, List, Tuple, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FarmRobotEnvironment(gym.Env):
    """Gymnasium environment wrapper for the farm robot simulator"""
    
    def __init__(self, game_url: str = "http://localhost:8000", headless: bool = True):
        super().__init__()
        
        self.game_url = game_url
        self.headless = headless
        self.driver = None
        
        # Action space: [move_x, move_y, action_type]
        # move_x, move_y are continuous coordinates (0-1, will be scaled to canvas size)
        # action_type is discrete: 0=point_nav, 1=arrow_up, 2=arrow_down, 3=arrow_left, 4=arrow_right,
        #                         5=pickup, 6=drop, 7=photo, 8=mow, 9=warn, 10=water, 11=charge, 12=refill, 13=wait
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0]), 
            high=np.array([1.0, 1.0, 13]), 
            dtype=np.float32
        )
        
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
        
        # Canvas dimensions for coordinate scaling
        self.canvas_width = 1000
        self.canvas_height = 700
        
        # Discrete actions
        self.discrete_actions = ['point_nav', 'arrow_up', 'arrow_down', 'arrow_left', 'arrow_right', 
                               'pickup', 'drop', 'photo', 'mow', 'warn', 'water', 'charge', 'refill', 'wait']
        
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
    
    def _execute_action(self, action: np.ndarray) -> float:
        """Execute the given action and return reward"""
        reward = 0.0
        
        try:
            # Get game state before action
            prev_state = self._get_game_state()
            prev_score = prev_state.get('game', {}).get('score', 0) if prev_state else 0
            prev_battery = prev_state.get('robot', {}).get('batteryLevel', 0) if prev_state else 0
            
            # Parse action: [move_x, move_y, action_type]
            move_x_norm, move_y_norm, action_type_float = action
            action_type = int(round(action_type_float))
            
            # Clamp action type to valid range
            action_type = max(0, min(len(self.discrete_actions) - 1, action_type))
            action_name = self.discrete_actions[action_type]
            
            if action_name == 'point_nav':
                # Point navigation: convert normalized coordinates to canvas coordinates
                # Clamp input coordinates first
                move_x_norm = max(0.0, min(1.0, move_x_norm))
                move_y_norm = max(0.0, min(1.0, move_y_norm))
                
                target_x = int(move_x_norm * self.canvas_width)
                target_y = int(move_y_norm * self.canvas_height)
                
                # Clamp to safe canvas bounds (larger margins)
                target_x = max(100, min(self.canvas_width - 100, target_x))
                target_y = max(100, min(self.canvas_height - 100, target_y))
                
                # Get current robot position for distance calculation
                robot_pos = prev_state.get('robot', {}).get('position', {}) if prev_state else {}
                current_x = robot_pos.get('x', self.canvas_width // 2)
                current_y = robot_pos.get('y', self.canvas_height // 2)
                
                # Calculate movement distance for reward shaping
                move_distance = np.sqrt((target_x - current_x) ** 2 + (target_y - current_y) ** 2)
                
                # Execute point navigation - simplified approach
                try:
                    canvas = self.driver.find_element(By.ID, "gameCanvas")
                    
                    # Much simpler approach: just click center of canvas for now
                    # This avoids coordinate conversion issues during training
                    canvas.click()
                    logger.debug(f"Clicked canvas center (simplified navigation)")
                    
                except Exception as click_error:
                    logger.error(f"Canvas click failed: {click_error}")
                    # If even center click fails, use keyboard movement
                    try:
                        # Fallback to arrow key movement
                        if target_x > self.canvas_width // 2:
                            self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_RIGHT)
                        else:
                            self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_LEFT)
                        
                        if target_y > self.canvas_height // 2:
                            self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_DOWN)
                        else:
                            self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_UP)
                            
                        logger.debug("Used keyboard fallback navigation")
                    except:
                        logger.error("All navigation methods failed")
                
                # Movement cost based on distance (encourage efficient movement)
                reward -= move_distance * 0.001
                
            elif action_name == 'arrow_up':
                self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_UP)
                reward -= 0.01  # Small cost for movement
                
            elif action_name == 'arrow_down':
                self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_DOWN)
                reward -= 0.01
                
            elif action_name == 'arrow_left':
                self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_LEFT)
                reward -= 0.01
                
            elif action_name == 'arrow_right':
                self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_RIGHT)
                reward -= 0.01
                
            elif action_name == 'pickup':
                self.driver.find_element(By.TAG_NAME, "body").send_keys('p')
            elif action_name == 'drop':
                self.driver.find_element(By.TAG_NAME, "body").send_keys('d')
            elif action_name == 'photo':
                self.driver.find_element(By.TAG_NAME, "body").send_keys('c')
            elif action_name == 'mow':
                self.driver.find_element(By.TAG_NAME, "body").send_keys('m')
            elif action_name == 'warn':
                self.driver.find_element(By.TAG_NAME, "body").send_keys('w')
            elif action_name == 'water':
                self.driver.find_element(By.TAG_NAME, "body").send_keys('t')
            elif action_name == 'charge':
                self.driver.find_element(By.TAG_NAME, "body").send_keys('g')
            elif action_name == 'refill':
                self.driver.find_element(By.TAG_NAME, "body").send_keys('f')
            elif action_name == 'wait':
                time.sleep(0.1)
            
            # Wait for action to complete
            time.sleep(0.2)
            
            # Get game state after action
            new_state = self._get_game_state()
            new_score = new_state.get('game', {}).get('score', 0) if new_state else 0
            new_battery = new_state.get('robot', {}).get('batteryLevel', 0) if new_state else 0
            new_water = new_state.get('robot', {}).get('waterLevel', 0) if new_state else 0
            
            # Reward calculation
            # Score improvement reward
            score_diff = new_score - prev_score
            if score_diff > 0:
                reward += score_diff * 0.1  # Scale down score rewards
            
            # Battery management rewards (adjusted for faster drain)
            if new_battery < 15:
                reward -= 10.0  # Heavy penalty for critically low battery
            elif new_battery < 25:
                reward -= 2.0   # Penalty for low battery
            elif new_battery > 80:
                reward += 0.3   # Small reward for maintaining good battery
            
            # Water management rewards (new for farm robot)
            if new_water < 15:
                reward -= 5.0   # Penalty for critically low water
            elif new_water < 25:
                reward -= 1.0   # Penalty for low water
            elif new_water > 80:
                reward += 0.2   # Small reward for maintaining good water
            
            # Resource efficiency penalties
            battery_diff = new_battery - prev_battery
            if battery_diff < -3:  # Significant battery drain
                reward -= 0.5
            
            water_diff = new_water - (prev_state.get('robot', {}).get('waterLevel', 0) if prev_state else 0)
            if water_diff < -10:  # Significant water usage
                reward -= 0.3
            
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
                detected_animal_count = len(env.get('detectedAnimals', []))
                
                reward -= (trash_count * 0.1 + weed_count * 0.1 + animal_count * 0.3)
                
                # Heavy penalty for detected animals (immediate threat)
                if detected_animal_count > 0:
                    reward -= detected_animal_count * 1.0
                    
                # Reward for warning detected animals
                if action_name == 'warn' and detected_animal_count > 0:
                    reward += detected_animal_count * 2.0
                
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
                
                # Reward for being near water station when water is low
                water_station = env.get('waterStation', {}).get('rect', {})
                if water_station and robot_pos:
                    ws_center_x = water_station.get('x', 0) + water_station.get('width', 0) / 2
                    ws_center_y = water_station.get('y', 0) + water_station.get('height', 0) / 2
                    distance_to_water = np.sqrt(
                        (robot_pos.get('x', 0) - ws_center_x) ** 2 + 
                        (robot_pos.get('y', 0) - ws_center_y) ** 2
                    )
                    
                    if new_water < 30 and distance_to_water < 50:
                        reward += 1.5  # Reward for going to water station when needed
            
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
        
        # Robot features (0-7)
        robot_pos = robot.get('position', {})
        features[0] = robot_pos.get('x', 0) / 1000.0  # Normalize position
        features[1] = robot_pos.get('y', 0) / 700.0
        features[2] = robot.get('batteryLevel', 0) / 100.0
        features[3] = robot.get('waterLevel', 0) / 100.0  # New: robot water level
        features[4] = 1.0 if robot.get('isCharging') else 0.0
        features[5] = 1.0 if robot.get('carriedItem') else 0.0
        features[6] = robot.get('angle', 0) / (2 * np.pi)  # Normalize angle
        
        # Game state features (7-12)
        features[7] = game.get('score', 0) / 1000.0  # Normalize score
        features[8] = game.get('level', 1) / 10.0
        features[9] = game.get('dayCount', 1) / 30.0
        features[10] = game.get('gameTime', 0) / 1440.0  # Normalize to day
        features[11] = len(game.get('activeTasks', []))
        
        # Environment object counts (12-19)
        features[12] = len(env.get('plants', [])) / 10.0
        features[13] = len(env.get('animals', [])) / 5.0
        features[14] = len(env.get('detectedAnimals', [])) / 5.0  # Detected animals count
        features[15] = len(env.get('trash', [])) / 10.0
        features[16] = len(env.get('weeds', [])) / 10.0
        features[17] = len([g for g in env.get('grassAreas', []) if g.get('needsMowing')]) / 5.0
        features[18] = len([g for g in env.get('grassAreas', []) if g.get('needsWatering')]) / 5.0  # New: dry grass areas
        
        # Distance to key locations (19-20)
        if robot_pos:
            # Distance to charging station
            charging_station = env.get('chargingStation', {}).get('rect', {})
            if charging_station:
                cs_x = charging_station.get('x', 0) + charging_station.get('width', 0) / 2
                cs_y = charging_station.get('y', 0) + charging_station.get('height', 0) / 2
                dist_to_charger = np.sqrt((robot_pos.get('x', 0) - cs_x) ** 2 + 
                                        (robot_pos.get('y', 0) - cs_y) ** 2)
                features[19] = min(dist_to_charger / 1000.0, 1.0)
            
            # Distance to water station
            water_station = env.get('waterStation', {}).get('rect', {})
            if water_station:
                ws_x = water_station.get('x', 0) + water_station.get('width', 0) / 2
                ws_y = water_station.get('y', 0) + water_station.get('height', 0) / 2
                dist_to_water = np.sqrt((robot_pos.get('x', 0) - ws_x) ** 2 + 
                                      (robot_pos.get('y', 0) - ws_y) ** 2)
                features[20] = min(dist_to_water / 1000.0, 1.0)
        
        # Plant water features (21-23)
        plants = env.get('plants', [])
        if plants:
            avg_water = np.mean([p.get('waterLevel', 100) for p in plants]) / 100.0
            min_water = min([p.get('waterLevel', 100) for p in plants]) / 100.0
            needs_water_count = sum([1 for p in plants if p.get('needsWater', False)])
            
            features[21] = avg_water
            features[22] = min_water
            features[23] = needs_water_count / len(plants)
        
        # Vision system features (24-25)
        vision = game_state.get('vision', {})
        features[24] = vision.get('detectedObjects', 0) / 10.0
        features[25] = vision.get('visionRange', 100) / 100.0
        
        # Time-based features (26-28)
        features[26] = (game.get('gameTime', 0) % 60) / 60.0  # Hour of day
        features[27] = np.sin(2 * np.pi * game.get('gameTime', 0) / 1440.0)  # Day cycle sin
        features[28] = np.cos(2 * np.pi * game.get('gameTime', 0) / 1440.0)  # Day cycle cos
        
        # Current task features (29-31) - Enhanced task information
        tasks = game.get('activeTasks', [])
        
        # Priority task detection (more comprehensive)
        task_keywords = {
            'animal': ['warn_animals', 'animal', 'warn', 'wild'],
            'mow': ['mow_lawn', 'mow', 'grass', 'pasture'],
            'weed': ['remove_weeds', 'weed', 'remove'],
            'trash': ['cleanup_trash', 'trash', 'clean'],
            'water': ['water_plants', 'water_lawn', 'water', 'refill', 'thirsty'],
            'photo': ['take_photos', 'photo', 'monitor', 'daily']
        }
        
        task_priorities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        for task in tasks:
            task_desc = task.get('description', '').lower()
            task_id = task.get('id', '').lower()
            
            for i, (task_type, keywords) in enumerate(task_keywords.items()):
                if any(keyword in task_desc or keyword in task_id for keyword in keywords):
                    task_priorities[i] = 1.0
                    break
        
        # Distribute task priorities across features 29-31
        features[29] = task_priorities[0]  # animal
        features[30] = task_priorities[1]  # mow
        features[31] = task_priorities[2]  # weed
        
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
                            waterLevel: window.game.waterLevel,
                            carriedItem: window.game.robot ? window.game.robot.carriedItem : null,
                            isCharging: window.game.isCharging
                        },
                        environment: {
                            farmHouse: window.game.farmHouse,
                            chargingStation: window.game.chargingStation,
                            waterStation: window.game.waterStation,
                            bins: window.game.bins,
                            plants: window.game.plants || [],
                            animals: window.game.animals || [],
                            detectedAnimals: window.game.detectedAnimals ? Array.from(window.game.detectedAnimals) : [],
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