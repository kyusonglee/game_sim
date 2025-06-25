#!/usr/bin/env python3
"""
AI Agent for Robot House Simulator
Uses computer vision and path planning to automatically play the game.
"""

import time
import json
import requests
import numpy as np
import cv2
import base64
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import random
import math
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import subprocess
import threading


@dataclass
class GameState:
    """Represents the current state of the game"""
    robot_pos: Tuple[float, float]
    robot_angle: float
    robot_room: str
    has_object: Optional[str]
    objects: List[Dict[str, Any]]
    task: Dict[str, Any]
    level: int
    score: int
    time_elapsed: float


@dataclass
class Action:
    """Represents an action the agent can take"""
    type: str  # 'move_forward', 'move_backward', 'rotate_left', 'rotate_right', 'pickup', 'drop'
    duration: float = 0.1  # How long to hold the key


class RobotAgent:
    """AI Agent that can play the Robot House Simulator"""
    
    def __init__(self, game_url="http://localhost:8000", headless=False):
        self.game_url = game_url
        self.headless = headless
        self.driver = None
        self.current_state = None
        self.action_history = []
        self.exploration_map = {}  # For mapping explored areas
        self.pathfinding_grid = None
        self.target_object = None
        self.target_location = None
        
        # Agent parameters
        self.action_delay = 0.2  # Delay between actions
        self.screenshot_interval = 1.0  # How often to take screenshots
        self.last_screenshot_time = 0
        
        # Movement parameters
        self.movement_step_size = 10  # pixels per movement
        self.rotation_step_size = 15  # degrees per rotation
        
    def start_browser(self):
        """Initialize the web browser"""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument("--window-size=1200,800")
        
        try:
            # Auto-install chromedriver
            import chromedriver_autoinstaller
            chromedriver_autoinstaller.install()
            
            # Try to use chromedriver
            self.driver = webdriver.Chrome(options=chrome_options)
            # Execute script to remove automation indicators
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        except Exception as e:
            print(f"❌ Failed to start Chrome: {e}")
            print("💡 Try installing chromedriver: pip install chromedriver-autoinstaller")
            return False
        
        # Navigate to the game
        try:
            print(f"🌐 Loading game from {self.game_url}")
            self.driver.get(self.game_url)
            
            # Wait for canvas to be present
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.ID, "gameCanvas"))
            )
            
            # Wait for game object to be initialized with more robust checking
            def check_game_loaded(driver):
                try:
                    result = driver.execute_script("""
                        return {
                            gameExists: typeof window.game !== 'undefined',
                            gameInitialized: window.game && window.game.robot ? true : false,
                            error: null
                        };
                    """)
                    return result['gameExists'] and result['gameInitialized']
                except Exception as e:
                    print(f"   Checking game status: {e}")
                    return False
            
            # Wait up to 20 seconds for game to fully initialize
            print("   Waiting for game to initialize...")
            WebDriverWait(self.driver, 20).until(check_game_loaded)
            
            print(f"✅ Successfully loaded game at {self.game_url}")
            
            # Give the game a moment to fully initialize
            time.sleep(2)
            
            return True
        except Exception as e:
            print(f"❌ Failed to load game: {e}")
            print("💡 Make sure the web server is running: python run_agent.py play-web")
            return False
    
    def take_screenshot(self) -> np.ndarray:
        """Take a screenshot of the game canvas"""
        if not self.driver:
            return None
        
        try:
            canvas = self.driver.find_element(By.ID, "gameCanvas")
            screenshot = canvas.screenshot_as_png
            
            # Convert to OpenCV format
            nparr = np.frombuffer(screenshot, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"❌ Failed to take screenshot: {e}")
            return None
    
    def get_game_state(self) -> Optional[GameState]:
        """Extract current game state from the browser"""
        if not self.driver:
            return None
        
        try:
            # Wait for game to be loaded
            self.driver.execute_script("return document.readyState === 'complete'")
            
            # Execute JavaScript to get game state with better error handling
            game_state_js = """
            try {
                if (!window.game) {
                    return { error: 'Game not loaded yet' };
                }
                
                var robot = window.game.robot || {};
                var task = window.game.task || {};
                var objects = window.game.objects || [];
                
                return {
                    robot: {
                        pos: robot.pos ? [robot.pos.x || 0, robot.pos.y || 0] : [0, 0],
                        angle: robot.angle || 0,
                        room: robot.room ? robot.room.name || 'Unknown' : 'Unknown',
                        has_object: robot.hasObject ? robot.hasObject.name || null : null
                    },
                    objects: objects.map(obj => ({
                        name: obj.name || 'unknown',
                        color: obj.color || 'gray',
                        room: obj.room ? obj.room.name || 'Unknown' : 'Unknown',
                        pos: obj.pos ? [obj.pos.x || 0, obj.pos.y || 0] : [0, 0]
                    })),
                    task: {
                        type: task.type || null,
                        target: task.object ? {
                            name: task.object.name || null,
                            color: task.object.color || null
                        } : null,
                        location: task.targetRoom ? task.targetRoom.name || null : null,
                        instruction: task.instruction || null
                    },
                    level: window.game.level || 1,
                    score: window.game.score || 0,
                    time: window.game.startTime ? (Date.now() - window.game.startTime) / 1000 : 0,
                    gameRunning: window.game.gameRunning || false
                };
            } catch (e) {
                return { error: 'Error extracting game state: ' + e.message };
            }
            """
            
            state_data = self.driver.execute_script(game_state_js)
            
            if state_data and 'error' in state_data:
                print(f"⚠️ Game state error: {state_data['error']}")
                return None
            
            if state_data and 'robot' in state_data:
                # Extract task target info
                task_target = None
                if state_data['task'] and state_data['task']['target']:
                    target = state_data['task']['target']
                    if target['name'] and target['color']:
                        task_target = (target['name'], target['color'])
                
                return GameState(
                    robot_pos=tuple(state_data['robot']['pos']),
                    robot_angle=state_data['robot']['angle'],
                    robot_room=state_data['robot']['room'],
                    has_object=state_data['robot']['has_object'],
                    objects=state_data['objects'],
                    task=state_data['task'],
                    level=state_data['level'],
                    score=state_data['score'],
                    time_elapsed=state_data['time']
                )
        except Exception as e:
            print(f"❌ Failed to get game state: {e}")
        
        return None
    
    def send_action(self, action: Action):
        """Send an action to the game"""
        if not self.driver:
            return
        
        key_mapping = {
            'move_forward': 'ArrowUp',
            'move_backward': 'ArrowDown', 
            'rotate_left': 'ArrowLeft',
            'rotate_right': 'ArrowRight',
            'pickup': 'KeyP',
            'drop': 'KeyD',
            'restart': 'KeyR'
        }
        
        if action.type not in key_mapping:
            print(f"❌ Unknown action type: {action.type}")
            return
        
        try:
            # Method 1: Use JavaScript to simulate key events (most reliable for games)
            key_code = key_mapping[action.type]
            
            # Simulate keydown event
            keydown_script = f"""
            var event = new KeyboardEvent('keydown', {{
                code: '{key_code}',
                key: '{self._get_key_value(key_code)}',
                bubbles: true,
                cancelable: true
            }});
            document.dispatchEvent(event);
            
            // Also ensure the game object receives the key state
            if (window.game && window.game.keys) {{
                window.game.keys['{key_code}'] = true;
            }}
            """
            
            self.driver.execute_script(keydown_script)
            
            # Hold the key for the specified duration
            time.sleep(action.duration)
            
            # Simulate keyup event
            keyup_script = f"""
            var event = new KeyboardEvent('keyup', {{
                code: '{key_code}',
                key: '{self._get_key_value(key_code)}',
                bubbles: true,
                cancelable: true
            }});
            document.dispatchEvent(event);
            
            // Also clear the key state in game
            if (window.game && window.game.keys) {{
                window.game.keys['{key_code}'] = false;
            }}
            """
            
            self.driver.execute_script(keyup_script)
            
            self.action_history.append(action)
            print(f"🎮 Action: {action.type} (duration: {action.duration}s)")
            
        except Exception as e:
            print(f"❌ Failed to send action: {e}")
            # Fallback method: try ActionChains
            try:
                self._send_action_fallback(action)
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
    
    def _get_key_value(self, key_code):
        """Get the key value for the given key code"""
        key_values = {
            'ArrowUp': 'ArrowUp',
            'ArrowDown': 'ArrowDown',
            'ArrowLeft': 'ArrowLeft', 
            'ArrowRight': 'ArrowRight',
            'KeyP': 'p',
            'KeyD': 'd',
            'KeyR': 'r'
        }
        return key_values.get(key_code, key_code)
    
    def _send_action_fallback(self, action: Action):
        """Fallback method using ActionChains and body element"""
        from selenium.webdriver.common.action_chains import ActionChains
        
        key_mapping = {
            'move_forward': Keys.ARROW_UP,
            'move_backward': Keys.ARROW_DOWN,
            'rotate_left': Keys.ARROW_LEFT,
            'rotate_right': Keys.ARROW_RIGHT,
            'pickup': 'p',
            'drop': 'd',
            'restart': 'r'
        }
        
        # Try to focus on the canvas first
        try:
            canvas = self.driver.find_element(By.ID, "gameCanvas")
            # Use JavaScript to focus instead of click
            self.driver.execute_script("arguments[0].focus();", canvas)
        except:
            pass
        
        # Send key to body element using ActionChains
        body = self.driver.find_element(By.TAG_NAME, "body")
        actions = ActionChains(self.driver)
        
        key = key_mapping[action.type]
        
        # Press and hold, then release
        actions.key_down(key, body)
        actions.pause(action.duration)
        actions.key_up(key, body)
        actions.perform()
        
        print(f"🎮 Fallback Action: {action.type} (duration: {action.duration}s)")
    
    def analyze_screenshot(self, img: np.ndarray) -> Dict[str, Any]:
        """Analyze screenshot to detect objects and features"""
        if img is None:
            return {}
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for object detection
        color_ranges = {
            'red': [(0, 50, 50), (10, 255, 255)],
            'blue': [(100, 50, 50), (130, 255, 255)],
            'green': [(40, 50, 50), (80, 255, 255)],
            'yellow': [(20, 50, 50), (30, 255, 255)],
        }
        
        detected_objects = []
        
        for color_name, (lower, upper) in color_ranges.items():
            # Create mask for this color
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Filter small objects
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    center = (x + w//2, y + h//2)
                    
                    detected_objects.append({
                        'color': color_name,
                        'center': center,
                        'area': area,
                        'bbox': (x, y, w, h)
                    })
        
        return {
            'detected_objects': detected_objects,
            'screenshot_shape': img.shape
        }
    
    def calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def calculate_angle_to_target(self, current_pos: Tuple[float, float], 
                                 target_pos: Tuple[float, float], current_angle: float) -> float:
        """Calculate the angle needed to face the target"""
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        target_angle = math.atan2(dy, dx) * 180 / math.pi
        
        # Normalize angles to [-180, 180]
        angle_diff = target_angle - current_angle
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        
        return angle_diff
    
    def plan_path_to_target(self, target_pos: Tuple[float, float]) -> List[Action]:
        """Plan a sequence of actions to reach the target position"""
        if not self.current_state:
            return []
        
        actions = []
        current_pos = self.current_state.robot_pos
        current_angle = self.current_state.robot_angle
        
        # Calculate distance and angle to target
        distance = self.calculate_distance(current_pos, target_pos)
        angle_diff = self.calculate_angle_to_target(current_pos, target_pos, current_angle)
        
        # First, rotate to face the target
        if abs(angle_diff) > 5:  # 5 degree tolerance
            rotation_steps = int(abs(angle_diff) / self.rotation_step_size)
            rotation_action = 'rotate_left' if angle_diff > 0 else 'rotate_right'
            
            for _ in range(rotation_steps):
                actions.append(Action(rotation_action, 0.1))
        
        # Then move forward towards the target
        if distance > 20:  # 20 pixel tolerance
            movement_steps = int(distance / self.movement_step_size)
            
            for _ in range(min(movement_steps, 10)):  # Limit to 10 steps at a time
                actions.append(Action('move_forward', 0.1))
        
        return actions
    
    def find_target_object(self) -> Optional[Tuple[float, float]]:
        """Find the position of the target object for the current task"""
        if not self.current_state or not self.current_state.task:
            return None
        
        task = self.current_state.task
        target_name = task.get('target', {}).get('name')
        target_color = task.get('target', {}).get('color')
        
        if not target_name or not target_color:
            return None
        
        # Look for the object in the game state
        for obj in self.current_state.objects:
            if (obj.get('name') == target_name and 
                obj.get('color') == target_color):
                return tuple(obj['pos'])
        
        return None
    
    def decide_next_action(self) -> Optional[Action]:
        """Decide the next action based on current game state"""
        if not self.current_state:
            return None
        
        task = self.current_state.task
        
        # If we have an object, find where to drop it
        if self.current_state.has_object:
            target_room = task.get('location')
            if target_room and target_room != self.current_state.robot_room:
                # Need to navigate to target room - implement room navigation
                return self.navigate_to_room(target_room)
            else:
                # We're in the right room, drop the object
                return Action('drop', 0.1)
        
        # If we don't have an object, find and pick it up
        else:
            target_pos = self.find_target_object()
            if target_pos:
                distance = self.calculate_distance(self.current_state.robot_pos, target_pos)
                
                if distance < 30:  # Close enough to pick up
                    return Action('pickup', 0.1)
                else:
                    # Navigate to the object
                    path = self.plan_path_to_target(target_pos)
                    if path:
                        return path[0]  # Return first action in path
            
            # If no target found, explore
            return self.explore()
    
    def navigate_to_room(self, target_room: str) -> Optional[Action]:
        """Navigate to a specific room"""
        # This is a simplified room navigation - in a real implementation,
        # you'd need to map door positions and plan paths between rooms
        
        # For now, just move randomly towards doors
        current_room = self.current_state.robot_room
        
        if current_room == target_room:
            return None
        
        # Move towards the center of the current room first, then towards doors
        # This is a simplified heuristic
        return Action('move_forward', 0.2)
    
    def explore(self) -> Action:
        """Explore the environment when no specific target is known"""
        # Simple exploration: rotate and move forward
        
        # 70% chance to move forward, 30% chance to rotate
        if random.random() < 0.7:
            return Action('move_forward', 0.2)
        else:
            # Randomly choose rotation direction
            rotation = 'rotate_left' if random.random() < 0.5 else 'rotate_right'
            return Action(rotation, 0.1)
    
    def run_episode(self, max_steps: int = 1000) -> Dict[str, Any]:
        """Run one episode of the game"""
        print(f"🚀 Starting new episode (max {max_steps} steps)")
        
        episode_data = {
            'start_time': time.time(),
            'actions_taken': 0,
            'final_score': 0,
            'final_level': 1,
            'success': False
        }
        
        for step in range(max_steps):
            # Update game state
            self.current_state = self.get_game_state()
            
            if not self.current_state:
                print("❌ Could not get game state")
                break
            
            # Take screenshot occasionally for debugging
            current_time = time.time()
            if current_time - self.last_screenshot_time > self.screenshot_interval:
                screenshot = self.take_screenshot()
                if screenshot is not None:
                    # Save screenshot for debugging
                    timestamp = int(current_time)
                    cv2.imwrite(f"debug_screenshot_{timestamp}.png", screenshot)
                    
                    # Analyze screenshot
                    analysis = self.analyze_screenshot(screenshot)
                    print(f"📸 Screenshot analysis: {len(analysis.get('detected_objects', []))} objects detected")
                
                self.last_screenshot_time = current_time
            
            # Decide next action
            action = self.decide_next_action()
            
            if action:
                self.send_action(action)
                episode_data['actions_taken'] += 1
                time.sleep(self.action_delay)
            else:
                print("🤔 No action decided, exploring...")
                self.send_action(Action('move_forward', 0.1))
                time.sleep(self.action_delay)
            
            # Check for level completion or game over
            if self.current_state:
                episode_data['final_score'] = self.current_state.score
                episode_data['final_level'] = self.current_state.level
                
                # Simple success heuristic: if score increased significantly
                if self.current_state.score > 100:
                    episode_data['success'] = True
                    print(f"🎉 Episode successful! Score: {self.current_state.score}")
                    break
        
        episode_data['end_time'] = time.time()
        episode_data['duration'] = episode_data['end_time'] - episode_data['start_time']
        
        print(f"📊 Episode completed:")
        print(f"   Duration: {episode_data['duration']:.1f}s")
        print(f"   Actions: {episode_data['actions_taken']}")
        print(f"   Final Score: {episode_data['final_score']}")
        print(f"   Final Level: {episode_data['final_level']}")
        print(f"   Success: {episode_data['success']}")
        
        return episode_data
    
    def train(self, num_episodes: int = 10):
        """Train the agent by running multiple episodes"""
        print(f"🎓 Starting training for {num_episodes} episodes")
        
        training_results = []
        
        for episode in range(num_episodes):
            print(f"\n📚 Episode {episode + 1}/{num_episodes}")
            
            # Restart the game
            self.send_action(Action('restart', 0.1))
            time.sleep(2)  # Wait for restart
            
            # Run episode
            episode_result = self.run_episode()
            episode_result['episode'] = episode + 1
            training_results.append(episode_result)
            
            # Save training progress
            with open(f"agent_training_results.json", 'w') as f:
                json.dump(training_results, f, indent=2)
        
        # Calculate training statistics
        success_rate = sum(1 for r in training_results if r['success']) / len(training_results)
        avg_score = np.mean([r['final_score'] for r in training_results])
        avg_actions = np.mean([r['actions_taken'] for r in training_results])
        
        print(f"\n🏆 Training Complete!")
        print(f"   Success Rate: {success_rate:.1%}")
        print(f"   Average Score: {avg_score:.1f}")
        print(f"   Average Actions per Episode: {avg_actions:.1f}")
        
        return training_results
    
    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()


def main():
    """Main function to run the agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Agent for Robot House Simulator")
    parser.add_argument('--url', default='http://localhost:8000', 
                       help='URL of the game server')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run')
    parser.add_argument('--headless', action='store_true',
                       help='Run browser in headless mode')
    
    args = parser.parse_args()
    
    # Create and run agent
    agent = RobotAgent(game_url=args.url, headless=args.headless)
    
    try:
        # Start browser and load game
        if not agent.start_browser():
            print("❌ Failed to start browser")
            return
        
        # Wait for game to load
        print("⏳ Waiting for game to load...")
        time.sleep(3)
        
        # Run training
        agent.train(num_episodes=args.episodes)
        
    finally:
        agent.cleanup()


if __name__ == "__main__":
    main() 