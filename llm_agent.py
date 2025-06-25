import asyncio
import json
import base64
import time
from typing import Dict, List, Tuple, Any, Optional
import openai
from PIL import Image
import io
import numpy as np
import cv2
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameStateExtractor:
    """Extracts game state information from the browser"""
    
    def __init__(self, driver):
        self.driver = driver
        
    def get_game_state(self) -> Dict[str, Any]:
        """Extract current game state from the browser"""
        try:
            # Execute JavaScript to get game state
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
                            plants: window.game.plants ? window.game.plants.map(p => ({
                                type: p.type,
                                position: p.pos,
                                needsWater: p.needsWater,
                                waterLevel: p.waterLevel
                            })) : [],
                            animals: window.game.animals ? window.game.animals.map(a => ({
                                type: a.type,
                                position: a.pos,
                                isLeaving: a.isLeaving
                            })) : [],
                            detectedAnimals: window.game.detectedAnimals ? Array.from(window.game.detectedAnimals).map(a => ({
                                type: a.type,
                                position: a.pos,
                                isLeaving: a.isLeaving
                            })) : [],
                            trash: window.game.trash ? window.game.trash.map(t => ({
                                type: t.type,
                                position: t.pos,
                                item: t.item
                            })) : [],
                            weeds: window.game.weeds ? window.game.weeds.map(w => ({
                                position: w.pos,
                                size: w.size
                            })) : [],
                            grassAreas: window.game.grassAreas ? window.game.grassAreas.map(g => ({
                                rect: g.rect,
                                grassHeight: g.grassHeight,
                                needsMowing: g.needsMowing,
                                waterLevel: g.waterLevel,
                                needsWatering: g.needsWatering
                            })) : []
                        },
                        game: {
                            score: window.game.score,
                            level: window.game.level,
                            dayCount: window.game.dayCount,
                            gameTime: window.game.gameTime,
                            activeTasks: window.game.activeTasks ? window.game.activeTasks.map(t => ({
                                id: t.id,
                                description: t.description
                            })) : [],
                            gameRunning: window.game.gameRunning
                        },
                        vision: {
                            detectedObjects: window.game.detectedObjects ? Array.from(window.game.detectedObjects).length : 0,
                            animalDetectionRange: window.game.animalDetectionRange,
                            visionRange: window.game.visionRange,
                            visionAngle: window.game.visionAngle
                        }
                    };
                }
                return null;
            """)
            
            return game_state
            
        except Exception as e:
            logger.error(f"Error extracting game state: {e}")
            return None
    
    def capture_screenshot(self) -> str:
        """Capture game canvas as base64 image"""
        try:
            # Get canvas element and capture screenshot
            canvas = self.driver.find_element(By.ID, "gameCanvas")
            canvas_screenshot = canvas.screenshot_as_png
            
            # Convert to base64
            img_base64 = base64.b64encode(canvas_screenshot).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Error capturing screenshot: {e}")
            return None

class FarmRobotLLMAgent:
    """LLM-powered agent for the farm robot simulator"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.driver = None
        self.state_extractor = None
        self.action_history = []
        self.game_context = {
            "session_start": time.time(),
            "total_actions": 0,
            "successful_actions": 0,
            "current_strategy": "explore_and_learn"
        }
        
        # Rate limiting
        self.last_api_call = 0
        self.min_api_interval = 2.0  # Minimum 2 seconds between API calls
        self.max_retries = 3
        self.base_retry_delay = 1.0
        
    def setup_browser(self, game_url: str = "http://localhost:8000"):
        """Setup browser and navigate to game"""
        options = webdriver.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        # Remove headless for visual debugging
        # options.add_argument("--headless")
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.get(game_url)
        
        # Wait for game to load
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, "gameCanvas"))
        )
        
        time.sleep(2)  # Additional wait for game initialization
        self.state_extractor = GameStateExtractor(self.driver)
        logger.info("Browser setup complete")
    
    def create_state_description(self, game_state: Dict[str, Any]) -> str:
        """Create a textual description of the game state"""
        if not game_state:
            return "Unable to read game state"
        
        robot = game_state.get('robot', {})
        env = game_state.get('environment', {})
        game = game_state.get('game', {})
        
        description = f"""
FARM ROBOT SIMULATOR - CURRENT STATE

FARM ROBOT STATUS:
- Position: ({robot.get('position', {}).get('x', 0):.1f}, {robot.get('position', {}).get('y', 0):.1f})
- Battery: {robot.get('batteryLevel', 0):.1f}%
- Water Tank: {robot.get('waterLevel', 0):.1f}%
- Charging: {'Yes' if robot.get('isCharging') else 'No'}
- Carrying: {robot.get('carriedItem', {}).get('type', 'Nothing') if robot.get('carriedItem') else 'Nothing'}

FARM STATUS:
- Score: {game.get('score', 0)}
- Level: {game.get('level', 1)}
- Day: {game.get('dayCount', 1)}
- Game Time: {game.get('gameTime', 0):.1f} minutes

CURRENT FARM TASKS:
"""
        
        tasks = game.get('activeTasks', [])
        if tasks:
            for task in tasks:
                description += f"- {task.get('description', 'Unknown task')}\n"
        else:
            description += "- No active tasks\n"
        
        description += "\nFARM ENVIRONMENT:\n"
        
        # Crops/Plants
        plants = env.get('plants', [])
        if plants:
            description += f"CROPS/PLANTS ({len(plants)}):\n"
            for i, plant in enumerate(plants):
                pos = plant.get('position', {})
                water = plant.get('waterLevel', 0)
                needs_water = " (THIRSTY - NEEDS WATER!)" if plant.get('needsWater') else ""
                description += f"  {i+1}. {plant.get('type', 'unknown')} at ({pos.get('x', 0):.1f}, {pos.get('y', 0):.1f}) - Water: {water:.1f}%{needs_water}\n"
        
        # Detected Wild Animals (only these are visible!)
        detected_animals = env.get('detectedAnimals', [])
        if detected_animals:
            description += f"WILD ANIMALS DETECTED ({len(detected_animals)}) - WARN THEM IMMEDIATELY:\n"
            for i, animal in enumerate(detected_animals):
                pos = animal.get('position', {})
                status = "leaving" if animal.get('isLeaving') else "THREATENING CROPS!"
                description += f"  {i+1}. {animal.get('type', 'unknown')} at ({pos.get('x', 0):.1f}, {pos.get('y', 0):.1f}) - Status: {status}\n"
        else:
            description += "WILD ANIMALS: No animals detected within range\n"
        
        # Farm Trash
        trash = env.get('trash', [])
        if trash:
            description += f"FARM TRASH ({len(trash)}):\n"
            for i, item in enumerate(trash):
                pos = item.get('position', {})
                description += f"  {i+1}. {item.get('type', 'unknown')} at ({pos.get('x', 0):.1f}, {pos.get('y', 0):.1f})\n"
        
        # Weeds
        weeds = env.get('weeds', [])
        if weeds:
            description += f"WEEDS ({len(weeds)}):\n"
            for i, weed in enumerate(weeds):
                pos = weed.get('position', {})
                description += f"  {i+1}. Weed at ({pos.get('x', 0):.1f}, {pos.get('y', 0):.1f}) - Size: {weed.get('size', 0):.1f}\n"
        
        # Pasture/Grass areas
        grass_areas = env.get('grassAreas', [])
        needs_mowing = [g for g in grass_areas if g.get('needsMowing')]
        needs_watering = [g for g in grass_areas if g.get('needsWatering')]
        
        if needs_mowing:
            description += f"PASTURE AREAS NEEDING MOWING ({len(needs_mowing)}):\n"
            for i, area in enumerate(needs_mowing):
                rect = area.get('rect', {})
                description += f"  {i+1}. Overgrown area at ({rect.get('x', 0):.1f}, {rect.get('y', 0):.1f}) - Size: {rect.get('width', 0)}x{rect.get('height', 0)}\n"
        
        if needs_watering:
            description += f"PASTURE AREAS NEEDING WATERING ({len(needs_watering)}):\n"
            for i, area in enumerate(needs_watering):
                rect = area.get('rect', {})
                water = area.get('waterLevel', 0)
                description += f"  {i+1}. Dry pasture at ({rect.get('x', 0):.1f}, {rect.get('y', 0):.1f}) - Size: {rect.get('width', 0)}x{rect.get('height', 0)}, Water: {water:.1f}%\n"
        
        # Key locations
        charging_station = env.get('chargingStation', {}).get('rect', {})
        water_station = env.get('waterStation', {}).get('rect', {})
        farm_house = env.get('farmHouse', {}).get('rect', {})
        
        description += f"""
KEY FARM LOCATIONS:
- Charging Station: ({charging_station.get('x', 0) + charging_station.get('width', 0)/2:.1f}, {charging_station.get('y', 0) + charging_station.get('height', 0)/2:.1f}) - Battery charging only
- Water Station: ({water_station.get('x', 0) + water_station.get('width', 0)/2:.1f}, {water_station.get('y', 0) + water_station.get('height', 0)/2:.1f}) - Water refill only  
- Farm House: ({farm_house.get('x', 0) + farm_house.get('width', 0)/2:.1f}, {farm_house.get('y', 0) + farm_house.get('height', 0)/2:.1f})

FARM BINS:
"""
        bins = env.get('bins', [])
        for bin_info in bins:
            rect = bin_info.get('rect', {})
            description += f"- {bin_info.get('type', 'unknown').title()} bin at ({rect.get('x', 0) + rect.get('width', 0)/2:.1f}, {rect.get('y', 0) + rect.get('height', 0)/2:.1f})\n"
        
        # Add stuck warning if present
        stuck_warning = game_state.get('stuck_warning', '')
        if stuck_warning:
            description += stuck_warning
        
        return description
    
    async def get_llm_decision(self, game_state: Dict[str, Any], screenshot: str = None) -> Dict[str, Any]:
        """Get action decision from LLM"""
        state_description = self.create_state_description(game_state)
        
        messages = [
            {
                "role": "system",
                "content": """You are an AI agent controlling a robot in a farm agricultural management simulator. Your goal is to efficiently manage farm operations while protecting crops from wild animals.

AVAILABLE ACTIONS:
1. MOVE - Navigate to coordinates: {"action": "move", "x": 100, "y": 200}
2. ARROW MOVEMENT - Precise local movement:
   - {"action": "arrow_up"} - Move up
   - {"action": "arrow_down"} - Move down  
   - {"action": "arrow_left"} - Move left
   - {"action": "arrow_right"} - Move right
3. PICKUP - Pick up nearby items: {"action": "pickup"}
4. DROP - Drop carried items: {"action": "drop"}
5. PHOTO - Take photo of crops: {"action": "photo"}
6. MOW - Mow pasture if on grass: {"action": "mow"}
7. WARN - Warn wild animals away: {"action": "warn"}
8. WATER - Water nearby crops/pasture: {"action": "water"}
9. CHARGE - Go to charging station: {"action": "charge"}
10. REFILL - Go to water station: {"action": "refill"}
11. WAIT - Wait and observe: {"action": "wait"}

FARM RULES:
- Battery drains when moving/working, charges at charging station
- Water tank drains when watering, refills at separate water station (blue, near charging station)
- Wild animals are only visible within detection range - warn them immediately!
- Farm trash goes to the single trash bin
- Wild animals threaten crops - warn them quickly for bonus points
- Take daily photos of crops to monitor their growth
- Water crops (get close to them) and pasture (stand on grass area) when dry
- Mow pasture when it gets too long
- Remove weeds to trash bin

CRITICAL: WILD ANIMAL DETECTION
- Wild animals are only visible when within ~120m detection range
- You can only see/warn animals that are currently detected
- Prioritize warning detected animals immediately - they damage crops!

STRATEGY TIPS:
- Keep battery and water above 20% for safety
- HIGHEST PRIORITY: Warn any detected wild animals immediately
- Water thirsty crops and dry pasture areas regularly
- Movement strategy:
  * Use MOVE for long-distance navigation to specific coordinates
  * Use ARROW keys for precise positioning and fine adjustments
  * Arrow movements are good for exploring and detecting hidden animals
- Be efficient with movement to conserve battery and water
- Complete multiple tasks in same area when possible

STUCK ROBOT RECOVERY:
- If repeating the same action, try a different approach
- If navigation fails, use arrow keys for local movement
- If repeatedly moving to same location, try a nearby location instead
- Sometimes pickup/drop/water/warn actions work better than repeated movement
- Consider moving to a completely different area or task

Always respond with a valid JSON action and brief reasoning."""
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": f"Current game state:\n{state_description}\n\nWhat action should the robot take next? Respond with JSON format."
                    }
                ]
            }
        ]
        
        # Add screenshot if available
        if screenshot:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": screenshot}
            })
        
        # Rate limiting - ensure minimum interval between API calls
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < self.min_api_interval:
            wait_time = self.min_api_interval - time_since_last_call
            logger.info(f"Rate limiting: waiting {wait_time:.1f}s before API call")
            await asyncio.sleep(wait_time)
        
        # Retry logic for rate limiting and other API errors
        for attempt in range(self.max_retries):
            try:
                self.last_api_call = time.time()
                
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=messages,
                    max_tokens=200,
                    temperature=0.1
                )
                
                response_text = response.choices[0].message.content
                logger.info(f"LLM Response: {response_text}")
                
                # Extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    decision = json.loads(json_match.group())
                    return decision
                else:
                    # Fallback action
                    return {"action": "wait", "reason": "Could not parse LLM response"}
                    
            except Exception as e:
                error_str = str(e)
                logger.warning(f"API call attempt {attempt + 1} failed: {error_str}")
                
                # Check if it's a rate limit error
                if "rate_limit_exceeded" in error_str or "429" in error_str:
                    # Extract wait time from error message if available
                    import re
                    wait_match = re.search(r'try again in (\d+(?:\.\d+)?)(?:ms|s)', error_str)
                    if wait_match:
                        wait_time = float(wait_match.group(1))
                        if 'ms' in error_str:
                            wait_time /= 1000  # Convert ms to seconds
                        wait_time = max(wait_time, 1.0)  # Minimum 1 second wait
                    else:
                        # Exponential backoff
                        wait_time = self.base_retry_delay * (2 ** attempt)
                    
                    if attempt < self.max_retries - 1:
                        logger.info(f"Rate limited. Waiting {wait_time:.1f}s before retry {attempt + 2}/{self.max_retries}")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Max retries reached for rate limit. Using fallback action.")
                        return {"action": "wait", "reason": f"Rate limit exceeded after {self.max_retries} attempts"}
                
                # For other errors, retry with exponential backoff
                elif attempt < self.max_retries - 1:
                    wait_time = self.base_retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {wait_time:.1f}s... (attempt {attempt + 2}/{self.max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Max retries reached. Final error: {e}")
                    return {"action": "wait", "reason": f"LLM error after {self.max_retries} attempts: {str(e)}"}
        
        # Should never reach here, but just in case
        return {"action": "wait", "reason": "Unexpected error in retry logic"}
    
    def execute_action(self, action: Dict[str, Any]) -> bool:
        """Execute the chosen action in the game"""
        try:
            action_type = action.get('action', 'wait')
            
            if action_type == 'move':
                x, y = action.get('x', 500), action.get('y', 350)
                # Click on canvas to navigate - canvas is 1000x700
                canvas = self.driver.find_element(By.ID, "gameCanvas")
                action_chains = ActionChains(self.driver)
                # Calculate offset from center of canvas
                offset_x = x - 500  # Canvas width / 2
                offset_y = y - 350  # Canvas height / 2
                action_chains.move_to_element_with_offset(canvas, offset_x, offset_y).click().perform()
                
            elif action_type == 'arrow_up':
                self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_UP)
                
            elif action_type == 'arrow_down':
                self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_DOWN)
                
            elif action_type == 'arrow_left':
                self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_LEFT)
                
            elif action_type == 'arrow_right':
                self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_RIGHT)
                
            elif action_type == 'pickup':
                self.driver.find_element(By.TAG_NAME, "body").send_keys('p')
                
            elif action_type == 'drop':
                self.driver.find_element(By.TAG_NAME, "body").send_keys('d')
                
            elif action_type == 'photo':
                self.driver.find_element(By.TAG_NAME, "body").send_keys('c')
                
            elif action_type == 'mow':
                self.driver.find_element(By.TAG_NAME, "body").send_keys('m')
                
            elif action_type == 'warn':
                self.driver.find_element(By.TAG_NAME, "body").send_keys('w')
                
            elif action_type == 'water':
                self.driver.find_element(By.TAG_NAME, "body").send_keys('t')
                
            elif action_type == 'charge':
                self.driver.find_element(By.TAG_NAME, "body").send_keys('g')
                
            elif action_type == 'refill':
                self.driver.find_element(By.TAG_NAME, "body").send_keys('f')
                
            elif action_type == 'wait':
                time.sleep(0.5)
            
            self.action_history.append({
                'timestamp': time.time(),
                'action': action,
                'success': True
            })
            
            self.game_context['total_actions'] += 1
            self.game_context['successful_actions'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
            self.action_history.append({
                'timestamp': time.time(),
                'action': action,
                'success': False,
                'error': str(e)
            })
            return False
    
    async def play_game(self, max_actions: int = 1000, action_delay: float = 1.0):
        """Main game playing loop"""
        logger.info("Starting LLM agent game session")
        
        action_count = 0
        last_score = 0
        last_robot_pos = None
        repeated_action_count = 0
        last_action = None
        
        while action_count < max_actions:
            try:
                # Get current game state
                game_state = self.state_extractor.get_game_state()
                if not game_state:
                    logger.warning("Could not extract game state")
                    time.sleep(action_delay)
                    continue
                
                # Check if game is still running
                if not game_state.get('game', {}).get('gameRunning', True):
                    logger.info("Game ended")
                    break
                
                # Capture screenshot for vision
                screenshot = self.state_extractor.capture_screenshot()
                
                # Check for stuck robot (repeated actions or no movement)
                robot_pos = game_state.get('robot', {}).get('position', {})
                
                # Detect if robot is stuck
                stuck_detection = ""
                if last_robot_pos and robot_pos:
                    distance_moved = ((robot_pos.get('x', 0) - last_robot_pos.get('x', 0))**2 + 
                                    (robot_pos.get('y', 0) - last_robot_pos.get('y', 0))**2)**0.5
                    
                    if distance_moved < 5:  # Robot hasn't moved much
                        repeated_action_count += 1
                    else:
                        repeated_action_count = 0
                

                
                # If stuck, add context to help LLM make better decision
                if repeated_action_count >= 3:
                    stuck_detection = f"\n\nWARNING: Robot appears stuck (repeated {repeated_action_count} times). Try a different approach or action!"
                    logger.warning(f"Robot appears stuck - repeated action count: {repeated_action_count}")
                    
                    # Force alternative actions when seriously stuck
                    if repeated_action_count >= 5:
                        # Try immediate task-based actions instead of movement
                        detected_animals = game_state.get('environment', {}).get('detectedAnimals', [])
                        robot_battery = game_state.get('robot', {}).get('batteryLevel', 100)
                        robot_water = game_state.get('robot', {}).get('waterLevel', 100)
                        
                        if detected_animals:
                            decision = {"action": "warn"}
                            logger.info("Force action: Warning animals due to stuck condition")
                        elif robot_battery < 30:
                            decision = {"action": "charge"}
                            logger.info("Force action: Going to charge due to stuck condition")
                        elif robot_water < 30:
                            decision = {"action": "refill"}
                            logger.info("Force action: Going to refill water due to stuck condition")
                        else:
                            # Random exploration action
                            import random
                            exploration_actions = [
                                {"action": "arrow_up"},
                                {"action": "arrow_down"},
                                {"action": "arrow_left"},
                                {"action": "arrow_right"},
                                {"action": "pickup"},
                                {"action": "photo"}
                            ]
                            decision = random.choice(exploration_actions)
                            logger.info(f"Force action: Random exploration due to stuck condition: {decision}")
                    else:
                        # Get LLM decision with stuck warning
                        modified_game_state = game_state.copy()
                        modified_game_state['stuck_warning'] = stuck_detection
                        decision = await self.get_llm_decision(modified_game_state, screenshot)
                else:
                    decision = await self.get_llm_decision(game_state, screenshot)
                
                logger.info(f"Action {action_count + 1}: {decision}")
                
                # Execute action
                success = self.execute_action(decision)
                
                # Update tracking variables for next iteration
                current_action_key = f"{decision.get('action')}_{decision.get('x')}_{decision.get('y')}"
                if current_action_key == last_action:
                    repeated_action_count += 1
                else:
                    repeated_action_count = max(0, repeated_action_count - 1)
                
                last_robot_pos = robot_pos
                last_action = current_action_key
                
                # Check score improvement
                current_score = game_state.get('game', {}).get('score', 0)
                if current_score > last_score:
                    logger.info(f"Score improved: {last_score} -> {current_score}")
                    last_score = current_score
                
                # Wait before next action
                time.sleep(action_delay)
                action_count += 1
                
            except KeyboardInterrupt:
                logger.info("Game session interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in game loop: {e}")
                time.sleep(action_delay)
                continue
        
        logger.info(f"Game session completed. Total actions: {action_count}")
        self.print_session_summary()
    
    def print_session_summary(self):
        """Print summary of the game session"""
        total_time = time.time() - self.game_context['session_start']
        success_rate = (self.game_context['successful_actions'] / 
                       max(1, self.game_context['total_actions'])) * 100
        
        print(f"""
=== LLM AGENT SESSION SUMMARY ===
Total Time: {total_time:.1f} seconds
Total Actions: {self.game_context['total_actions']}
Successful Actions: {self.game_context['successful_actions']}
Success Rate: {success_rate:.1f}%
Actions per Minute: {(self.game_context['total_actions'] / max(1, total_time/60)):.1f}
================================
        """)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.driver:
            self.driver.quit()

async def main():
    """Main function to run the LLM agent"""
    # You need to provide your OpenAI API key
    API_KEY = input("Enter your OpenAI API key: ").strip()
    
    if not API_KEY:
        print("API key required to run LLM agent")
        return
    
    agent = FarmRobotLLMAgent(API_KEY)
    
    try:
        # Setup browser and navigate to game
        agent.setup_browser("http://localhost:8000")
        
        # Start playing
        await agent.play_game(max_actions=500, action_delay=2.0)
        
    except Exception as e:
        logger.error(f"Error running agent: {e}")
    finally:
        agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 