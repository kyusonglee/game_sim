#!/usr/bin/env python3
"""
Simple AI Agent for Robot House Simulator (Python version)
A basic reinforcement learning agent that learns to play the game.
"""

import pygame
import random
import numpy as np
import time
import json
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import math

# Import the game classes from simple.py
import sys
import os
sys.path.append(os.path.dirname(__file__))

@dataclass
class State:
    """Represents the current state of the game for the agent"""
    robot_pos: Tuple[float, float]
    robot_room: str
    has_object: bool
    object_positions: List[Tuple[float, float, str, str]]  # x, y, name, color
    task_target: Optional[Tuple[str, str]]  # name, color
    task_location: Optional[str]  # target room


@dataclass
class Experience:
    """Represents an experience for learning"""
    state: State
    action: str
    reward: float
    next_state: State
    done: bool


class SimpleAgent:
    """A simple Q-learning agent for the robot simulator"""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        # Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon  # Exploration rate
        
        # Q-table (state-action values)
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Actions the agent can take
        self.actions = [
            'move_forward', 'move_backward', 
            'rotate_left', 'rotate_right',
            'pickup', 'drop'
        ]
        
        # Memory for experience replay
        self.memory = deque(maxlen=10000)
        
        # Statistics
        self.episode_rewards = []
        self.episode_steps = []
        self.success_rate = []
        
    def get_state_key(self, state: State) -> str:
        """Convert state to a string key for Q-table"""
        # Discretize position to reduce state space
        robot_x = int(state.robot_pos[0] // 50)  # Grid of 50x50 pixels
        robot_y = int(state.robot_pos[1] // 50)
        
        # Find closest object
        closest_object = None
        min_distance = float('inf')
        
        if state.object_positions and state.task_target:
            target_name, target_color = state.task_target
            for x, y, name, color in state.object_positions:
                if name == target_name and color == target_color:
                    distance = math.sqrt((state.robot_pos[0] - x)**2 + (state.robot_pos[1] - y)**2)
                    if distance < min_distance:
                        min_distance = distance
                        # Discretize object position
                        obj_x = int(x // 50)
                        obj_y = int(y // 50)
                        closest_object = (obj_x, obj_y)
        
        # Create state key
        state_key = f"{robot_x},{robot_y},{state.robot_room},{state.has_object}"
        
        if closest_object:
            state_key += f",{closest_object[0]},{closest_object[1]}"
        
        if state.task_location:
            state_key += f",{state.task_location}"
            
        return state_key
    
    def choose_action(self, state: State) -> str:
        """Choose action using epsilon-greedy strategy"""
        state_key = self.get_state_key(state)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            # Choose action with highest Q-value
            q_values = self.q_table[state_key]
            if not q_values:
                return random.choice(self.actions)
            
            best_action = max(q_values.keys(), key=lambda a: q_values[a])
            return best_action
    
    def update_q_table(self, state: State, action: str, reward: float, next_state: State, done: bool):
        """Update Q-table using Q-learning formula"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Best next action Q-value
        if done:
            next_max_q = 0
        else:
            next_q_values = self.q_table[next_state_key]
            next_max_q = max(next_q_values.values()) if next_q_values else 0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        
        self.q_table[state_key][action] = new_q
    
    def calculate_reward(self, prev_state: State, action: str, new_state: State, 
                        task_completed: bool, collision: bool) -> float:
        """Calculate reward for the action taken"""
        reward = 0
        
        # Large positive reward for completing task
        if task_completed:
            reward += 100
            return reward
        
        # Negative reward for collision
        if collision:
            reward -= 10
            return reward
        
        # Reward for picking up correct object
        if action == 'pickup' and not prev_state.has_object and new_state.has_object:
            if new_state.task_target:
                reward += 20
            else:
                reward -= 5  # Wrong object
        
        # Reward for dropping object in correct room
        if action == 'drop' and prev_state.has_object and not new_state.has_object:
            if new_state.robot_room == new_state.task_location:
                reward += 30
            else:
                reward -= 5  # Wrong room
        
        # Small reward for moving closer to target
        if new_state.task_target and new_state.object_positions:
            target_name, target_color = new_state.task_target
            
            # Find target object
            target_pos = None
            for x, y, name, color in new_state.object_positions:
                if name == target_name and color == target_color:
                    target_pos = (x, y)
                    break
            
            if target_pos:
                prev_distance = math.sqrt(
                    (prev_state.robot_pos[0] - target_pos[0])**2 + 
                    (prev_state.robot_pos[1] - target_pos[1])**2
                )
                new_distance = math.sqrt(
                    (new_state.robot_pos[0] - target_pos[0])**2 + 
                    (new_state.robot_pos[1] - target_pos[1])**2
                )
                
                if new_distance < prev_distance:
                    reward += 1  # Moving closer
                else:
                    reward -= 0.5  # Moving away
        
        # Small negative reward for time (encourage efficiency)
        reward -= 0.1
        
        return reward
    
    def save_model(self, filename: str):
        """Save the Q-table to a file"""
        # Convert defaultdict to regular dict for JSON serialization
        q_table_dict = {}
        for state_key, actions in self.q_table.items():
            q_table_dict[state_key] = dict(actions)
        
        model_data = {
            'q_table': q_table_dict,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps
        }
        
        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"💾 Model saved to {filename}")
    
    def load_model(self, filename: str):
        """Load the Q-table from a file"""
        try:
            with open(filename, 'r') as f:
                model_data = json.load(f)
            
            # Reconstruct Q-table
            self.q_table = defaultdict(lambda: defaultdict(float))
            for state_key, actions in model_data['q_table'].items():
                for action, q_value in actions.items():
                    self.q_table[state_key][action] = q_value
            
            self.learning_rate = model_data.get('learning_rate', 0.1)
            self.discount_factor = model_data.get('discount_factor', 0.95)
            self.epsilon = model_data.get('epsilon', 0.1)
            self.episode_rewards = model_data.get('episode_rewards', [])
            self.episode_steps = model_data.get('episode_steps', [])
            
            print(f"📂 Model loaded from {filename}")
            return True
            
        except FileNotFoundError:
            print(f"❌ Model file {filename} not found")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False


class GameInterface:
    """Interface between the agent and the game"""
    
    def __init__(self):
        # Import game modules
        from simple import (
            generate_floor_plan, generate_furniture, generate_objects, 
            generate_task, move_robot, check_task_completion, Robot, 
            draw_house_structure, draw_robot, draw_object, draw_hud,
            screen, clock, font, COLORS
        )
        
        self.generate_floor_plan = generate_floor_plan
        self.generate_furniture = generate_furniture
        self.generate_objects = generate_objects
        self.generate_task = generate_task
        self.move_robot = move_robot
        self.check_task_completion = check_task_completion
        self.Robot = Robot
        self.draw_house_structure = draw_house_structure
        self.draw_robot = draw_robot
        self.draw_object = draw_object
        self.draw_hud = draw_hud
        self.screen = screen
        self.clock = clock
        self.font = font
        self.COLORS = COLORS
        
        self.reset_game()
    
    def reset_game(self):
        """Reset the game to initial state"""
        # Generate new level
        self.rooms, self.doors = self.generate_floor_plan()
        self.furniture = self.generate_furniture(self.rooms)
        self.objects = self.generate_objects(self.rooms, level=1)
        self.robot = self.Robot(self.rooms[0], self.furniture)
        self.task = self.generate_task(self.objects, self.rooms, self.furniture, level=1)
        
        self.score = 0
        self.level = 1
        self.start_time = time.time()
        self.last_collision = False
        self.task_completed = False
    
    def get_current_state(self) -> State:
        """Get current game state"""
        # Get object positions
        object_positions = []
        for obj in self.objects:
            object_positions.append((obj.pos[0], obj.pos[1], obj.name, obj.color))
        
        # Get task target
        task_target = None
        if self.task and hasattr(self.task, 'target_object'):
            task_target = (self.task.target_object.name, self.task.target_object.color)
        
        # Get task location
        task_location = None
        if self.task and hasattr(self.task, 'target_room'):
            task_location = self.task.target_room.name
        
        return State(
            robot_pos=(self.robot.pos[0], self.robot.pos[1]),
            robot_room=self.robot.room.name,
            has_object=self.robot.has_object is not None,
            object_positions=object_positions,
            task_target=task_target,
            task_location=task_location
        )
    
    def execute_action(self, action: str) -> Tuple[bool, bool]:
        """Execute action and return (collision, task_completed)"""
        collision = False
        
        if action == 'move_forward':
            result = self.move_robot(self.robot, 0, -5, self.rooms, self.doors, self.furniture)
            collision = not result
        elif action == 'move_backward':
            result = self.move_robot(self.robot, 0, 5, self.rooms, self.doors, self.furniture)
            collision = not result
        elif action == 'rotate_left':
            # Rotation is handled differently - no collision possible
            pass
        elif action == 'rotate_right':
            # Rotation is handled differently - no collision possible
            pass
        elif action == 'pickup':
            # Try to pick up nearby object
            for obj in self.objects:
                distance = math.sqrt(
                    (self.robot.pos[0] - obj.pos[0])**2 + 
                    (self.robot.pos[1] - obj.pos[1])**2
                )
                if distance < 30 and self.robot.has_object is None:
                    self.robot.has_object = obj
                    self.objects.remove(obj)
                    break
        elif action == 'drop':
            # Drop object if carrying one
            if self.robot.has_object:
                self.robot.has_object.pos = self.robot.pos.copy()
                self.robot.has_object.room = self.robot.room
                self.objects.append(self.robot.has_object)
                self.robot.has_object = None
        
        # Check task completion
        task_completed = self.check_task_completion(self.robot, self.task, self.rooms)
        
        if task_completed:
            self.score += 100
            self.task_completed = True
        
        return collision, task_completed
    
    def render(self, show_agent_info=True):
        """Render the current game state"""
        # Clear screen
        self.screen.fill(self.COLORS['gray'])
        
        # Draw game elements
        self.draw_house_structure(self.rooms, self.doors, self.furniture)
        
        for obj in self.objects:
            self.draw_object(obj)
        
        self.draw_robot(self.robot)
        
        # Draw HUD
        elapsed_time = time.time() - self.start_time
        self.draw_hud(self.score, elapsed_time, self.level)
        
        # Draw task info
        if self.task and hasattr(self.task, 'description'):
            task_text = self.font.render(f"Task: {self.task.description}", True, (255, 255, 255))
            self.screen.blit(task_text, (10, 50))
        
        if show_agent_info:
            # Draw AI agent indicator
            ai_text = self.font.render("🤖 AI AGENT PLAYING", True, (0, 255, 0))
            self.screen.blit(ai_text, (10, 10))
        
        pygame.display.flip()


def train_agent(episodes=100, render=True, save_interval=10):
    """Train the agent for a specified number of episodes"""
    print(f"🎓 Training agent for {episodes} episodes...")
    
    # Initialize
    agent = SimpleAgent()
    game = GameInterface()
    
    # Try to load existing model
    agent.load_model("robot_agent_model.json")
    
    for episode in range(episodes):
        print(f"\n📚 Episode {episode + 1}/{episodes}")
        
        # Reset game
        game.reset_game()
        
        episode_reward = 0
        episode_steps = 0
        max_steps = 500  # Prevent infinite loops
        
        prev_state = game.get_current_state()
        
        for step in range(max_steps):
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Choose and execute action
            action = agent.choose_action(prev_state)
            collision, task_completed = game.execute_action(action)
            
            # Get new state
            new_state = game.get_current_state()
            
            # Calculate reward
            reward = agent.calculate_reward(prev_state, action, new_state, task_completed, collision)
            episode_reward += reward
            episode_steps += 1
            
            # Update Q-table
            agent.update_q_table(prev_state, action, reward, new_state, task_completed)
            
            # Render if requested
            if render and episode % 5 == 0:  # Render every 5th episode
                game.render()
                game.clock.tick(30)  # Limit FPS for visualization
            
            # Check if episode is done
            if task_completed or step >= max_steps - 1:
                break
            
            prev_state = new_state
        
        # Record episode statistics
        agent.episode_rewards.append(episode_reward)
        agent.episode_steps.append(episode_steps)
        
        # Calculate success rate (last 10 episodes)
        if len(agent.episode_rewards) >= 10:
            recent_successes = sum(1 for r in agent.episode_rewards[-10:] if r > 50)
            success_rate = recent_successes / 10
            agent.success_rate.append(success_rate)
        
        # Print episode results
        print(f"   Reward: {episode_reward:.1f}, Steps: {episode_steps}, Task: {'✅' if task_completed else '❌'}")
        
        if len(agent.success_rate) > 0:
            print(f"   Success Rate (last 10): {agent.success_rate[-1]:.1%}")
        
        # Save model periodically
        if (episode + 1) % save_interval == 0:
            agent.save_model("robot_agent_model.json")
        
        # Decay epsilon (reduce exploration over time)
        if agent.epsilon > 0.01:
            agent.epsilon *= 0.995
    
    # Final save
    agent.save_model("robot_agent_model.json")
    
    print(f"\n🏆 Training Complete!")
    print(f"   Final Success Rate: {agent.success_rate[-1]:.1%}" if agent.success_rate else "   No success rate data")
    print(f"   Average Reward (last 10): {np.mean(agent.episode_rewards[-10:]):.1f}")
    print(f"   Q-table size: {len(agent.q_table)} states")


def play_with_agent(episodes=5):
    """Watch the trained agent play"""
    print(f"🎮 Watching trained agent play for {episodes} episodes...")
    
    # Initialize
    agent = SimpleAgent()
    game = GameInterface()
    
    # Load trained model
    if not agent.load_model("robot_agent_model.json"):
        print("❌ No trained model found. Please train the agent first.")
        return
    
    # Set epsilon to 0 for pure exploitation
    agent.epsilon = 0
    
    for episode in range(episodes):
        print(f"\n🎯 Demo Episode {episode + 1}/{episodes}")
        
        # Reset game
        game.reset_game()
        
        episode_reward = 0
        episode_steps = 0
        max_steps = 500
        
        state = game.get_current_state()
        
        for step in range(max_steps):
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Choose action (no exploration)
            action = agent.choose_action(state)
            collision, task_completed = game.execute_action(action)
            
            # Update state
            state = game.get_current_state()
            episode_steps += 1
            
            # Render game
            game.render(show_agent_info=True)
            game.clock.tick(10)  # Slower for better visualization
            
            # Check if episode is done
            if task_completed:
                print(f"   ✅ Task completed in {episode_steps} steps!")
                time.sleep(2)  # Pause to show success
                break
            elif step >= max_steps - 1:
                print(f"   ❌ Task not completed (timeout)")
                break
        
        time.sleep(1)  # Pause between episodes


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Agent for Robot House Simulator")
    parser.add_argument('--mode', choices=['train', 'play'], default='train',
                       help='Mode: train the agent or watch it play')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering during training')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_agent(episodes=args.episodes, render=not args.no_render)
    elif args.mode == 'play':
        play_with_agent(episodes=args.episodes)


if __name__ == "__main__":
    main() 