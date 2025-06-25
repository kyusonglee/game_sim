#!/usr/bin/env python3
"""
Training Data Analysis Script for Robot Simulator
Analyzes collected gameplay data for training insights and ML preparation.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import argparse
from collections import Counter, defaultdict

class TrainingDataAnalyzer:
    def __init__(self, logs_dir="training_logs"):
        self.logs_dir = Path(logs_dir)
        self.sessions = []
        self.actions_df = None
        self.states_df = None
        
    def load_all_sessions(self):
        """Load all training sessions from logs directory"""
        if not self.logs_dir.exists():
            print(f"❌ Logs directory {self.logs_dir} does not exist")
            return
        
        session_dirs = [d for d in self.logs_dir.iterdir() if d.is_dir()]
        print(f"📁 Found {len(session_dirs)} training sessions")
        
        all_actions = []
        all_states = []
        
        for session_dir in session_dirs:
            session_id = session_dir.name
            session_data = {"session_id": session_id, "actions": [], "states": []}
            
            # Load actions
            actions_file = session_dir / 'actions.jsonl'
            if actions_file.exists():
                with open(actions_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            action = json.loads(line)
                            action['session_id'] = session_id
                            session_data["actions"].append(action)
                            all_actions.append(action)
            
            # Load states
            states_file = session_dir / 'states.jsonl'
            if states_file.exists():
                with open(states_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            state = json.loads(line)
                            state['session_id'] = session_id
                            session_data["states"].append(state)
                            all_states.append(state)
            
            self.sessions.append(session_data)
        
        # Convert to DataFrames for analysis
        if all_actions:
            self.actions_df = pd.json_normalize(all_actions)
        if all_states:
            self.states_df = pd.json_normalize(all_states)
        
        print(f"📊 Loaded {len(all_actions)} actions and {len(all_states)} states")
    
    def analyze_gameplay_patterns(self):
        """Analyze gameplay patterns and behaviors"""
        if self.actions_df is None or len(self.actions_df) == 0:
            print("❌ No action data available for analysis")
            return
        
        print("\n🎮 GAMEPLAY PATTERN ANALYSIS")
        print("=" * 50)
        
        # Action frequency analysis
        action_counts = self.actions_df['actionType'].value_counts()
        print("\n📈 Action Frequency:")
        for action, count in action_counts.items():
            print(f"   {action}: {count}")
        
        # Success rates
        if 'pickup_success' in action_counts.index and 'pickup_failed' in action_counts.index:
            pickup_success_rate = action_counts['pickup_success'] / (action_counts['pickup_success'] + action_counts['pickup_failed'])
            print(f"\n🎯 Pickup Success Rate: {pickup_success_rate:.2%}")
        
        # Movement patterns
        movement_actions = ['move_forward', 'move_backward', 'rotate_left', 'rotate_right']
        movement_counts = {action: action_counts.get(action, 0) for action in movement_actions}
        total_movements = sum(movement_counts.values())
        
        print(f"\n🚶 Movement Distribution:")
        for action, count in movement_counts.items():
            percentage = (count / total_movements * 100) if total_movements > 0 else 0
            print(f"   {action.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        # Collision analysis
        collision_actions = ['collision_boundary', 'collision_furniture', 'collision_wall']
        collision_counts = {action: action_counts.get(action, 0) for action in collision_actions}
        total_collisions = sum(collision_counts.values())
        
        print(f"\n💥 Collision Analysis:")
        print(f"   Total Collisions: {total_collisions}")
        for action, count in collision_counts.items():
            percentage = (count / total_collisions * 100) if total_collisions > 0 else 0
            print(f"   {action.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    def analyze_learning_progression(self):
        """Analyze how players improve over time"""
        if self.actions_df is None:
            return
        
        print("\n📚 LEARNING PROGRESSION ANALYSIS")
        print("=" * 50)
        
        # Task completion analysis by session
        session_completions = {}
        session_times = {}
        
        for session in self.sessions:
            session_id = session['session_id']
            actions = session['actions']
            
            completed_tasks = [a for a in actions if a.get('actionType') == 'task_completed']
            total_time = max([a.get('gameTime', 0) for a in actions], default=0)
            
            session_completions[session_id] = len(completed_tasks)
            session_times[session_id] = total_time
        
        avg_completions = np.mean(list(session_completions.values()))
        avg_time = np.mean(list(session_times.values()))
        
        print(f"📊 Average completions per session: {avg_completions:.2f}")
        print(f"⏱️  Average session time: {avg_time:.1f}s")
        
        # Efficiency over time (completions per minute)
        efficiencies = []
        for session_id in session_completions:
            if session_times[session_id] > 0:
                efficiency = (session_completions[session_id] / session_times[session_id]) * 60
                efficiencies.append(efficiency)
        
        if efficiencies:
            print(f"⚡ Average efficiency: {np.mean(efficiencies):.2f} completions/minute")
    
    def analyze_spatial_behavior(self):
        """Analyze spatial movement patterns"""
        if self.states_df is None:
            return
        
        print("\n🗺️  SPATIAL BEHAVIOR ANALYSIS")
        print("=" * 50)
        
        # Room usage patterns
        room_visits = self.states_df['robot.room'].value_counts()
        print("\n🏠 Room Usage:")
        for room, visits in room_visits.items():
            print(f"   {room}: {visits} visits")
        
        # Position heatmap data
        positions_x = self.states_df['robot.position.x'].dropna()
        positions_y = self.states_df['robot.position.y'].dropna()
        
        print(f"\n📍 Position Statistics:")
        print(f"   X range: {positions_x.min():.1f} - {positions_x.max():.1f}")
        print(f"   Y range: {positions_y.min():.1f} - {positions_y.max():.1f}")
        print(f"   Total position samples: {len(positions_x)}")
    
    def analyze_visual_data(self):
        """Analyze screenshot and visual data"""
        if self.actions_df is None and self.states_df is None:
            return
        
        print("\n📸 VISUAL DATA ANALYSIS")
        print("=" * 50)
        
        # Count screenshots in actions and states
        action_screenshots = 0
        state_screenshots = 0
        
        if self.actions_df is not None:
            action_screenshots = self.actions_df['screenshot'].notna().sum()
        
        if self.states_df is not None:
            state_screenshots = self.states_df['screenshot'].notna().sum()
        
        total_screenshots = action_screenshots + state_screenshots
        
        print(f"📷 Total Screenshots: {total_screenshots}")
        print(f"   Action Screenshots: {action_screenshots}")
        print(f"   State Screenshots: {state_screenshots}")
        
        # Analyze object visibility and positions
        if self.states_df is not None and 'objectPositions.all_objects' in self.states_df.columns:
            print(f"\n🎯 Object Position Analysis:")
            
            # Sample object position data
            object_position_samples = self.states_df['objectPositions.all_objects'].dropna()
            if len(object_position_samples) > 0:
                sample_objects = object_position_samples.iloc[0] if len(object_position_samples) > 0 else []
                if isinstance(sample_objects, list) and len(sample_objects) > 0:
                    print(f"   Objects per game: {len(sample_objects)}")
                    
                    # Analyze object types
                    object_types = {}
                    object_colors = {}
                    
                    for entry in object_position_samples:
                        if isinstance(entry, list):
                            for obj in entry:
                                if isinstance(obj, dict):
                                    obj_type = obj.get('name', '').split()[-1] if obj.get('name') else 'unknown'
                                    obj_color = obj.get('color', 'unknown')
                                    
                                    object_types[obj_type] = object_types.get(obj_type, 0) + 1
                                    object_colors[obj_color] = object_colors.get(obj_color, 0) + 1
                    
                    print(f"   Most common object types: {sorted(object_types.items(), key=lambda x: x[1], reverse=True)[:5]}")
                    print(f"   Most common colors: {sorted(object_colors.items(), key=lambda x: x[1], reverse=True)[:5]}")
    
    def analyze_slam_data(self):
        """Analyze SLAM (Simultaneous Localization and Mapping) data"""
        if self.actions_df is None and self.states_df is None:
            return
        
        print("\n🗺️  SLAM DATA ANALYSIS")
        print("=" * 50)
        
        # Analyze occupancy grids
        print("📊 Occupancy Grid Analysis:")
        slam_entries = []
        
        # Collect SLAM data from actions and states
        for session in self.sessions:
            for action in session['actions']:
                if 'slam_data' in action and action['slam_data']:
                    slam_entries.append(action['slam_data'])
            
            for state in session['states']:
                if 'slam_data' in state and state['slam_data']:
                    slam_entries.append(state['slam_data'])
        
        if len(slam_entries) > 0:
            print(f"   Total SLAM observations: {len(slam_entries)}")
            
            # Analyze occupancy grid evolution
            self.analyze_occupancy_grids(slam_entries)
            
            # Analyze landmark detection
            self.analyze_landmarks(slam_entries)
            
            # Analyze range sensor data
            self.analyze_range_data(slam_entries)
            
            # Analyze trajectory and path planning
            self.analyze_trajectories(slam_entries)
            
            # Analyze topological mapping
            self.analyze_topological_maps(slam_entries)
        else:
            print("   No SLAM data found in logs")
    
    def analyze_occupancy_grids(self, slam_entries):
        """Analyze occupancy grid data for mapping quality"""
        print(f"\n📋 Occupancy Grid Analysis:")
        
        grids = [entry['occupancy_grid'] for entry in slam_entries if 'occupancy_grid' in entry]
        if not grids:
            return
        
        grid_width = grids[0]['width']
        grid_height = grids[0]['height']
        resolution = grids[0]['resolution']
        
        print(f"   Grid dimensions: {grid_width} x {grid_height} cells")
        print(f"   Resolution: {resolution} pixels/cell")
        
        # Analyze grid coverage and exploration
        total_cells = grid_width * grid_height
        occupied_ratios = []
        free_ratios = []
        unknown_ratios = []
        
        for grid in grids:
            data = grid['data']
            occupied = sum(1 for cell in data if cell == 100)
            free = sum(1 for cell in data if cell == 0)
            unknown = sum(1 for cell in data if cell == -1)
            
            occupied_ratios.append(occupied / total_cells)
            free_ratios.append(free / total_cells)
            unknown_ratios.append(unknown / total_cells)
        
        avg_occupied = np.mean(occupied_ratios) * 100
        avg_free = np.mean(free_ratios) * 100
        avg_unknown = np.mean(unknown_ratios) * 100
        
        print(f"   Average cell distribution:")
        print(f"     Occupied: {avg_occupied:.1f}%")
        print(f"     Free space: {avg_free:.1f}%")
        print(f"     Unknown: {avg_unknown:.1f}%")
        print(f"   Exploration efficiency: {100 - avg_unknown:.1f}%")
    
    def analyze_landmarks(self, slam_entries):
        """Analyze landmark detection and tracking"""
        print(f"\n🎯 Landmark Analysis:")
        
        all_landmarks = []
        for entry in slam_entries:
            if 'landmarks' in entry:
                all_landmarks.extend(entry['landmarks'])
        
        if not all_landmarks:
            print("   No landmarks detected")
            return
        
        # Analyze landmark types
        landmark_types = {}
        confidence_scores = []
        distances = []
        
        for landmark in all_landmarks:
            ltype = landmark.get('type', 'unknown')
            landmark_types[ltype] = landmark_types.get(ltype, 0) + 1
            
            if 'confidence' in landmark:
                confidence_scores.append(landmark['confidence'])
            
            if 'distance' in landmark:
                distances.append(landmark['distance'])
        
        print(f"   Total landmarks detected: {len(all_landmarks)}")
        print(f"   Landmark types: {dict(sorted(landmark_types.items(), key=lambda x: x[1], reverse=True))}")
        
        if confidence_scores:
            print(f"   Average confidence: {np.mean(confidence_scores):.3f}")
            print(f"   Confidence std: {np.std(confidence_scores):.3f}")
        
        if distances:
            print(f"   Average detection distance: {np.mean(distances):.1f} pixels")
            print(f"   Max detection distance: {np.max(distances):.1f} pixels")
    
    def analyze_range_data(self, slam_entries):
        """Analyze simulated range sensor (lidar) data"""
        print(f"\n📡 Range Sensor Analysis:")
        
        range_readings = []
        for entry in slam_entries:
            if 'range_data' in entry and 'ranges' in entry['range_data']:
                for reading in entry['range_data']['ranges']:
                    if 'range' in reading:
                        range_readings.append(reading['range'])
        
        if not range_readings:
            print("   No range data found")
            return
        
        print(f"   Total range measurements: {len(range_readings)}")
        print(f"   Average range: {np.mean(range_readings):.1f} pixels")
        print(f"   Range std: {np.std(range_readings):.1f} pixels")
        print(f"   Min range: {np.min(range_readings):.1f} pixels")
        print(f"   Max range: {np.max(range_readings):.1f} pixels")
        
        # Analyze obstacle density
        max_range = 150  # From the simulation
        obstacle_hits = sum(1 for r in range_readings if r < max_range)
        obstacle_ratio = obstacle_hits / len(range_readings) * 100
        print(f"   Obstacle detection rate: {obstacle_ratio:.1f}%")
    
    def analyze_trajectories(self, slam_entries):
        """Analyze robot trajectories and path efficiency"""
        print(f"\n🛤️  Trajectory Analysis:")
        
        trajectories = []
        path_efficiencies = []
        total_distances = []
        
        for entry in slam_entries:
            if 'trajectory' in entry:
                traj = entry['trajectory']
                trajectories.append(traj)
                
                if 'path_efficiency' in traj:
                    path_efficiencies.append(traj['path_efficiency'])
                
                if 'total_distance' in traj:
                    total_distances.append(traj['total_distance'])
        
        if not trajectories:
            print("   No trajectory data found")
            return
        
        print(f"   Trajectory samples: {len(trajectories)}")
        
        if path_efficiencies:
            avg_efficiency = np.mean(path_efficiencies)
            print(f"   Average path efficiency: {avg_efficiency:.3f}")
            print(f"   Best path efficiency: {np.max(path_efficiencies):.3f}")
            print(f"   Worst path efficiency: {np.min(path_efficiencies):.3f}")
        
        if total_distances:
            print(f"   Average total distance: {np.mean(total_distances):.1f} pixels")
            print(f"   Max distance traveled: {np.max(total_distances):.1f} pixels")
        
        # Analyze path lengths
        path_lengths = []
        for traj in trajectories:
            if 'path' in traj:
                path_lengths.append(len(traj['path']))
        
        if path_lengths:
            print(f"   Average path length: {np.mean(path_lengths):.1f} waypoints")
            print(f"   Longest path: {np.max(path_lengths)} waypoints")
    
    def analyze_topological_maps(self, slam_entries):
        """Analyze topological mapping data"""
        print(f"\n🕸️  Topological Map Analysis:")
        
        topo_maps = [entry['topological_map'] for entry in slam_entries 
                    if 'topological_map' in entry]
        
        if not topo_maps:
            print("   No topological map data found")
            return
        
        # Analyze map structure
        node_counts = []
        edge_counts = []
        room_types = set()
        
        for tmap in topo_maps:
            if 'nodes' in tmap:
                node_counts.append(len(tmap['nodes']))
                for node in tmap['nodes']:
                    if 'name' in node:
                        room_types.add(node['name'])
            
            if 'edges' in tmap:
                edge_counts.append(len(tmap['edges']))
        
        print(f"   Map samples: {len(topo_maps)}")
        if node_counts:
            print(f"   Average nodes (rooms): {np.mean(node_counts):.1f}")
            print(f"   Room types discovered: {sorted(list(room_types))}")
        
        if edge_counts:
            print(f"   Average edges (connections): {np.mean(edge_counts):.1f}")
            
        # Analyze connectivity
        if edge_counts and node_counts:
            connectivity_ratios = [edges / max(nodes, 1) for edges, nodes in zip(edge_counts, node_counts)]
            print(f"   Average connectivity ratio: {np.mean(connectivity_ratios):.2f}")
    
    def export_ml_dataset(self, output_file="robot_ml_dataset.json"):
        """Export data in format suitable for machine learning"""
        if not self.actions_df is not None or not self.states_df is not None:
            print("❌ No data available for ML export")
            return
        
        ml_dataset = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "total_sessions": len(self.sessions),
                "total_actions": len(self.actions_df) if self.actions_df is not None else 0,
                "total_states": len(self.states_df) if self.states_df is not None else 0,
                "action_types": list(self.actions_df['actionType'].unique()) if self.actions_df is not None else [],
                "features": [
                    "robot_position", "robot_angle", "robot_room", "has_object",
                    "game_score", "level", "task_type", "collision_history"
                ]
            },
            "training_sequences": []
        }
        
        # Create training sequences by session
        for session in self.sessions:
            session_actions = [a for a in session['actions']]
            session_states = [s for s in session['states']]
            
            if not session_actions or not session_states:
                continue
            
            sequence = {
                "session_id": session['session_id'],
                "sequence_length": len(session_actions),
                "actions": session_actions,
                "states": session_states,
                "features": self.extract_ml_features(session_actions, session_states)
            }
            
            ml_dataset["training_sequences"].append(sequence)
        
        # Save ML dataset
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(ml_dataset, f, indent=2)
        
        print(f"🤖 ML dataset exported to: {output_path.absolute()}")
        print(f"📊 Dataset contains {len(ml_dataset['training_sequences'])} training sequences")
    
    def extract_ml_features(self, actions, states):
        """Extract machine learning features from session data"""
        features = {
            "movement_patterns": {},
            "decision_points": [],
            "success_metrics": {},
            "spatial_coverage": {}
        }
        
        # Movement pattern features
        movement_actions = [a for a in actions if a.get('actionType') in ['move_forward', 'move_backward', 'rotate_left', 'rotate_right']]
        features["movement_patterns"] = {
            "total_movements": len(movement_actions),
            "forward_ratio": len([a for a in movement_actions if a.get('actionType') == 'move_forward']) / max(len(movement_actions), 1),
            "rotation_frequency": len([a for a in movement_actions if 'rotate' in a.get('actionType', '')]) / max(len(movement_actions), 1)
        }
        
        # Success metrics
        successful_pickups = len([a for a in actions if a.get('actionType') == 'pickup_success'])
        failed_pickups = len([a for a in actions if a.get('actionType') == 'pickup_failed'])
        task_completions = len([a for a in actions if a.get('actionType') == 'task_completed'])
        
        features["success_metrics"] = {
            "pickup_success_rate": successful_pickups / max(successful_pickups + failed_pickups, 1),
            "task_completions": task_completions,
            "collision_count": len([a for a in actions if 'collision' in a.get('actionType', '')])
        }
        
        # Spatial coverage
        if states:
            positions_x = [s.get('robot', {}).get('position', {}).get('x') for s in states if s.get('robot', {}).get('position')]
            positions_y = [s.get('robot', {}).get('position', {}).get('y') for s in states if s.get('robot', {}).get('position')]
            
            if positions_x and positions_y:
                features["spatial_coverage"] = {
                    "x_range": max(positions_x) - min(positions_x),
                    "y_range": max(positions_y) - min(positions_y),
                    "position_variance_x": np.var(positions_x),
                    "position_variance_y": np.var(positions_y)
                }
        
        return features
    
    def export_visual_dataset(self, output_dir="visual_training_data"):
        """Export visual training dataset for computer vision applications"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n🖼️  EXPORTING VISUAL DATASET to {output_path.absolute()}")
        
        # Create subdirectories
        (output_path / "images").mkdir(exist_ok=True)
        (output_path / "annotations").mkdir(exist_ok=True)
        (output_path / "metadata").mkdir(exist_ok=True)
        
        image_count = 0
        annotation_count = 0
        
        for session in self.sessions:
            session_id = session['session_id']
            
            # Process actions with screenshots
            for i, action in enumerate(session['actions']):
                if action.get('screenshot'):
                    image_count += 1
                    
                    # Create annotation data
                    annotation = {
                        "image_id": f"{session_id}_action_{i}",
                        "session_id": session_id,
                        "type": "action",
                        "action_type": action.get('actionType'),
                        "timestamp": action.get('timestamp'),
                        "game_time": action.get('gameTime'),
                        "robot_state": action.get('robotState'),
                        "object_positions": action.get('objectPositions'),
                        "environment": action.get('environment'),
                        "task": action.get('gameState', {}).get('taskInstruction')
                    }
                    
                    # Save annotation
                    annotation_file = output_path / "annotations" / f"{session_id}_action_{i}.json"
                    with open(annotation_file, 'w') as f:
                        json.dump(annotation, f, indent=2)
                    
                    annotation_count += 1
            
            # Process states with screenshots
            for i, state in enumerate(session['states']):
                if state.get('screenshot'):
                    image_count += 1
                    
                    # Create annotation data
                    annotation = {
                        "image_id": f"{session_id}_state_{i}",
                        "session_id": session_id,
                        "type": "state",
                        "timestamp": state.get('timestamp'),
                        "game_time": state.get('gameTime'),
                        "robot_state": state.get('robot'),
                        "object_positions": state.get('objectPositions'),
                        "environment": state.get('environment'),
                        "task": state.get('task')
                    }
                    
                    # Save annotation
                    annotation_file = output_path / "annotations" / f"{session_id}_state_{i}.json"
                    with open(annotation_file, 'w') as f:
                        json.dump(annotation, f, indent=2)
                    
                    annotation_count += 1
        
        # Create dataset metadata
        dataset_metadata = {
            "created": datetime.now().isoformat(),
            "total_sessions": len(self.sessions),
            "total_images": image_count,
            "total_annotations": annotation_count,
            "description": "Robot Simulator Visual Training Dataset",
            "format": {
                "images": "PNG files referenced by screenshot paths in logs",
                "annotations": "JSON files with robot state, object positions, and task information",
                "coordinate_system": "Canvas coordinates (0,0 at top-left, 800x600 resolution)"
            },
            "classes": {
                "objects": self.get_object_classes(),
                "rooms": self.get_room_classes(),
                "furniture": self.get_furniture_classes()
            }
        }
        
        metadata_file = output_path / "metadata" / "dataset_info.json"
        with open(metadata_file, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        print(f"✅ Visual dataset exported successfully!")
        print(f"   📊 {annotation_count} annotations created")
        print(f"   🖼️  {image_count} images referenced")
        print(f"   📁 Dataset structure created in {output_path}")
        
        return str(output_path.absolute())
    
    def export_slam_dataset(self, output_dir="slam_training_data"):
        """Export SLAM-specific training dataset"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n🗺️  EXPORTING SLAM DATASET to {output_path.absolute()}")
        
        # Create subdirectories
        (output_path / "occupancy_grids").mkdir(exist_ok=True)
        (output_path / "trajectories").mkdir(exist_ok=True)
        (output_path / "landmarks").mkdir(exist_ok=True)
        (output_path / "range_data").mkdir(exist_ok=True)
        (output_path / "topological_maps").mkdir(exist_ok=True)
        
        slam_count = 0
        
        for session in self.sessions:
            session_id = session['session_id']
            
            # Process SLAM data from actions and states
            all_slam_data = []
            
            for i, action in enumerate(session['actions']):
                if 'slam_data' in action and action['slam_data']:
                    slam_entry = {
                        "id": f"{session_id}_action_{i}",
                        "type": "action",
                        "timestamp": action.get('timestamp'),
                        "game_time": action.get('gameTime'),
                        "action_type": action.get('actionType'),
                        **action['slam_data']
                    }
                    all_slam_data.append(slam_entry)
            
            for i, state in enumerate(session['states']):
                if 'slam_data' in state and state['slam_data']:
                    slam_entry = {
                        "id": f"{session_id}_state_{i}",
                        "type": "state", 
                        "timestamp": state.get('timestamp'),
                        "game_time": state.get('gameTime'),
                        **state['slam_data']
                    }
                    all_slam_data.append(slam_entry)
            
            # Save SLAM data components separately
            for slam_data in all_slam_data:
                slam_id = slam_data['id']
                slam_count += 1
                
                # Save occupancy grid
                if 'occupancy_grid' in slam_data:
                    grid_file = output_path / "occupancy_grids" / f"{slam_id}.json"
                    with open(grid_file, 'w') as f:
                        json.dump({
                            "id": slam_id,
                            "robot_pose": slam_data.get('robot_pose'),
                            "occupancy_grid": slam_data['occupancy_grid']
                        }, f, indent=2)
                
                # Save trajectory data
                if 'trajectory' in slam_data:
                    traj_file = output_path / "trajectories" / f"{slam_id}.json"
                    with open(traj_file, 'w') as f:
                        json.dump({
                            "id": slam_id,
                            "robot_pose": slam_data.get('robot_pose'),
                            "trajectory": slam_data['trajectory']
                        }, f, indent=2)
                
                # Save landmarks
                if 'landmarks' in slam_data:
                    landmarks_file = output_path / "landmarks" / f"{slam_id}.json"
                    with open(landmarks_file, 'w') as f:
                        json.dump({
                            "id": slam_id,
                            "robot_pose": slam_data.get('robot_pose'),
                            "landmarks": slam_data['landmarks']
                        }, f, indent=2)
                
                # Save range data
                if 'range_data' in slam_data:
                    range_file = output_path / "range_data" / f"{slam_id}.json"
                    with open(range_file, 'w') as f:
                        json.dump({
                            "id": slam_id,
                            "robot_pose": slam_data.get('robot_pose'),
                            "range_data": slam_data['range_data']
                        }, f, indent=2)
                
                # Save topological map
                if 'topological_map' in slam_data:
                    topo_file = output_path / "topological_maps" / f"{slam_id}.json"
                    with open(topo_file, 'w') as f:
                        json.dump({
                            "id": slam_id,
                            "robot_pose": slam_data.get('robot_pose'),
                            "topological_map": slam_data['topological_map']
                        }, f, indent=2)
        
        # Create SLAM dataset metadata
        slam_metadata = {
            "created": datetime.now().isoformat(),
            "total_sessions": len(self.sessions),
            "total_slam_observations": slam_count,
            "description": "Robot Simulator SLAM Training Dataset",
            "components": {
                "occupancy_grids": "2D grid maps with occupied/free/unknown cells",
                "trajectories": "Robot path history with efficiency metrics",
                "landmarks": "Detected environmental features for localization",
                "range_data": "Simulated lidar sensor readings",
                "topological_maps": "High-level room connectivity graphs"
            },
            "coordinate_system": "Canvas coordinates (0,0 at top-left, 800x600 resolution)",
            "applications": [
                "SLAM algorithm training",
                "Path planning optimization",
                "Landmark detection",
                "Occupancy grid mapping",
                "Topological navigation"
            ]
        }
        
        metadata_file = output_path / "slam_dataset_info.json"
        with open(metadata_file, 'w') as f:
            json.dump(slam_metadata, f, indent=2)
        
        print(f"✅ SLAM dataset exported successfully!")
        print(f"   🗺️  {slam_count} SLAM observations processed")
        print(f"   📊 Components: occupancy grids, trajectories, landmarks, range data, topological maps")
        print(f"   📁 Dataset ready for SLAM research at {output_path}")
        
        return str(output_path.absolute())
    
    def get_object_classes(self):
        """Extract unique object classes from the data"""
        object_types = set()
        object_colors = set()
        
        for session in self.sessions:
            for action in session['actions']:
                obj_positions = action.get('objectPositions', {})
                if isinstance(obj_positions, dict) and 'all_objects' in obj_positions:
                    for obj in obj_positions['all_objects']:
                        if isinstance(obj, dict):
                            if 'name' in obj:
                                obj_type = obj['name'].split()[-1]
                                object_types.add(obj_type)
                            if 'color' in obj:
                                object_colors.add(obj['color'])
        
        return {
            "object_types": sorted(list(object_types)),
            "object_colors": sorted(list(object_colors))
        }
    
    def get_room_classes(self):
        """Extract unique room classes from the data"""
        room_names = set()
        
        for session in self.sessions:
            for action in session['actions']:
                env = action.get('environment', {})
                if isinstance(env, dict) and 'rooms' in env:
                    for room in env['rooms']:
                        if isinstance(room, dict) and 'name' in room:
                            room_names.add(room['name'])
        
        return sorted(list(room_names))
    
    def get_furniture_classes(self):
        """Extract unique furniture classes from the data"""
        furniture_types = set()
        
        for session in self.sessions:
            for action in session['actions']:
                env = action.get('environment', {})
                if isinstance(env, dict) and 'furniture' in env:
                    for furniture in env['furniture']:
                        if isinstance(furniture, dict) and 'name' in furniture:
                            furniture_types.add(furniture['name'])
        
        return sorted(list(furniture_types))
    
    def create_training_readme(self, output_dir="visual_training_data"):
        """Create comprehensive README for the training dataset"""
        readme_content = f"""# Robot Simulator Visual Training Dataset

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview

This dataset contains screenshots and detailed annotations from the Robot Simulator game, designed for training computer vision and robotics AI models.

### Dataset Statistics
- **Total Sessions**: {len(self.sessions)}
- **Total Actions**: {len(self.actionLog) if hasattr(self, 'actionLog') else 'N/A'}
- **Total States**: {len(self.stateLog) if hasattr(self, 'stateLog') else 'N/A'}
- **Resolution**: 800x600 pixels

### Data Structure

```
{output_dir}/
├── images/              # Referenced by screenshot paths in logs
├── annotations/         # JSON files with detailed annotations
├── metadata/           # Dataset metadata and class information
└── README.md          # This file
```

### Annotation Format

Each annotation file contains:

```json
{{
  "image_id": "session_xxx_action_yyy",
  "session_id": "session_xxx",
  "type": "action|state",
  "action_type": "move_forward|rotate_left|pickup_success|...",
  "robot_state": {{
    "position": {{"x": 100, "y": 200}},
    "angle": 1.57,
    "room": "Living Room",
    "hasObject": "red cup",
    "isMoving": true
  }},
  "object_positions": {{
    "all_objects": [...],
    "visible_objects": [...],
    "carried_object": {{...}}
  }},
  "environment": {{
    "rooms": [...],
    "furniture": [...],
    "doors": [...]
  }},
  "task": "Pick up the red cup and drop it in the Kitchen"
}}
```

### Use Cases

1. **Robot Navigation**: Train models to navigate through rooms and avoid obstacles
2. **Object Detection**: Detect and classify objects in the environment
3. **Task Planning**: Learn to complete multi-step tasks
4. **Spatial Reasoning**: Understand room layouts and spatial relationships
5. **Imitation Learning**: Learn from human gameplay demonstrations

### Coordinate System

- Origin (0,0) is at the top-left corner
- X-axis increases rightward
- Y-axis increases downward
- Canvas size: 800x600 pixels

### Classes

**Objects**: {self.get_object_classes()['object_types']}
**Colors**: {self.get_object_classes()['object_colors']}
**Rooms**: {self.get_room_classes()}
**Furniture**: {self.get_furniture_classes()}

### Loading the Dataset

```python
import json
from pathlib import Path

def load_annotations(annotation_dir):
    annotations = []
    for annotation_file in Path(annotation_dir).glob("*.json"):
        with open(annotation_file) as f:
            annotations.append(json.load(f))
    return annotations

# Load all annotations
annotations = load_annotations("{output_dir}/annotations")
```

### Citation

If you use this dataset in your research, please cite:

```
Robot Simulator Visual Training Dataset
Generated by Robot Simulator Training Data Collection System
Date: {datetime.now().strftime('%Y-%m-%d')}
```
"""
        
        readme_path = Path(output_dir) / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"📄 README created: {readme_path}")
        
        return str(readme_path)
    
    def generate_report(self, output_file="training_analysis_report.txt"):
        """Generate comprehensive analysis report"""
        print(f"\n📄 Generating comprehensive analysis report...")
        
        # Redirect print output to file
        import sys
        old_stdout = sys.stdout
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                sys.stdout = f
                
                print("🤖 ROBOT SIMULATOR TRAINING DATA ANALYSIS REPORT")
                print("=" * 60)
                print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Total Sessions Analyzed: {len(self.sessions)}")
                print()
                
                # Run all analyses and capture output
                self.analyze_gameplay_patterns()
                self.analyze_learning_progression()
                self.analyze_spatial_behavior()
                self.analyze_visual_data()
                self.analyze_slam_data()
        finally:
            sys.stdout = old_stdout
        
        print(f"✅ Report saved to: {output_file}")
        return output_file

def main():
    parser = argparse.ArgumentParser(description="Analyze Robot Simulator training data")
    parser.add_argument('--logs-dir', default='training_logs', help='Directory containing training logs')
    parser.add_argument('--export-ml', action='store_true', help='Export data for machine learning')
    parser.add_argument('--export-visual', action='store_true', help='Export visual training dataset')
    parser.add_argument('--export-slam', action='store_true', help='Export SLAM training dataset')
    parser.add_argument('--generate-report', action='store_true', help='Generate analysis report')
    parser.add_argument('--output', default='robot_ml_dataset.json', help='Output file for ML dataset')
    parser.add_argument('--visual-output', default='visual_training_data', help='Output directory for visual dataset')
    parser.add_argument('--slam-output', default='slam_training_data', help='Output directory for SLAM dataset')
    
    args = parser.parse_args()
    
    analyzer = TrainingDataAnalyzer(args.logs_dir)
    analyzer.load_all_sessions()
    
    if len(analyzer.sessions) == 0:
        print("❌ No training sessions found. Play the game to generate training data!")
        return
    
    # Run analysis
    analyzer.analyze_gameplay_patterns()
    analyzer.analyze_learning_progression()
    analyzer.analyze_spatial_behavior()
    analyzer.analyze_visual_data()
    analyzer.analyze_slam_data()
    
    # Export ML dataset if requested
    if args.export_ml:
        analyzer.export_ml_dataset(args.output)
    
    # Export visual dataset if requested
    if args.export_visual:
        output_dir = analyzer.export_visual_dataset(args.visual_output)
        analyzer.create_training_readme(args.visual_output)
        print(f"\n🎯 Visual training dataset ready for computer vision applications!")
        print(f"📁 Dataset location: {output_dir}")
    
    # Export SLAM dataset if requested
    if args.export_slam:
        slam_output_dir = analyzer.export_slam_dataset(args.slam_output)
        print(f"\n🗺️ SLAM training dataset ready for robotics applications!")
        print(f"📁 Dataset location: {slam_output_dir}")
    
    # Generate report if requested
    if args.generate_report:
        analyzer.generate_report()
    
    print(f"\n✅ Analysis complete! Sessions: {len(analyzer.sessions)}")
    
    # Summary of capabilities
    print(f"\n🚀 TRAINING DATA CAPABILITIES:")
    print(f"   📊 Behavioral Analysis: ✅ Action patterns, success rates, learning progression")
    print(f"   🗺️  Spatial Analysis: ✅ Room usage, position tracking, navigation patterns")
    print(f"   📸 Visual Analysis: ✅ Screenshots, object detection, environment mapping")
    print(f"   🤖 ML Ready: ✅ Structured datasets for reinforcement learning")
    print(f"   👁️  Computer Vision: ✅ Annotated images for object detection/tracking")
    print(f"   🗺️  SLAM Ready: ✅ Occupancy grids, landmarks, trajectories, range data")

if __name__ == "__main__":
    main() 