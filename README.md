# 🤖 Robot Simulator - Web Edition with Enhanced Training Data Collection

A comprehensive web-based robot simulation game designed for collecting rich training data for AI and robotics research. Features realistic robot movement, task-based gameplay, and extensive data logging including **SLAM (Simultaneous Localization and Mapping)** capabilities.

## 🚀 Features

### Game Features
- **Realistic Robot Controls**: Forward/backward movement and rotation (like real robots)
- **Multi-Room Navigation**: Living room, bedroom, kitchen, bathroom with doors
- **Object Interaction**: Pick up and drop objects with physics
- **Progressive Difficulty**: Multiple levels with increasing complexity
- **Task-Based Gameplay**: Room delivery tasks and object manipulation

### Training Data Collection
- **📸 Visual Data**: Screenshots captured for every action
- **🎯 Object Tracking**: Detailed position and state data for all objects
- **🗺️ SLAM Data**: Complete mapping and localization information
- **🤖 Behavioral Logs**: All user actions, collisions, and decisions
- **📊 Performance Metrics**: Success rates, completion times, efficiency scores

### SLAM (Simultaneous Localization and Mapping) Features
- **🗺️ Occupancy Grids**: 2D grid maps showing occupied/free/unknown space
- **🎯 Landmark Detection**: Furniture corners, doors, and room features
- **📡 Range Sensor Simulation**: 16-ray lidar-like distance measurements
- **🛤️ Trajectory Tracking**: Path history with efficiency analysis
- **🕸️ Topological Mapping**: High-level room connectivity graphs
- **📍 Robot Pose**: Continuous position and orientation tracking

## 🎮 How to Play

### Controls
- **↑**: Move Forward
- **↓**: Move Backward  
- **←**: Rotate Left
- **→**: Rotate Right
- **P**: Pick Up Object
- **D**: Drop Object
- **R**: Restart Level
- **E**: Export Training Data

### Gameplay
1. Complete tasks by picking up objects and moving them to target rooms
2. Navigate through doors to access different rooms
3. Avoid collisions with furniture and walls
4. Complete tasks efficiently to earn higher scores

## 🛠️ Setup and Installation

### Quick Start (Python)
```bash
# Clone or download the project
cd robot-simulator

# Start the server (includes all dependencies)
python server.py

# Open your browser to http://localhost:8000
# Start playing and generating training data!
```

### Docker Setup
```bash
# Setup directories and permissions
./docker-setup.sh

# Build and run with persistent data
docker-compose up --build

# View logs
docker-compose logs -f robot-simulator

# Stop the container
docker-compose down
```

### Manual Setup
```bash
# Install Python dependencies for analysis
pip install pandas numpy matplotlib seaborn

# Start the web server
python server.py

# In another terminal, analyze collected data
python analyze_training_data.py --export-slam --export-visual
```

## 📊 Data Analysis and Export

### Visual Training Data
```bash
# Export computer vision dataset
python analyze_training_data.py --export-visual

# Output: visual_training_data/
#   ├── annotations/     # JSON files with object positions, robot state
#   ├── metadata/        # Dataset information and class definitions
#   └── README.md        # Dataset documentation
```

### SLAM Training Data
```bash
# Export SLAM dataset for robotics research
python analyze_training_data.py --export-slam

# Output: slam_training_data/
#   ├── occupancy_grids/   # 2D grid maps
#   ├── trajectories/      # Robot path data
#   ├── landmarks/         # Detected features
#   ├── range_data/        # Simulated lidar readings
#   ├── topological_maps/  # Room connectivity graphs
#   └── slam_dataset_info.json
```

### Machine Learning Dataset
```bash
# Export structured ML dataset
python analyze_training_data.py --export-ml

# Output: robot_ml_dataset.json
# - Action sequences for reinforcement learning
# - State-action pairs for imitation learning
# - Behavioral patterns for analysis
```

### Comprehensive Analysis
```bash
# Generate complete analysis report
python analyze_training_data.py --generate-report

# Output: training_analysis_report.txt
# - Gameplay pattern analysis
# - Learning progression metrics
# - Spatial behavior insights
# - Visual data statistics
# - SLAM performance analysis
```

## 🗺️ SLAM Data Structure

### Robot Pose
```json
{
  "robot_pose": {
    "x": 250.5,
    "y": 180.3,
    "theta": 1.57,
    "confidence": 1.0
  }
}
```

### Occupancy Grid
```json
{
  "occupancy_grid": {
    "width": 40,
    "height": 30,
    "resolution": 20,
    "data": [100, 0, -1, ...],  // 100=occupied, 0=free, -1=unknown
    "grid_2d": [[100, 0], [0, -1], ...]
  }
}
```

### Landmarks
```json
{
  "landmarks": [
    {
      "id": 0,
      "type": "furniture_corner",
      "position": {"x": 120, "y": 85},
      "distance": 45.2,
      "bearing": 0.785,
      "confidence": 0.85,
      "associated_object": "sofa"
    }
  ]
}
```

### Range Data (Simulated Lidar)
```json
{
  "range_data": {
    "ranges": [
      {"angle": 0.0, "range": 87.3, "hit_type": "obstacle"},
      {"angle": 0.393, "range": 150.0, "hit_type": "max_range"}
    ],
    "angle_min": 0,
    "angle_max": 6.283,
    "angle_increment": 0.393,
    "range_min": 10,
    "range_max": 150
  }
}
```

### Trajectory
```json
{
  "trajectory": {
    "path": [
      {
        "timestamp": 1750118496430,
        "position": {"x": 250, "y": 180},
        "angle": 1.57,
        "room": "Living Room"
      }
    ],
    "total_distance": 234.7,
    "path_efficiency": 0.85
  }
}
```

### Topological Map
```json
{
  "topological_map": {
    "nodes": [
      {
        "id": 0,
        "type": "room",
        "name": "Living Room",
        "center": {"x": 200, "y": 150},
        "contains_robot": true
      }
    ],
    "edges": [
      {
        "from": 0,
        "to": 1,
        "door_id": 0,
        "door_position": {"x": 300, "y": 150}
      }
    ],
    "current_node": 0
  }
}
```

## 📁 Data Storage Structure

```
simulator/
├── training_logs/
│   └── session_[timestamp_id]/
│       ├── actions.jsonl              # Action logs with SLAM data
│       ├── states.jsonl               # State logs with SLAM data
│       └── screenshots/               # PNG images for each action
│           ├── action_[timestamp].png
│           └── state_[timestamp].png
├── visual_training_data/              # Computer vision dataset
├── slam_training_data/                # SLAM research dataset
└── robot_ml_dataset.json            # Machine learning dataset
```

## 🔬 Research Applications

### Computer Vision
- **Object Detection**: Train models to detect cups, books, phones, etc.
- **Scene Understanding**: Learn room layouts and spatial relationships
- **Visual Navigation**: Path planning using visual features

### Robotics & SLAM
- **Mapping**: Train occupancy grid generation algorithms
- **Localization**: Learn robot pose estimation from sensor data
- **Path Planning**: Optimize navigation strategies using trajectory data
- **Landmark Detection**: Train feature detection for SLAM systems

### Machine Learning
- **Reinforcement Learning**: Action-reward sequences for policy optimization
- **Imitation Learning**: Human demonstration data for behavior cloning
- **Multi-Task Learning**: Combined navigation, manipulation, and planning

## 🔧 Technical Details

### Server Features
- **Port Auto-Detection**: Automatically finds available ports
- **CORS Support**: Cross-origin requests for development
- **Data Persistence**: Automatic saving of all training data
- **RESTful API**: Standard endpoints for data export

### Performance Optimization
- **Efficient Logging**: Screenshots saved separately to reduce memory usage
- **Batch Processing**: State data sent in batches for performance
- **Selective Screenshots**: Action screenshots always captured, state screenshots every 10th entry

### Docker Integration
- **Volume Mounts**: Training data persisted to host filesystem
- **Multi-platform**: Works on Linux, macOS, and Windows
- **Easy Deployment**: Single command setup with docker-compose

## 🤝 Contributing

This project is designed for research and educational purposes. Contributions welcome for:
- Additional SLAM algorithms
- Enhanced sensor simulations
- New task types
- Analysis improvements
- Performance optimizations

## 📜 License

Open source project for research and educational use. See individual files for specific licensing information.

---

**Ready to collect comprehensive robotics training data? Start the simulator and begin generating SLAM datasets for your research!** 🚀🤖📊 