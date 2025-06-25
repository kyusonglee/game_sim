# Unity3D Robot House Simulator Setup Guide

## 🎮 Project Overview
This Unity3D version provides:
- 3D robot navigation with realistic physics
- Multiple room environments with furniture
- Advanced AI training with ML-Agents
- Visual object detection and manipulation
- Procedural room generation
- Professional 3D graphics and lighting

## 📁 Project Structure
```
RobotHouseSimulator3D/
├── Assets/
│   ├── Scripts/
│   │   ├── Core/
│   │   │   ├── GameManager.cs
│   │   │   ├── RobotController.cs
│   │   │   └── TaskManager.cs
│   │   ├── Environment/
│   │   │   ├── RoomGenerator.cs
│   │   │   ├── FurnitureSpawner.cs
│   │   │   └── ObjectSpawner.cs
│   │   ├── AI/
│   │   │   ├── RobotAgent.cs
│   │   │   ├── MLAgentTrainer.cs
│   │   │   └── RewardSystem.cs
│   │   └── UI/
│   │       ├── GameUI.cs
│   │       └── TrainingUI.cs
│   ├── Prefabs/
│   │   ├── Robot/
│   │   ├── Rooms/
│   │   ├── Furniture/
│   │   └── Objects/
│   ├── Materials/
│   ├── Scenes/
│   │   ├── MainGame.unity
│   │   ├── Training.unity
│   │   └── TestEnvironment.unity
│   └── ML-Agents/
│       └── Config/
│           └── robot_trainer.yaml
├── Packages/
│   └── manifest.json
└── ProjectSettings/
```

## 🛠️ Setup Instructions

### 1. Create New Unity Project
- Unity Version: 2022.3 LTS or newer
- Template: 3D Core
- Project Name: RobotHouseSimulator3D

### 2. Install Required Packages
```
Window > Package Manager:
- ML-Agents (com.unity.ml-agents)
- ProBuilder (com.unity.probuilder)
- Cinemachine (com.unity.cinemachine)
- Input System (com.unity.inputsystem)
```

### 3. Import Scripts (provided below)

### 4. Setup Scenes
- Create main game scene with procedural rooms
- Setup training environment for AI
- Configure cameras and lighting

## 🚀 Getting Started
1. Open MainGame scene
2. Press Play to run human-controllable version
3. Open Training scene for AI training
4. Use ML-Agents commands for training

## 📈 AI Training Features
- ML-Agents integration for reinforcement learning
- Curriculum learning with increasing difficulty
- Multi-agent training support
- Real-time performance metrics
- Automated hyperparameter tuning

## 🎯 Training Commands
```bash
# Install ML-Agents Python package
pip install mlagents

# Start training
mlagents-learn Assets/ML-Agents/Config/robot_trainer.yaml --run-id=robot_training_01

# Resume training
mlagents-learn Assets/ML-Agents/Config/robot_trainer.yaml --run-id=robot_training_01 --resume

# View training progress
tensorboard --logdir results
``` 