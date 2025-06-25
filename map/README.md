# ROS2 Map & Navigation Web Viewer

A modern web interface for visualizing ROS2 maps and navigation data through rosbridge websocket connection.

## Features

- 🗺️ **Real-time Map Visualization**: Display occupancy grid maps from ROS2
- 🤖 **Robot Pose Tracking**: Show current robot position and orientation
- 🛣️ **Path Planning**: Visualize navigation paths and trajectories
- 🔍 **Interactive Controls**: Zoom, pan, and navigate the map
- 📊 **Live Status Updates**: Real-time connection and robot status information
- 📝 **Connection Logging**: Monitor rosbridge connection and data flow

## Prerequisites

1. **ROS2 Installation**: Make sure you have ROS2 installed and running
2. **rosbridge_server**: Install and run rosbridge_server for websocket communication

### Install rosbridge_server

```bash
sudo apt update
sudo apt install ros-<your-ros-distro>-rosbridge-suite
```

### Start rosbridge_server

```bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml port:=9091
```

## Usage

### 1. Launch the Web Interface

Simply open `index.html` in a modern web browser:

```bash
# Using Python's built-in server (recommended)
python3 -m http.server 8000

# Then open http://localhost:8000 in your browser
```

Or use any other local web server:

```bash
# Using Node.js http-server
npx http-server -p 8000

# Using PHP
php -S localhost:8000
```

### 2. Connect to ROS2

1. **Set Connection URL**: The default URL `ws://localhost:9091` should work if rosbridge is running locally
2. **Click Connect**: Press the "Connect" button to establish the websocket connection
3. **Check Status**: The connection indicator should turn green when connected

### 3. Subscribe to Topics

After connecting successfully:

#### Map Subscription
- **Map Topic**: Default is `/map` (OccupancyGrid message type)
- **Click "Subscribe to Map"**: This will start receiving map data
- The map will appear on the canvas once data is received

#### Robot Pose Subscription
- **Pose Topic**: Default is `/amcl_pose` (PoseWithCovarianceStamped message type)
- **Click "Subscribe to Pose"**: Shows robot position and orientation
- Robot appears as an orange circle with direction indicator

#### Path Subscription
- **Path Topic**: Default is `/plan` (Path message type)
- **Click "Subscribe to Path"**: Displays planned navigation paths
- Paths are shown as blue lines on the map

### 4. Map Navigation

- **Zoom**: Use mouse wheel or zoom buttons (🔍+ / 🔍-)
- **Pan**: Click and drag on the map to move around
- **Reset View**: Click 🎯 to return to original view
- **Clear Path**: Click 🗑️ to remove displayed paths

## Common ROS2 Topics

Depending on your ROS2 setup, you might need to adjust the topic names:

### Map Topics
- `/map` - Static map from map_server
- `/global_costmap/costmap` - Global costmap from nav2
- `/local_costmap/costmap` - Local costmap from nav2

### Pose Topics
- `/amcl_pose` - AMCL localization
- `/pose` - Simple pose topic
- `/robot_pose` - Custom pose topic

### Path Topics
- `/plan` - Global path from nav2
- `/local_plan` - Local path from nav2
- `/path` - Custom path topic

## Troubleshooting

### Connection Issues

1. **Check rosbridge_server**: Make sure it's running on the correct port
   ```bash
   ros2 topic list
   ros2 service list | grep rosbridge
   ```

2. **Firewall**: Ensure websocket port (9091) is not blocked

3. **CORS Issues**: Use a local web server instead of opening HTML directly

### No Map Data

1. **Check map publisher**: Verify map is being published
   ```bash
   ros2 topic echo /map --once
   ros2 topic hz /map
   ```

2. **Topic Names**: Ensure the topic name matches exactly (case-sensitive)

3. **Message Types**: Verify the message type is correct:
   ```bash
   ros2 topic info /map
   ```

### Robot Not Visible

1. **Check pose data**: Verify pose is being published
   ```bash
   ros2 topic echo /amcl_pose --once
   ```

2. **Coordinate Frame**: Ensure pose is in the same frame as the map

## Development

### File Structure
```
├── index.html          # Main HTML interface
├── styles.css          # Modern CSS styling
├── script.js           # JavaScript application logic
└── README.md          # This documentation
```

### Customization

You can easily customize the interface by modifying:

- **Topic names**: Change default topics in the HTML inputs
- **Robot appearance**: Modify the `drawRobot()` function in `script.js`
- **Map colors**: Adjust the color mapping in `drawMap()` function
- **UI styling**: Update `styles.css` for different appearance

## Browser Compatibility

- ✅ Chrome/Chromium (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Edge

Note: Modern browser with WebSocket and Canvas support required.

## License

This project is open source. Feel free to modify and use for your ROS2 projects! 