class ROS2MapViewer {
    constructor() {
        this.ros = null;
        this.mapTopic = null;
        this.poseTopic = null;
        this.pathTopic = null;
        this.isConnected = false;
        
        // Canvas and rendering
        this.canvas = document.getElementById('map-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.mapData = null;
        this.robotPose = null;
        this.robotPath = [];
        
        // Robot control
        this.cmdVelTopic = null;
        this.currentTwist = { linear: { x: 0, y: 0, z: 0 }, angular: { x: 0, y: 0, z: 0 } };
        this.isControlling = false;
        
        // View transformation
        this.zoom = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        
        this.initializeEventListeners();
        this.setupCanvasInteraction();
        this.resizeCanvas();
        
        // Resize canvas on window resize
        window.addEventListener('resize', () => this.resizeCanvas());
    }
    
    initializeEventListeners() {
        // Connection controls
        document.getElementById('connect-btn').addEventListener('click', () => this.connect());
        document.getElementById('disconnect-btn').addEventListener('click', () => this.disconnect());
        
        // Subscription controls
        document.getElementById('list-topics-btn').addEventListener('click', () => this.listTopics());
        document.getElementById('subscribe-map-btn').addEventListener('click', () => this.subscribeToMap());
        document.getElementById('subscribe-pose-btn').addEventListener('click', () => this.subscribeToPose());
        document.getElementById('subscribe-path-btn').addEventListener('click', () => this.subscribeToPath());
        
        // Map controls
        document.getElementById('zoom-in-btn').addEventListener('click', () => this.zoomIn());
        document.getElementById('zoom-out-btn').addEventListener('click', () => this.zoomOut());
        document.getElementById('reset-view-btn').addEventListener('click', () => this.resetView());
        document.getElementById('clear-path-btn').addEventListener('click', () => this.clearPath());
        
        // Log controls
        document.getElementById('clear-log-btn').addEventListener('click', () => this.clearLog());
        
        // Robot control event listeners
        this.setupRobotControls();
        this.setupKeyboardControls();
        this.setupSpeedControls();
    }
    
    setupCanvasInteraction() {
        // Mouse wheel for zooming
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            this.zoom *= delta;
            this.zoom = Math.max(0.1, Math.min(10, this.zoom));
            this.drawMap();
        });
        
        // Mouse dragging for panning
        this.canvas.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            this.canvas.style.cursor = 'grabbing';
        });
        
        this.canvas.addEventListener('mousemove', (e) => {
            if (this.isDragging) {
                const deltaX = e.clientX - this.lastMouseX;
                const deltaY = e.clientY - this.lastMouseY;
                
                this.offsetX += deltaX / this.zoom;
                this.offsetY += deltaY / this.zoom;
                
                this.lastMouseX = e.clientX;
                this.lastMouseY = e.clientY;
                
                this.drawMap();
            }
        });
        
        this.canvas.addEventListener('mouseup', () => {
            this.isDragging = false;
            this.canvas.style.cursor = 'crosshair';
        });
        
        this.canvas.addEventListener('mouseleave', () => {
            this.isDragging = false;
            this.canvas.style.cursor = 'crosshair';
        });
    }
    
    resizeCanvas() {
        const container = this.canvas.parentElement;
        const containerRect = container.getBoundingClientRect();
        const controlsHeight = document.querySelector('.map-controls').offsetHeight;
        
        this.canvas.width = containerRect.width - 40; // 20px padding on each side
        this.canvas.height = Math.max(400, containerRect.height - controlsHeight - 60);
        
        this.drawMap();
    }
    
    connect() {
        const url = document.getElementById('ros-url').value;
        this.log(`Attempting to connect to ${url}`, 'info');
        this.updateConnectionStatus('connecting');
        
        this.ros = new ROSLIB.Ros({
            url: url
        });
        
        this.ros.on('connection', () => {
            this.isConnected = true;
            this.log('Successfully connected to ROS2 bridge', 'success');
            this.updateConnectionStatus('connected');
            this.enableSubscriptionButtons(true);
        });
        
        this.ros.on('error', (error) => {
            this.log(`Connection error: ${error}`, 'error');
            this.updateConnectionStatus('disconnected');
            this.enableSubscriptionButtons(false);
        });
        
        this.ros.on('close', () => {
            this.isConnected = false;
            this.log('Connection closed', 'warning');
            this.updateConnectionStatus('disconnected');
            this.enableSubscriptionButtons(false);
        });
    }
    
    disconnect() {
        if (this.ros) {
            this.ros.close();
            this.log('Disconnected from ROS2 bridge', 'info');
        }
    }
    
    updateConnectionStatus(status) {
        const indicator = document.getElementById('status-indicator');
        const text = document.getElementById('status-text');
        const connectBtn = document.getElementById('connect-btn');
        const disconnectBtn = document.getElementById('disconnect-btn');
        
        indicator.className = `status-indicator ${status}`;
        
        switch(status) {
            case 'connected':
                text.textContent = 'Connected';
                connectBtn.disabled = true;
                disconnectBtn.disabled = false;
                break;
            case 'connecting':
                text.textContent = 'Connecting...';
                connectBtn.disabled = true;
                disconnectBtn.disabled = true;
                break;
            case 'disconnected':
                text.textContent = 'Disconnected';
                connectBtn.disabled = false;
                disconnectBtn.disabled = true;
                break;
        }
    }
    
    enableSubscriptionButtons(enabled) {
        document.getElementById('list-topics-btn').disabled = !enabled;
        document.getElementById('subscribe-map-btn').disabled = !enabled;
        document.getElementById('subscribe-pose-btn').disabled = !enabled;
        document.getElementById('subscribe-path-btn').disabled = !enabled;
    }
    
    listTopics() {
        if (!this.isConnected) return;
        
        this.log('Requesting list of available topics...', 'info');
        
        const getTopicsService = new ROSLIB.Service({
            ros: this.ros,
            name: '/rosapi/topics',
            serviceType: 'rosapi/Topics'
        });
        
        const request = new ROSLIB.ServiceRequest({});
        
        getTopicsService.callService(request, (result) => {
            this.log(`Found ${result.topics.length} topics:`, 'success');
            result.topics.forEach(topic => {
                if (topic.includes('map') || topic.includes('pose') || topic.includes('plan') || topic.includes('path')) {
                    this.log(`  → ${topic}`, 'info');
                }
            });
            
            // Also get topic types
            const getTopicTypesService = new ROSLIB.Service({
                ros: this.ros,
                name: '/rosapi/topic_type',
                serviceType: 'rosapi/TopicType'
            });
            
            // Check specific map-related topics
            const mapTopics = result.topics.filter(topic => 
                topic.includes('map') || topic.includes('costmap')
            );
            
            mapTopics.forEach(topic => {
                const typeRequest = new ROSLIB.ServiceRequest({
                    topic: topic
                });
                
                getTopicTypesService.callService(typeRequest, (typeResult) => {
                    this.log(`  ${topic} → ${typeResult.type}`, 'info');
                });
            });
        }, (error) => {
            this.log(`Error getting topics: ${error}`, 'error');
        });
    }

    subscribeToMap() {
        if (!this.isConnected) return;
        
        const topicName = document.getElementById('map-topic').value;
        this.log(`Subscribing to map topic: ${topicName}`, 'info');
        
        if (this.mapTopic) {
            this.mapTopic.unsubscribe();
        }
        
        this.mapTopic = new ROSLIB.Topic({
            ros: this.ros,
            name: topicName,
            messageType: 'nav_msgs/OccupancyGrid'
        });
        
        this.mapTopic.subscribe((message) => {
            this.log('Received map data', 'success');
            this.processMapData(message);
        }, (error) => {
            this.log(`Map subscription error: ${error}`, 'error');
        });
    }
    
    subscribeToPose() {
        if (!this.isConnected) return;
        
        const topicName = document.getElementById('pose-topic').value;
        this.log(`Subscribing to pose topic: ${topicName}`, 'info');
        
        if (this.poseTopic) {
            this.poseTopic.unsubscribe();
        }
        
        this.poseTopic = new ROSLIB.Topic({
            ros: this.ros,
            name: topicName,
            messageType: 'geometry_msgs/PoseWithCovarianceStamped'
        });
        
        this.poseTopic.subscribe((message) => {
            this.processPoseData(message);
        });
    }
    
    subscribeToPath() {
        if (!this.isConnected) return;
        
        const topicName = document.getElementById('path-topic').value;
        this.log(`Subscribing to path topic: ${topicName}`, 'info');
        
        if (this.pathTopic) {
            this.pathTopic.unsubscribe();
        }
        
        this.pathTopic = new ROSLIB.Topic({
            ros: this.ros,
            name: topicName,
            messageType: 'nav_msgs/Path'
        });
        
        this.pathTopic.subscribe((message) => {
            this.processPathData(message);
        });
    }
    
    processMapData(mapMessage) {
        this.log(`Processing map data: ${mapMessage.info.width}x${mapMessage.info.height}, resolution: ${mapMessage.info.resolution}`, 'info');
        
        if (!mapMessage.info || !mapMessage.data) {
            this.log('Invalid map data received - missing info or data', 'error');
            return;
        }
        
        if (mapMessage.data.length === 0) {
            this.log('Map data is empty', 'warning');
            return;
        }
        
        this.mapData = {
            info: mapMessage.info,
            data: mapMessage.data
        };
        
        // Update map info display
        document.getElementById('resolution').textContent = mapMessage.info.resolution.toFixed(3) + ' m/px';
        document.getElementById('width').textContent = mapMessage.info.width + ' px';
        document.getElementById('height').textContent = mapMessage.info.height + ' px';
        document.getElementById('origin').textContent = 
            `(${mapMessage.info.origin.position.x.toFixed(2)}, ${mapMessage.info.origin.position.y.toFixed(2)})`;
        
        // Hide the overlay
        document.getElementById('map-overlay').style.display = 'none';
        
        this.log(`Map loaded successfully: ${mapMessage.data.length} data points`, 'success');
        this.drawMap();
    }
    
    processPoseData(poseMessage) {
        this.robotPose = {
            position: poseMessage.pose.pose.position,
            orientation: poseMessage.pose.pose.orientation
        };
        
        // Update robot status display
        const pos = this.robotPose.position;
        document.getElementById('robot-position').textContent = 
            `(${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)})`;
        
        // Convert quaternion to euler angle (yaw)
        const q = this.robotPose.orientation;
        const yaw = Math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
        document.getElementById('robot-orientation').textContent = 
            `${(yaw * 180 / Math.PI).toFixed(1)}°`;
        
        this.drawMap();
    }
    
    processPathData(pathMessage) {
        this.robotPath = pathMessage.poses.map(poseStamped => ({
            x: poseStamped.pose.position.x,
            y: poseStamped.pose.position.y
        }));
        
        this.log(`Received path with ${this.robotPath.length} waypoints`, 'info');
        this.drawMap();
    }
    
    drawMap() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        if (!this.mapData) return;
        
        const mapInfo = this.mapData.info;
        const mapWidth = mapInfo.width;
        const mapHeight = mapInfo.height;
        
        // Create ImageData for the map
        const imageData = this.ctx.createImageData(mapWidth, mapHeight);
        const data = imageData.data;
        
        // Convert occupancy grid to image data
        for (let i = 0; i < this.mapData.data.length; i++) {
            const value = this.mapData.data[i];
            let color;
            
            if (value === -1) {
                // Unknown area - gray
                color = [128, 128, 128, 255];
            } else if (value === 0) {
                // Free space - white
                color = [255, 255, 255, 255];
            } else {
                // Occupied space - black, with intensity based on probability
                const intensity = Math.floor((1 - value / 100) * 255);
                color = [intensity, intensity, intensity, 255];
            }
            
            const pixelIndex = i * 4;
            data[pixelIndex] = color[0];     // R
            data[pixelIndex + 1] = color[1]; // G
            data[pixelIndex + 2] = color[2]; // B
            data[pixelIndex + 3] = color[3]; // A
        }
        
        // Save context state
        this.ctx.save();
        
        // Apply transformations
        this.ctx.translate(this.canvas.width / 2, this.canvas.height / 2);
        this.ctx.scale(this.zoom, this.zoom);
        this.ctx.translate(this.offsetX, this.offsetY);
        
        // Draw the map (flip Y axis to match ROS coordinate system)
        this.ctx.scale(1, -1);
        this.ctx.putImageData(imageData, -mapWidth / 2, -mapHeight / 2);
        
        // Draw robot path
        if (this.robotPath.length > 1) {
            this.ctx.scale(1, -1); // Flip back for path drawing
            this.drawPath();
            this.ctx.scale(1, -1); // Flip again for robot drawing
        }
        
        // Draw robot pose
        if (this.robotPose) {
            this.ctx.scale(1, -1); // Flip back for robot drawing
            this.drawRobot();
        }
        
        // Restore context state
        this.ctx.restore();
    }
    
    drawPath() {
        if (this.robotPath.length < 2) return;
        
        const resolution = this.mapData.info.resolution;
        const origin = this.mapData.info.origin.position;
        
        this.ctx.strokeStyle = '#2196F3';
        this.ctx.lineWidth = 2 / this.zoom;
        this.ctx.beginPath();
        
        for (let i = 0; i < this.robotPath.length; i++) {
            const point = this.robotPath[i];
            const pixelX = (point.x - origin.x) / resolution - this.mapData.info.width / 2;
            const pixelY = (point.y - origin.y) / resolution - this.mapData.info.height / 2;
            
            if (i === 0) {
                this.ctx.moveTo(pixelX, pixelY);
            } else {
                this.ctx.lineTo(pixelX, pixelY);
            }
        }
        
        this.ctx.stroke();
    }
    
    drawRobot() {
        const resolution = this.mapData.info.resolution;
        const origin = this.mapData.info.origin.position;
        
        const robotX = (this.robotPose.position.x - origin.x) / resolution - this.mapData.info.width / 2;
        const robotY = (this.robotPose.position.y - origin.y) / resolution - this.mapData.info.height / 2;
        
        // Convert quaternion to yaw angle
        const q = this.robotPose.orientation;
        const yaw = Math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
        
        this.ctx.save();
        this.ctx.translate(robotX, robotY);
        this.ctx.rotate(-yaw); // Negative because of flipped Y axis
        
        // Draw robot body (circle)
        this.ctx.fillStyle = '#FF5722';
        this.ctx.beginPath();
        this.ctx.arc(0, 0, 8 / this.zoom, 0, 2 * Math.PI);
        this.ctx.fill();
        
        // Draw robot direction indicator (triangle)
        this.ctx.fillStyle = '#FFC107';
        this.ctx.beginPath();
        this.ctx.moveTo(12 / this.zoom, 0);
        this.ctx.lineTo(-6 / this.zoom, 6 / this.zoom);
        this.ctx.lineTo(-6 / this.zoom, -6 / this.zoom);
        this.ctx.closePath();
        this.ctx.fill();
        
        this.ctx.restore();
    }
    
    zoomIn() {
        this.zoom *= 1.2;
        this.zoom = Math.min(10, this.zoom);
        this.drawMap();
    }
    
    zoomOut() {
        this.zoom *= 0.8;
        this.zoom = Math.max(0.1, this.zoom);
        this.drawMap();
    }
    
    resetView() {
        this.zoom = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this.drawMap();
    }
    
    clearPath() {
        this.robotPath = [];
        this.log('Cleared robot path', 'info');
        this.drawMap();
    }
    
    log(message, type = 'info') {
        const logOutput = document.getElementById('log-output');
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${type}`;
        logEntry.textContent = `[${timestamp}] ${message}`;
        
        logOutput.appendChild(logEntry);
        logOutput.scrollTop = logOutput.scrollHeight;
        
        // Keep only last 100 log entries
        const entries = logOutput.children;
        while (entries.length > 100) {
            logOutput.removeChild(entries[0]);
        }
    }
    
    clearLog() {
        document.getElementById('log-output').innerHTML = '';
    }
    
    setupRobotControls() {
        // Control button event listeners
        document.getElementById('forward-btn').addEventListener('mousedown', () => this.startMovement('forward'));
        document.getElementById('backward-btn').addEventListener('mousedown', () => this.startMovement('backward'));
        document.getElementById('left-btn').addEventListener('mousedown', () => this.startMovement('left'));
        document.getElementById('right-btn').addEventListener('mousedown', () => this.startMovement('right'));
        document.getElementById('stop-btn').addEventListener('click', () => this.stopMovement());
        
        // Stop movement when mouse is released
        ['forward-btn', 'backward-btn', 'left-btn', 'right-btn'].forEach(id => {
            const btn = document.getElementById(id);
            btn.addEventListener('mouseup', () => this.stopMovement());
            btn.addEventListener('mouseleave', () => this.stopMovement());
        });
        
        // Touch events for mobile
        ['forward-btn', 'backward-btn', 'left-btn', 'right-btn'].forEach(id => {
            const btn = document.getElementById(id);
            btn.addEventListener('touchstart', (e) => {
                e.preventDefault();
                this.startMovement(id.replace('-btn', ''));
            });
            btn.addEventListener('touchend', (e) => {
                e.preventDefault();
                this.stopMovement();
            });
        });
    }
    
    setupKeyboardControls() {
        document.addEventListener('keydown', (e) => {
            if (this.isControlling) return; // Prevent repeated keydown events
            
            switch(e.key.toLowerCase()) {
                case 'w':
                case 'arrowup':
                    e.preventDefault();
                    this.startMovement('forward');
                    break;
                case 's':
                case 'arrowdown':
                    e.preventDefault();
                    this.startMovement('backward');
                    break;
                case 'a':
                case 'arrowleft':
                    e.preventDefault();
                    this.startMovement('left');
                    break;
                case 'd':
                case 'arrowright':
                    e.preventDefault();
                    this.startMovement('right');
                    break;
                case ' ':
                    e.preventDefault();
                    this.stopMovement();
                    break;
            }
        });
        
        document.addEventListener('keyup', (e) => {
            switch(e.key.toLowerCase()) {
                case 'w':
                case 'arrowup':
                case 's':
                case 'arrowdown':
                case 'a':
                case 'arrowleft':
                case 'd':
                case 'arrowright':
                    e.preventDefault();
                    this.stopMovement();
                    break;
            }
        });
    }
    
    setupSpeedControls() {
        const linearSpeed = document.getElementById('linear-speed');
        const angularSpeed = document.getElementById('angular-speed');
        const linearValue = document.getElementById('linear-speed-value');
        const angularValue = document.getElementById('angular-speed-value');
        
        linearSpeed.addEventListener('input', (e) => {
            linearValue.textContent = `${e.target.value} m/s`;
        });
        
        angularSpeed.addEventListener('input', (e) => {
            angularValue.textContent = `${e.target.value} rad/s`;
        });
    }
    
    startMovement(direction) {
        if (!this.isConnected) {
            this.log('Connect to ROS2 first to control the robot', 'warning');
            return;
        }
        
        this.isControlling = true;
        
        // Initialize cmd_vel topic if not already done
        if (!this.cmdVelTopic) {
            const topicName = document.getElementById('cmd-vel-topic').value;
            this.cmdVelTopic = new ROSLIB.Topic({
                ros: this.ros,
                name: topicName,
                messageType: 'geometry_msgs/Twist'
            });
            this.log(`Robot control topic set to: ${topicName}`, 'info');
        }
        
        const linearSpeed = parseFloat(document.getElementById('linear-speed').value);
        const angularSpeed = parseFloat(document.getElementById('angular-speed').value);
        
        // Reset twist
        this.currentTwist = { linear: { x: 0, y: 0, z: 0 }, angular: { x: 0, y: 0, z: 0 } };
        
        // Set movement based on direction
        switch(direction) {
            case 'forward':
                this.currentTwist.linear.x = linearSpeed;
                document.getElementById('forward-btn').classList.add('pressed');
                break;
            case 'backward':
                this.currentTwist.linear.x = -linearSpeed;
                document.getElementById('backward-btn').classList.add('pressed');
                break;
            case 'left':
                this.currentTwist.angular.z = angularSpeed;
                document.getElementById('left-btn').classList.add('pressed');
                break;
            case 'right':
                this.currentTwist.angular.z = -angularSpeed;
                document.getElementById('right-btn').classList.add('pressed');
                break;
        }
        
        this.publishTwist();
    }
    
    stopMovement() {
        if (!this.isConnected || !this.cmdVelTopic) return;
        
        this.isControlling = false;
        
        // Remove pressed state from all buttons
        ['forward-btn', 'backward-btn', 'left-btn', 'right-btn'].forEach(id => {
            document.getElementById(id).classList.remove('pressed');
        });
        
        // Send stop command
        this.currentTwist = { linear: { x: 0, y: 0, z: 0 }, angular: { x: 0, y: 0, z: 0 } };
        this.publishTwist();
    }
    
    publishTwist() {
        if (!this.cmdVelTopic) return;
        
        const twist = new ROSLIB.Message({
            linear: {
                x: this.currentTwist.linear.x,
                y: this.currentTwist.linear.y,
                z: this.currentTwist.linear.z
            },
            angular: {
                x: this.currentTwist.angular.x,
                y: this.currentTwist.angular.y,
                z: this.currentTwist.angular.z
            }
        });
        
        this.cmdVelTopic.publish(twist);
        
        if (this.currentTwist.linear.x !== 0 || this.currentTwist.angular.z !== 0) {
            this.log(`Sent: linear=${this.currentTwist.linear.x.toFixed(2)}, angular=${this.currentTwist.angular.z.toFixed(2)}`, 'info');
        }
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const viewer = new ROS2MapViewer();
    viewer.log('ROS2 Map Viewer initialized', 'success');
    viewer.log('Use WASD keys or buttons to control the robot', 'info');
}); 