// Farm Robot Simulator - Agricultural Management Game

class Game {
    constructor() {
        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.canvas.width = 1000;
        this.canvas.height = 700;
        
        // Check if canvas is properly initialized
        if (!this.canvas || !this.ctx) {
            console.error('Failed to initialize canvas or context!');
            return;
        }
        
        console.log('Canvas initialized:', this.canvas.width, 'x', this.canvas.height);
        
        // Game state
        this.level = 1;
        this.score = 0;
        this.startTime = Date.now();
        this.gameRunning = true;
        this.dayCount = 1;
        this.gameTime = 0; // Time in minutes (simulated)
        this.timeSpeed = 10; // 1 real second = 10 game minutes
        
        // Robot battery system (lasts 3-5 minutes of real time for more challenge)
        this.batteryLevel = 100; // Percentage
        this.batteryDrainRate = 0.6; // Per second when moving (100% / 2.8 minutes = 0.6/s)
        this.batteryIdleDrain = 0.15; // Per second when idle (faster idle drain too)
        this.isCharging = false;
        this.chargingRate = 10; // Per second when charging (slightly faster charging to compensate)
        
        // Game objects and areas
        this.farmHouse = null;
        this.chargingStation = null;
        this.plants = [];
        this.animals = [];
        this.trash = [];
        this.weeds = [];
        this.grassAreas = [];
        this.bins = []; // Only one black trash bin
        this.robot = null;
        
        // Game systems
        this.plantGrowthSystem = new PlantGrowthSystem();
        this.animalSpawnSystem = new AnimalSpawnSystem();
        this.weatherSystem = new WeatherSystem();
        
        // Task management
        this.activeTasks = [];
        this.completedTasks = [];
        this.taskQueue = [];
        
        // Input handling
        this.keys = {};
        this.setupEventListeners();
        
        // Point navigation system (kept from original)
        this.navigationTarget = null;
        this.navigationPath = [];
        this.currentPathIndex = 0;
        this.isNavigating = false;
        this.pathfindingGrid = null;
        this.gridSize = 15;
        
        // Logging system for training data
        this.gameSession = {
            sessionId: this.generateSessionId(),
            startTime: Date.now(),
            endTime: null,
            level: this.level,
            completed: false,
            finalScore: 0
        };
        this.actionLog = [];
        this.stateLog = [];
        this.lastLogTime = Date.now();
        this.logInterval = 200; // Log state every 200ms
        
        // Photo system for plant monitoring
        this.photosTaken = [];
        this.lastPhotoTime = 0;
        this.photoInterval = 1440; // 24 hours in game minutes
        
        // Watering system
        this.waterLevel = 100; // Robot's water tank level
        this.maxWaterLevel = 100;
        this.waterUsagePerAction = 15; // Water used per watering action
        this.waterRefillRate = 50; // Water gained per second at charging station
        
        // Vision and detection system for animals
        this.animalDetectionRange = 120; // Distance robot can see animals
        this.visionRange = 100; // Distance robot can see trash/weeds
        this.visionAngle = Math.PI / 3; // 60 degrees field of view
        this.detectedObjects = new Set(); // Track what robot has visually detected
        this.detectedAnimals = new Set(); // Track detected animals
        
        // Timing and scoring system
        this.animalSpawnTimes = new Map(); // Track when animals appear
        this.lastPointMessage = '';
        this.pointMessageTime = 0;
        
        // Layout variation system
        this.layoutType = null;
        
        // Colors and constants
        this.COLORS = {
            grass: '#4CAF50', darkGrass: '#2E7D32', farmHouse: '#8D6E63', 
            concrete: '#9E9E9E', dirt: '#8D6E63',
            plant: '#4CAF50', flower: '#E91E63', tree: '#2E7D32',
            robot: '#2196F3', trash: '#424242',
            weed: '#689F38',
            animal: '#8D6E63', charging: '#FFC107', battery: '#4CAF50',
            sky: '#87CEEB', cloud: '#FFFFFF'
        };
        
        console.log('Farm robot game initialized, starting level...');
        
        // Initialize game
        this.initializeLevel();
        this.gameLoop();
        
        // Debug: Check initialization status
        setTimeout(() => {
            console.log('=== FARM GAME DEBUG INFO ===');
            console.log('Farm House:', this.farmHouse ? 'Created' : 'Missing');
            console.log('Plants:', this.plants?.length || 0);
            console.log('Robot:', this.robot ? 'Created' : 'Missing');
            console.log('Battery:', this.batteryLevel + '%');
            console.log('===========================');
        }, 1000);
    }
    
    setupEventListeners() {
        // Keyboard input
        document.addEventListener('keydown', (e) => {
            this.keys[e.code] = true;
            this.handleKeyPress(e.code);
        });
        
        document.addEventListener('keyup', (e) => {
            this.keys[e.code] = false;
        });
        
        // Mouse click navigation
        this.canvas.addEventListener('click', (e) => {
            this.handleCanvasClick(e);
        });
        
        // Modal buttons
        document.getElementById('continueBtn').addEventListener('click', () => {
            this.nextLevel();
        });
        
        document.getElementById('retryBtn').addEventListener('click', () => {
            this.restartLevel();
        });
        
        document.getElementById('restartBtn').addEventListener('click', () => {
            this.restartGame();
        });
    }
    
    handleKeyPress(code) {
        switch(code) {
            case 'KeyP':
                this.performPickupAction();
                break;
            case 'KeyD':
                this.performDropAction();
                break;
            case 'KeyC':
                // Take photo of plants
                this.takePhoto();
                break;
            case 'KeyM':
                // Mow lawn (if near grass and grass is long)
                this.mowLawn();
                break;
            case 'KeyW':
                // Warn animals
                this.warnAnimals();
                break;
            case 'KeyT':
                // Water plants/lawn
                this.waterPlants();
                break;
            case 'KeyR':
                this.logAction('restart_level');
                this.restartLevel();
                break;
            case 'KeyE':
                this.exportLogs();
                break;
            case 'KeyG':
                // Go to charging station
                this.goToChargingStation();
                break;
            case 'KeyF':
                // Go to water station
                this.goToWaterStation();
                break;
            case 'Escape':
                if (this.isNavigating) {
                    this.cancelNavigation();
                }
                break;
        }
    }
    
    initializeLevel() {
        this.logAction('level_start', { 
            level: this.level,
            dayCount: this.dayCount 
        });
        
        this.generateEnvironment();
        this.generateRobot();
        this.generateInitialObjects();
        this.startTime = Date.now();
        this.updateUI();
        
        this.logAction('level_initialized', {
            plantCount: this.plants.length,
            robotStartPosition: { x: this.robot.pos.x, y: this.robot.pos.y },
            batteryLevel: this.batteryLevel
        });
    }
    
    generateEnvironment() {
        // Generate random layout
        this.layoutType = this.generateRandomLayout();
        
        // Generate grass areas
        this.generateGrassAreas();
        
        // Generate plants around the yard
        this.generatePlants();
        
        // Generate pathways
        this.generatePathways();
    }
    
    generateRandomLayout() {
        const layouts = ['compact', 'spread', 'corner', 'central'];
        const selectedLayout = layouts[Math.floor(Math.random() * layouts.length)];
        
        switch(selectedLayout) {
            case 'compact':
                this.generateCompactLayout();
                break;
            case 'spread':
                this.generateSpreadLayout();
                break;
            case 'corner':
                this.generateCornerLayout();
                break;
            case 'central':
                this.generateCentralLayout();
                break;
        }
        
        return selectedLayout;
    }
    
    generateCompactLayout() {
        // House in top-left, everything clustered
        this.farmHouse = {
            rect: { x: 200, y: 50, width: 350, height: 180 },
            color: this.COLORS.farmHouse
        };
        
        this.chargingStation = {
            rect: { x: 600, y: 120, width: 50, height: 50 },
            color: this.COLORS.charging,
            isActive: false
        };
        
        this.waterStation = {
            rect: { x: 660, y: 120, width: 50, height: 50 },
            color: '#2196F3', // Blue for water
            isActive: false
        };
        
        this.bins = [
            { type: 'trash', rect: { x: 50, y: 300, width: 40, height: 50 }, color: this.COLORS.trash }
        ];
    }
    
    generateSpreadLayout() {
        // House in center, everything spread out
        this.farmHouse = {
            rect: { x: 350, y: 100, width: 300, height: 150 },
            color: this.COLORS.farmHouse
        };
        
        this.chargingStation = {
            rect: { x: 50, y: 50, width: 50, height: 50 },
            color: this.COLORS.charging,
            isActive: false
        };
        
        this.waterStation = {
            rect: { x: 110, y: 50, width: 50, height: 50 },
            color: '#2196F3', // Blue for water
            isActive: false
        };
        
        this.bins = [
            { type: 'trash', rect: { x: 850, y: 200, width: 40, height: 50 }, color: this.COLORS.trash }
        ];
    }
    
    generateCornerLayout() {
        // House in corner, diagonal arrangement
        this.farmHouse = {
            rect: { x: 100, y: 80, width: 280, height: 160 },
            color: this.COLORS.farmHouse
        };
        
        this.chargingStation = {
            rect: { x: 800, y: 600, width: 50, height: 50 },
            color: this.COLORS.charging,
            isActive: false
        };
        
        this.waterStation = {
            rect: { x: 860, y: 600, width: 50, height: 50 },
            color: '#2196F3', // Blue for water
            isActive: false
        };
        
        this.bins = [
            { type: 'trash', rect: { x: 700, y: 100, width: 40, height: 50 }, color: this.COLORS.trash }
        ];
    }
    
    generateCentralLayout() {
        // House in center, everything around it
        this.farmHouse = {
            rect: { x: 400, y: 200, width: 200, height: 180 },
            color: this.COLORS.farmHouse
        };
        
        this.chargingStation = {
            rect: { x: 200, y: 200, width: 50, height: 50 },
            color: this.COLORS.charging,
            isActive: false
        };
        
        this.waterStation = {
            rect: { x: 260, y: 200, width: 50, height: 50 },
            color: '#2196F3', // Blue for water
            isActive: false
        };
        
        this.bins = [
            { type: 'trash', rect: { x: 650, y: 200, width: 40, height: 50 }, color: this.COLORS.trash }
        ];
    }
    
    generateGrassAreas() {
        this.grassAreas = [
            // Main lawn area
            { 
                rect: { x: 200, y: 300, width: 600, height: 300 },
                grassHeight: Math.random() * 5 + 2, // 2-7 units
                growthRate: 0.1, // units per day
                needsMowing: false,
                waterLevel: 60 + Math.random() * 40, // 60-100% water
                needsWatering: false
            },
            // Side lawn
            { 
                rect: { x: 50, y: 500, width: 150, height: 150 },
                grassHeight: Math.random() * 4 + 2,
                growthRate: 0.08,
                needsMowing: false,
                waterLevel: 60 + Math.random() * 40, // 60-100% water
                needsWatering: false
            }
        ];
        
        // Check if grass needs mowing and watering
        this.grassAreas.forEach(area => {
            area.needsMowing = area.grassHeight > 6;
            area.needsWatering = area.waterLevel < 40;
        });
    }
    
    generatePlants() {
        this.plants = [];
        
        // Flower beds near house (but not inside)
        for (let i = 0; i < 6; i++) {
            let pos;
            let attempts = 0;
            do {
                pos = { 
                    x: 250 + i * 80 + Math.random() * 20, 
                    y: 280 + Math.random() * 20 
                };
                attempts++;
            } while (this.isPositionInHouse(pos) && attempts < 10);
            
            this.plants.push({
                type: 'flower',
                pos: pos,
                growth: Math.random() * 50 + 30,
                age: Math.random() * 30,
                lastPhotoTime: 0,
                waterLevel: 50 + Math.random() * 50, // 50-100% water
                needsWater: false, // Will be calculated based on waterLevel
                hasWeeds: Math.random() < 0.2
            });
        }
        
        // Trees around perimeter (avoid house)
        for (let i = 0; i < 4; i++) {
            let pos;
            let attempts = 0;
            do {
                pos = {
                    x: 100 + i * 200 + Math.random() * 50,
                    y: 600 + Math.random() * 50
                };
                attempts++;
            } while (this.isPositionInHouse(pos) && attempts < 10);
            
            this.plants.push({
                type: 'tree',
                pos: pos,
                growth: Math.random() * 30 + 60,
                age: Math.random() * 100 + 50,
                lastPhotoTime: 0,
                waterLevel: 60 + Math.random() * 40, // 60-100% water
                needsWater: false, // Will be calculated based on waterLevel
                hasWeeds: false
            });
        }
    }
    
    generatePathways() {
        // Concrete pathway from street to front door
        this.pathways = [
            { rect: { x: 480, y: 260, width: 40, height: 140 }, color: this.COLORS.concrete },
            { rect: { x: 400, y: 380, width: 200, height: 20 }, color: this.COLORS.concrete }
        ];
    }
    
    generateRobot() {
        // Start robot at charging station (position based on layout)
        const chargingCenter = {
            x: this.chargingStation.rect.x + this.chargingStation.rect.width / 2,
            y: this.chargingStation.rect.y + this.chargingStation.rect.height / 2
        };
        
        this.robot = {
            pos: { x: chargingCenter.x, y: chargingCenter.y },
            radius: 20,
            angle: 0,
            isMoving: false,
            isRotating: false,
            carriedItem: null,
            currentTask: null,
            isCharging: false
        };
        
        console.log(`Farm robot placed at charging station (${this.robot.pos.x}, ${this.robot.pos.y})`);
    }
    
    generateInitialObjects() {
        // Initial weeds
        this.generateWeeds();
        
        // Initial trash
        this.generateTrash();
        
        // Initial animals (chance)
        if (Math.random() < 0.4) {
            this.spawnAnimal();
        }
        
        // No package deliveries in farm setting
    }
    
    generateWeeds() {
        this.weeds = [];
        const weedCount = Math.floor(Math.random() * 8) + 3;
        
        for (let i = 0; i < weedCount; i++) {
            // Place weeds randomly in grass areas (but not in house)
            const grassArea = this.grassAreas[Math.floor(Math.random() * this.grassAreas.length)];
            let pos;
            let attempts = 0;
            do {
                pos = {
                    x: grassArea.rect.x + Math.random() * grassArea.rect.width,
                    y: grassArea.rect.y + Math.random() * grassArea.rect.height
                };
                attempts++;
            } while (this.isPositionInHouse(pos) && attempts < 10);
            
            this.weeds.push({
                pos: pos,
                size: Math.random() * 3 + 2,
                age: Math.random() * 10
            });
        }
    }
    
    generateTrash() {
        this.trash = [];
        const trashCount = Math.floor(Math.random() * 6) + 2;
        
        for (let i = 0; i < trashCount; i++) {
            let pos;
            let attempts = 0;
            do {
                pos = {
                    x: 200 + Math.random() * 600,
                    y: 300 + Math.random() * 350
                };
                attempts++;
            } while (this.isPositionInHouse(pos) && attempts < 10);
            
            this.trash.push({
                type: 'trash',
                pos: pos,
                item: ['wrapper', 'food_waste', 'paper'][Math.floor(Math.random() * 3)]
            });
        }
    }
    
    spawnAnimal() {
        const animalTypes = ['rabbit', 'deer', 'fox', 'bird'];
        const animalType = animalTypes[Math.floor(Math.random() * animalTypes.length)];
        
        let pos;
        let attempts = 0;
        do {
            pos = {
                x: 200 + Math.random() * 600,
                y: 300 + Math.random() * 300
            };
            attempts++;
        } while (this.isPositionInHouse(pos) && attempts < 10);
        
        const newAnimal = {
            type: animalType,
            id: Date.now() + Math.random(), // Unique ID
            pos: pos,
            targetPlant: this.plants[Math.floor(Math.random() * this.plants.length)],
            speed: Math.random() * 1 + 0.5,
            warningCount: 0,
            isLeaving: false,
            timeUntilLeave: 0,
            spawnTime: Date.now()
        };
        
        this.animals.push(newAnimal);
        this.animalSpawnTimes.set(newAnimal.id, Date.now());
        console.log(`🐺 Wild ${animalType} appeared near crops! Warn it quickly for bonus points!`);
    }
    
    // No package delivery system in farm setting
    
    update() {
        if (!this.gameRunning) return;
        
        // Update game time
        this.gameTime += this.timeSpeed / 60; // Convert to game minutes
        
        // Update battery (drain when moving, charge when at station)
        this.updateBattery();
        
        // Update systems
        this.updatePlantGrowth();
        this.updateAnimals();
        this.updateGrassGrowth();
        this.updateVisionSystem();
        
        // Handle robot navigation and movement
        if (this.isNavigating) {
            if (this.keys['ArrowLeft'] || this.keys['ArrowRight'] || 
                this.keys['ArrowUp'] || this.keys['ArrowDown']) {
                this.cancelNavigation();
            } else {
                this.updateNavigation();
            }
        }
        
        // Manual robot movement
        if (!this.isNavigating) {
            this.handleManualMovement();
        }
        
        // Check for critical battery (earlier warning with faster drain)
        if (this.batteryLevel < 25 && !this.isCharging && !this.hasTask('emergency_charge')) {
            this.addTask('emergency_charge', 'Battery low! Return to charging station');
        }
        
        // Generate new tasks based on conditions
        this.checkAndGenerateTasks();
        
        // Update UI
        this.updateTimeDisplay();
        this.updateUI();
        this.logGameState();
        
        // Check day progression
        if (this.gameTime >= 1440) { // 24 hours passed
            this.advanceDay();
        }
    }
    
    updateBattery() {
        const chargingDistance = Math.sqrt(
            Math.pow(this.robot.pos.x - (this.chargingStation.rect.x + 25), 2) + 
            Math.pow(this.robot.pos.y - (this.chargingStation.rect.y + 25), 2)
        );
        
        const waterDistance = Math.sqrt(
            Math.pow(this.robot.pos.x - (this.waterStation.rect.x + 25), 2) + 
            Math.pow(this.robot.pos.y - (this.waterStation.rect.y + 25), 2)
        );
        
        // Battery charging
        if (chargingDistance < 30) {
            this.isCharging = true;
            this.chargingStation.isActive = true;
            this.batteryLevel = Math.min(100, this.batteryLevel + this.chargingRate / 60);
        } else {
            this.isCharging = false;
            this.chargingStation.isActive = false;
            
            // Drain battery when moving or idle
            if (this.robot.isMoving || this.robot.isRotating) {
                this.batteryLevel = Math.max(0, this.batteryLevel - this.batteryDrainRate / 60);
            } else {
                // Slow drain when idle
                this.batteryLevel = Math.max(0, this.batteryLevel - this.batteryIdleDrain / 60);
            }
        }
        
        // Water refilling at separate water station
        if (waterDistance < 30) {
            this.isRefillWater = true;
            this.waterStation.isActive = true;
            this.waterLevel = Math.min(this.maxWaterLevel, this.waterLevel + this.waterRefillRate / 60);
        } else {
            this.isRefillWater = false;
            this.waterStation.isActive = false;
        }
        
        // Robot stops if battery is dead
        if (this.batteryLevel <= 0 && !this.isCharging) {
            this.gameOver('Battery depleted!');
        }
    }
    
    updatePlantGrowth() {
        this.plants.forEach(plant => {
            // Plants grow over time
            plant.growth += 0.01; // Slow growth per frame
            plant.age += 0.01;
            
            // Water level decreases slowly over time
            if (Math.random() < 0.0005) { // Very slow water evaporation
                plant.waterLevel = Math.max(0, plant.waterLevel - 0.5);
            }
            
            // Update water needs
            plant.needsWater = plant.waterLevel < 30;
            
            plant.waterLevel = Math.max(0, Math.min(100, plant.waterLevel));
        });
    }
    
    updateAnimals() {
        this.animals.forEach((animal, index) => {
            if (animal.isLeaving) {
                // Move animal away from yard
                animal.pos.x += Math.random() * 4 - 2;
                animal.pos.y += Math.random() * 4 - 2;
                animal.timeUntilLeave--;
                
                if (animal.timeUntilLeave <= 0) {
                    this.animals.splice(index, 1); // Remove animal
                }
            } else if (animal.targetPlant) {
                // Move towards target plant
                const dx = animal.targetPlant.pos.x - animal.pos.x;
                const dy = animal.targetPlant.pos.y - animal.pos.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                
                if (distance > 5) {
                    animal.pos.x += (dx / distance) * animal.speed;
                    animal.pos.y += (dy / distance) * animal.speed;
                }
            }
        });
        
        // Spawn new animals occasionally
        if (Math.random() < 0.0005 && this.animals.length < 3) { // Very rare spawn
            this.spawnAnimal();
        }
    }
    
    updateGrassGrowth() {
        this.grassAreas.forEach(area => {
            area.grassHeight += area.growthRate / 1000; // Slow growth per frame
            area.needsMowing = area.grassHeight > 6;
            
            // Water level decreases slowly over time
            if (Math.random() < 0.0003) { // Very slow evaporation
                area.waterLevel = Math.max(0, area.waterLevel - 0.3);
            }
            
            // Update watering needs
            area.needsWatering = area.waterLevel < 40;
        });
    }
    
    updateVisionSystem() {
        if (!this.robot) return;
        
        // Clear previously detected objects
        this.detectedObjects.clear();
        this.detectedAnimals.clear();
        
        // Check trash in vision cone
        this.trash.forEach(item => {
            if (this.isInVisionCone(item.pos)) {
                this.detectedObjects.add(item);
            }
        });
        
        // Check weeds in vision cone
        this.weeds.forEach(weed => {
            if (this.isInVisionCone(weed.pos)) {
                this.detectedObjects.add(weed);
            }
        });
        
        // Check animals within detection range (not vision cone based)
        this.animals.forEach(animal => {
            const distance = this.getDistance(this.robot.pos, animal.pos);
            if (distance <= this.animalDetectionRange) {
                this.detectedAnimals.add(animal);
            }
        });
    }
    
    isInVisionCone(targetPos) {
        if (!this.robot) return false;
        
        const dx = targetPos.x - this.robot.pos.x;
        const dy = targetPos.y - this.robot.pos.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        // Check if within vision range
        if (distance > this.visionRange) return false;
        
        // Check if within vision angle
        const targetAngle = Math.atan2(dy, dx);
        let angleDiff = targetAngle - this.robot.angle;
        
        // Normalize angle difference
        while (angleDiff > Math.PI) angleDiff -= 2 * Math.PI;
        while (angleDiff < -Math.PI) angleDiff += 2 * Math.PI;
        
        return Math.abs(angleDiff) <= this.visionAngle / 2;
    }
    
    checkAndGenerateTasks() {
        // Animal warning task - only for detected animals
        if (this.detectedAnimals.size > 0 && !this.hasTask('warn_animals')) {
            this.addTask('warn_animals', 'Warn wild animals away from crops!');
        }
        
        // Lawn mowing task
        if (this.grassAreas.some(area => area.needsMowing) && !this.hasTask('mow_lawn')) {
            this.addTask('mow_lawn', 'Mow the grass - getting too long');
        }
        
        // Weed removal task
        if (this.weeds.length > 5 && !this.hasTask('remove_weeds')) {
            this.addTask('remove_weeds', 'Remove weeds from the farm');
        }
        
        // Trash cleanup task
        if (this.trash.length > 3 && !this.hasTask('cleanup_trash')) {
            this.addTask('cleanup_trash', 'Clean up farm trash');
        }
        
        // Daily photo task
        if (Date.now() - this.lastPhotoTime > this.photoInterval * 60 * 1000 / this.timeSpeed && !this.hasTask('take_photos')) {
            this.addTask('take_photos', 'Take daily photos of crops to monitor growth');
        }
        
        // Plant watering task
        if (this.plants.some(plant => plant.needsWater) && !this.hasTask('water_plants')) {
            this.addTask('water_plants', 'Water thirsty crops and trees');
        }
        
        // Lawn watering task
        if (this.grassAreas.some(area => area.needsWatering) && !this.hasTask('water_lawn')) {
            this.addTask('water_lawn', 'Water dry pasture areas');
        }
        
        // Water refill task
        if (this.waterLevel < 30 && !this.hasTask('refill_water')) {
            this.addTask('refill_water', 'Water tank low! Go to water station to refill');
        }
    }
    
    // ... existing code continues with more methods ...

    render() {
        // Clear canvas with sky background
        const gradient = this.ctx.createLinearGradient(0, 0, 0, this.canvas.height);
        gradient.addColorStop(0, this.COLORS.sky);
        gradient.addColorStop(1, '#98FB98'); // Light green for ground
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw grass areas
        this.grassAreas.forEach(area => {
            // Base grass color adjusted by water level
            let baseColor = area.needsMowing ? this.COLORS.darkGrass : this.COLORS.grass;
            if (area.waterLevel < 40) {
                baseColor = area.needsMowing ? '#4A5D23' : '#8BC34A'; // Drier looking grass
            }
            
            this.ctx.fillStyle = baseColor;
            this.ctx.fillRect(area.rect.x, area.rect.y, area.rect.width, area.rect.height);
            
            // Grass texture
            this.ctx.strokeStyle = area.needsMowing ? '#1B5E20' : '#2E7D32';
            this.ctx.lineWidth = 1;
            for (let i = 0; i < area.rect.width; i += 10) {
                this.ctx.beginPath();
                this.ctx.moveTo(area.rect.x + i, area.rect.y);
                this.ctx.lineTo(area.rect.x + i, area.rect.y + area.rect.height);
                this.ctx.stroke();
            }
            
            // Water level indicator for grass
            const centerX = area.rect.x + area.rect.width / 2;
            const centerY = area.rect.y + area.rect.height / 2;
            
            this.ctx.fillStyle = '#FFFFFF';
            this.ctx.font = '12px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(`Water: ${Math.round(area.waterLevel)}%`, centerX, centerY);
            
            // Dry grass indicator
            if (area.needsWatering) {
                this.ctx.fillStyle = '#FF9800';
                this.ctx.font = '16px Arial';
                this.ctx.fillText('💧', centerX, centerY - 20);
            }
            
            this.ctx.textAlign = 'left';
        });
        
        // Draw pathways
        if (this.pathways) {
            this.pathways.forEach(path => {
                this.ctx.fillStyle = path.color;
                this.ctx.fillRect(path.rect.x, path.rect.y, path.rect.width, path.rect.height);
            });
        }
        
        // Draw house
        this.ctx.fillStyle = this.COLORS.farmHouse;
        this.ctx.fillRect(this.farmHouse.rect.x, this.farmHouse.rect.y, this.farmHouse.rect.width, this.farmHouse.rect.height);
        this.ctx.strokeStyle = '#5D4037';
        this.ctx.lineWidth = 3;
        this.ctx.strokeRect(this.farmHouse.rect.x, this.farmHouse.rect.y, this.farmHouse.rect.width, this.farmHouse.rect.height);
        
        // Farm house label
        this.ctx.fillStyle = '#3E2723';
        this.ctx.font = 'bold 16px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('FARM HOUSE', this.farmHouse.rect.x + this.farmHouse.rect.width/2, this.farmHouse.rect.y + 30);
        this.ctx.textAlign = 'left';
        
        // Draw charging station
        this.ctx.fillStyle = this.chargingStation.isActive ? '#FFF176' : this.COLORS.charging;
        this.ctx.fillRect(this.chargingStation.rect.x, this.chargingStation.rect.y, 
                         this.chargingStation.rect.width, this.chargingStation.rect.height);
        this.ctx.strokeStyle = '#F57F17';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(this.chargingStation.rect.x, this.chargingStation.rect.y,
                          this.chargingStation.rect.width, this.chargingStation.rect.height);
        
        // Charging station label
        this.ctx.fillStyle = '#E65100';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('CHARGE', this.chargingStation.rect.x + 25, this.chargingStation.rect.y + 30);
        
        // Draw water station
        this.ctx.fillStyle = this.waterStation.isActive ? '#64B5F6' : this.waterStation.color;
        this.ctx.fillRect(this.waterStation.rect.x, this.waterStation.rect.y, 
                         this.waterStation.rect.width, this.waterStation.rect.height);
        this.ctx.strokeStyle = '#1976D2';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(this.waterStation.rect.x, this.waterStation.rect.y,
                          this.waterStation.rect.width, this.waterStation.rect.height);
        
        // Water station label
        this.ctx.fillStyle = '#0D47A1';
        this.ctx.font = '12px Arial';
        this.ctx.fillText('WATER', this.waterStation.rect.x + 25, this.waterStation.rect.y + 30);
        this.ctx.textAlign = 'left';
        
        // Draw bins
        this.bins.forEach(bin => {
            this.ctx.fillStyle = bin.color;
            this.ctx.fillRect(bin.rect.x, bin.rect.y, bin.rect.width, bin.rect.height);
            this.ctx.strokeStyle = '#000000';
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(bin.rect.x, bin.rect.y, bin.rect.width, bin.rect.height);
            
            // Bin labels
            this.ctx.fillStyle = '#FFFFFF';
            this.ctx.font = '10px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(bin.type.toUpperCase(), bin.rect.x + 20, bin.rect.y + 25);
            this.ctx.textAlign = 'left';
        });
        
        // Draw plants
        this.plants.forEach(plant => {
            // Plant color based on water level instead of health
            const plantColor = plant.waterLevel > 50 ? this.COLORS.plant : 
                              plant.waterLevel > 25 ? '#FFA726' : '#8BC34A';
            this.ctx.fillStyle = plantColor;
            
            if (plant.type === 'flower') {
                this.ctx.beginPath();
                this.ctx.arc(plant.pos.x, plant.pos.y, 8, 0, 2 * Math.PI);
                this.ctx.fill();
                
                // Flower petals
                this.ctx.fillStyle = this.COLORS.flower;
                for (let i = 0; i < 5; i++) {
                    const angle = (i / 5) * 2 * Math.PI;
                    const petalX = plant.pos.x + Math.cos(angle) * 6;
                    const petalY = plant.pos.y + Math.sin(angle) * 6;
                    this.ctx.beginPath();
                    this.ctx.arc(petalX, petalY, 3, 0, 2 * Math.PI);
                    this.ctx.fill();
                }
            } else if (plant.type === 'tree') {
                // Tree trunk
                this.ctx.fillStyle = '#8D6E63';
                this.ctx.fillRect(plant.pos.x - 3, plant.pos.y - 5, 6, 20);
                
                // Tree crown
                this.ctx.fillStyle = plantColor;
                this.ctx.beginPath();
                this.ctx.arc(plant.pos.x, plant.pos.y - 10, 15, 0, 2 * Math.PI);
                this.ctx.fill();
            }
            
            // Water level indicator only
            const waterColor = plant.waterLevel > 60 ? '#2196F3' : 
                              plant.waterLevel > 30 ? '#FF9800' : '#F44336';
            this.ctx.fillStyle = waterColor;
            this.ctx.font = '10px Arial';
            this.ctx.fillText(`${Math.round(plant.waterLevel)}%`, plant.pos.x + 12, plant.pos.y - 5);
            
            // Thirsty indicator
            if (plant.needsWater) {
                this.ctx.fillStyle = '#03A9F4';
                this.ctx.font = '14px Arial';
                this.ctx.fillText('💧', plant.pos.x - 8, plant.pos.y - 15);
            }
        });
        
        // No packages in farm setting
        
        // Draw animals (only if detected within range)
        this.animals.forEach(animal => {
            const isDetected = this.detectedAnimals.has(animal);
            if (!isDetected) return; // Only draw detected animals
            
            this.ctx.fillStyle = this.COLORS.animal;
            this.ctx.beginPath();
            this.ctx.arc(animal.pos.x, animal.pos.y, 8, 0, 2 * Math.PI);
            this.ctx.fill();
            
            // Animal type indicator
            let emoji = '🐺'; // Wild animal default
            if (animal.type === 'rabbit') emoji = '🐰';
            if (animal.type === 'bird') emoji = '🐦';
            if (animal.type === 'deer') emoji = '🦌';
            if (animal.type === 'fox') emoji = '🦊';
            
            this.ctx.font = '16px Arial';
            this.ctx.fillText(emoji, animal.pos.x - 8, animal.pos.y + 5);
            
            // Detection indicator
            this.ctx.fillStyle = '#FF5722';
            this.ctx.font = '12px Arial';
            this.ctx.fillText('!', animal.pos.x - 3, animal.pos.y - 15);
        });
        
        // Draw trash (only if detected)
        this.trash.forEach(item => {
            const isDetected = this.detectedObjects.has(item);
            if (!isDetected) return; // Only draw if detected
            
            const color = item.type === 'recycling' ? this.COLORS.recycling : this.COLORS.trash;
            
            this.ctx.fillStyle = color;
            this.ctx.beginPath();
            this.ctx.arc(item.pos.x, item.pos.y, 6, 0, 2 * Math.PI);
            this.ctx.fill();
            
            this.ctx.strokeStyle = '#000000';
            this.ctx.lineWidth = 1;
            this.ctx.stroke();
            
            // Detection indicator
            this.ctx.fillStyle = '#FFEB3B';
            this.ctx.font = '12px Arial';
            this.ctx.fillText('!', item.pos.x - 3, item.pos.y - 12);
        });
        
        // Draw weeds (only if detected)
        this.weeds.forEach(weed => {
            const isDetected = this.detectedObjects.has(weed);
            if (!isDetected) return; // Only draw if detected
            
            this.ctx.fillStyle = this.COLORS.weed;
            this.ctx.beginPath();
            this.ctx.arc(weed.pos.x, weed.pos.y, weed.size, 0, 2 * Math.PI);
            this.ctx.fill();
            
            // Weed spikes
            for (let i = 0; i < 6; i++) {
                const angle = (i / 6) * 2 * Math.PI;
                const spikeX = weed.pos.x + Math.cos(angle) * weed.size;
                const spikeY = weed.pos.y + Math.sin(angle) * weed.size;
                this.ctx.strokeStyle = this.COLORS.weed;
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.moveTo(weed.pos.x, weed.pos.y);
                this.ctx.lineTo(spikeX, spikeY);
                this.ctx.stroke();
            }
            
            // Detection indicator
            this.ctx.fillStyle = '#FFEB3B';
            this.ctx.font = '12px Arial';
            this.ctx.fillText('!', weed.pos.x - 3, weed.pos.y - 8);
        });
        
        // Draw navigation path if active
        if (this.isNavigating && this.navigationPath && this.navigationPath.length > 0) {
            this.ctx.strokeStyle = '#FF6B6B';
            this.ctx.lineWidth = 3;
            this.ctx.setLineDash([5, 5]);
            this.ctx.beginPath();
            
            this.ctx.moveTo(this.robot.pos.x, this.robot.pos.y);
            for (const waypoint of this.navigationPath) {
                this.ctx.lineTo(waypoint.x, waypoint.y);
            }
            this.ctx.stroke();
            this.ctx.setLineDash([]);
        }
        
        // Draw robot
        if (this.robot) {
            // Robot body
            this.ctx.fillStyle = this.COLORS.robot;
            this.ctx.beginPath();
            this.ctx.arc(this.robot.pos.x, this.robot.pos.y, this.robot.radius, 0, 2 * Math.PI);
            this.ctx.fill();
            
            this.ctx.strokeStyle = '#1976D2';
            this.ctx.lineWidth = 3;
            this.ctx.stroke();
            
            // Robot direction indicator
            const frontX = this.robot.pos.x + Math.cos(this.robot.angle) * 15;
            const frontY = this.robot.pos.y + Math.sin(this.robot.angle) * 15;
            
            this.ctx.fillStyle = '#FFFFFF';
            this.ctx.beginPath();
            this.ctx.moveTo(frontX, frontY);
            this.ctx.lineTo(
                frontX - Math.cos(this.robot.angle + 2.5) * 8,
                frontY - Math.sin(this.robot.angle + 2.5) * 8
            );
            this.ctx.lineTo(
                frontX - Math.cos(this.robot.angle - 2.5) * 8,
                frontY - Math.sin(this.robot.angle - 2.5) * 8
            );
            this.ctx.closePath();
            this.ctx.fill();
            
            // Battery indicator on robot
            const batteryColor = this.batteryLevel > 50 ? '#4CAF50' : 
                                this.batteryLevel > 20 ? '#FF9800' : '#F44336';
            this.ctx.fillStyle = batteryColor;
            this.ctx.fillRect(this.robot.pos.x - 8, this.robot.pos.y - 25, 16, 4);
            
            // Water indicator on robot
            const waterColor = this.waterLevel > 50 ? '#2196F3' : 
                              this.waterLevel > 20 ? '#FF9800' : '#F44336';
            this.ctx.fillStyle = waterColor;
            this.ctx.fillRect(this.robot.pos.x - 8, this.robot.pos.y - 35, 16, 4);
            
            this.ctx.fillStyle = '#000000';
            this.ctx.font = '9px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(`B:${Math.round(this.batteryLevel)}%`, this.robot.pos.x, this.robot.pos.y - 40);
            this.ctx.fillText(`W:${Math.round(this.waterLevel)}%`, this.robot.pos.x, this.robot.pos.y - 28);
            this.ctx.textAlign = 'left';
            
            // Carried item
            if (this.robot.carriedItem) {
                this.ctx.fillStyle = this.COLORS.package;
                this.ctx.fillRect(this.robot.pos.x + 15, this.robot.pos.y - 10, 15, 10);
            }
            
            // Charging indicator
            if (this.isCharging) {
                this.ctx.fillStyle = '#FFEB3B';
                this.ctx.font = '20px Arial';
                this.ctx.fillText('⚡', this.robot.pos.x + 10, this.robot.pos.y - 15);
            }
            
            // Water refilling indicator
            if (this.isRefillWater) {
                this.ctx.fillStyle = '#2196F3';
                this.ctx.font = '20px Arial';
                this.ctx.fillText('💧', this.robot.pos.x - 25, this.robot.pos.y - 15);
            }
            
            // Draw vision cone
            this.ctx.strokeStyle = 'rgba(255, 255, 0, 0.3)';
            this.ctx.fillStyle = 'rgba(255, 255, 0, 0.1)';
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.moveTo(this.robot.pos.x, this.robot.pos.y);
            
            // Calculate vision cone arc
            const leftAngle = this.robot.angle - this.visionAngle / 2;
            const rightAngle = this.robot.angle + this.visionAngle / 2;
            
            this.ctx.arc(this.robot.pos.x, this.robot.pos.y, this.visionRange, leftAngle, rightAngle);
            this.ctx.closePath();
            this.ctx.fill();
            this.ctx.stroke();
        }
        
        // Draw point message
        if (this.lastPointMessage && Date.now() - this.pointMessageTime < 3000) {
            const alpha = Math.max(0, 1 - (Date.now() - this.pointMessageTime) / 3000);
            this.ctx.fillStyle = `rgba(76, 175, 80, ${alpha})`;
            this.ctx.font = 'bold 18px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(this.lastPointMessage, this.canvas.width / 2, 150);
            this.ctx.textAlign = 'left';
        }
        
        // Draw layout type indicator
        this.ctx.fillStyle = '#2196F3';
        this.ctx.font = '14px Arial';
        this.ctx.fillText(`Layout: ${this.layoutType}`, 20, this.canvas.height - 30);
    }

    // Method implementations
    performPickupAction() {
        if (this.robot.carriedItem) {
            this.logAction('pickup_failed', { reason: 'already_carrying' });
            return;
        }
        
        // Check for visible trash (in vision cone, reasonable pickup distance)
        for (let i = 0; i < this.trash.length; i++) {
            const trashItem = this.trash[i];
            if (this.detectedObjects.has(trashItem) && 
                this.getDistance(this.robot.pos, trashItem.pos) < 80) {
                this.robot.carriedItem = { type: 'trash', item: trashItem };
                this.trash.splice(i, 1);
                this.showPointMessage('+20 points - Farm trash collected!');
                this.logAction('pickup_success', { itemType: 'trash' });
                return;
            }
        }
        
        // Check for visible weeds (in vision cone, reasonable pickup distance)
        for (let i = 0; i < this.weeds.length; i++) {
            const weed = this.weeds[i];
            if (this.detectedObjects.has(weed) && 
                this.getDistance(this.robot.pos, weed.pos) < 60) {
                this.robot.carriedItem = { type: 'weed', item: weed };
                this.weeds.splice(i, 1);
                this.showPointMessage('+15 points - Weed removed!');
                this.logAction('pickup_success', { itemType: 'weed' });
                return;
            }
        }
        
        // Check if there are any items nearby but not detected
        let nearbyItemsNotSeen = [];
        
        // Check for nearby trash not in vision
        const nearbyTrash = this.trash.filter(item => 
            this.getDistance(this.robot.pos, item.pos) < 80 && 
            !this.detectedObjects.has(item)
        );
        if (nearbyTrash.length > 0) {
            nearbyItemsNotSeen.push('trash');
        }
        
        // Check for nearby weeds not in vision
        const nearbyWeeds = this.weeds.filter(weed => 
            this.getDistance(this.robot.pos, weed.pos) < 60 && 
            !this.detectedObjects.has(weed)
        );
        if (nearbyWeeds.length > 0) {
            nearbyItemsNotSeen.push('weeds');
        }
        
        if (nearbyItemsNotSeen.length > 0) {
            this.showPointMessage('Look around - there might be something nearby!');
            this.logAction('pickup_failed', { 
                reason: 'items_not_in_vision', 
                nearbyTypes: nearbyItemsNotSeen 
            });
        } else {
            this.logAction('pickup_failed', { reason: 'no_item_in_range' });
        }
    }
    
    performDropAction() {
        if (!this.robot.carriedItem) {
            this.logAction('drop_failed', { reason: 'no_item_carried' });
            return;
        }
        
        const item = this.robot.carriedItem;
        
        // Handle trash - drop in trash bin
        if (item.type === 'trash') {
            const trashBin = this.bins.find(bin => bin.type === 'trash');
            if (trashBin && this.getDistance(this.robot.pos, this.getBinCenter(trashBin)) < 40) {
                this.score += 20;
                this.showPointMessage('+20 points - Farm trash disposed!');
                this.logAction('drop_success', { itemType: 'trash', binType: 'trash', points: 20 });
            } else {
                // Drop trash back on ground
                this.trash.push({
                    ...item.item,
                    pos: { x: this.robot.pos.x, y: this.robot.pos.y }
                });
                this.logAction('drop_failed', { reason: 'wrong_bin' });
            }
        }
        
        // Handle weeds - drop in trash bin (since no compost bin)
        else if (item.type === 'weed') {
            const trashBin = this.bins.find(bin => bin.type === 'trash');
            if (trashBin && this.getDistance(this.robot.pos, this.getBinCenter(trashBin)) < 40) {
                this.score += 15;
                this.showPointMessage('+15 points - Weed disposed!');
                this.logAction('drop_success', { itemType: 'weed', binType: 'trash', points: 15 });
            } else {
                // Drop weed back on ground
                this.weeds.push({
                    ...item.item,
                    pos: { x: this.robot.pos.x, y: this.robot.pos.y }
                });
                this.logAction('drop_failed', { reason: 'wrong_bin' });
            }
        }
        
        this.robot.carriedItem = null;
    }
    
    takePhoto() {
        // Find nearby plants
        const nearbyPlants = this.plants.filter(plant => 
            this.getDistance(this.robot.pos, plant.pos) < 50
        );
        
        if (nearbyPlants.length === 0) {
            this.logAction('photo_failed', { reason: 'no_plants_nearby' });
            return;
        }
        
        const photo = {
            timestamp: Date.now(),
            plants: nearbyPlants.map(plant => ({
                type: plant.type,
                health: plant.health,
                growth: plant.growth,
                pos: plant.pos
            })),
            robotPosition: { x: this.robot.pos.x, y: this.robot.pos.y }
        };
        
        this.photosTaken.push(photo);
        nearbyPlants.forEach(plant => {
            plant.lastPhotoTime = Date.now();
        });
        
        this.score += 10;
        this.lastPhotoTime = this.gameTime;
        this.showPointMessage(`+10 points - Photographed ${nearbyPlants.length} plants!`);
        this.logAction('photo_taken', { plantsCount: nearbyPlants.length, points: 10 });
        this.removeTask('take_photos');
    }
    
    mowLawn() {
        // Check if robot is in a grass area that needs mowing
        const currentGrassArea = this.grassAreas.find(area => 
            this.robot.pos.x >= area.rect.x && 
            this.robot.pos.x <= area.rect.x + area.rect.width &&
            this.robot.pos.y >= area.rect.y && 
            this.robot.pos.y <= area.rect.y + area.rect.height
        );
        
        if (!currentGrassArea) {
            this.logAction('mow_failed', { reason: 'not_on_grass' });
            return;
        }
        
        if (!currentGrassArea.needsMowing) {
            this.logAction('mow_failed', { reason: 'grass_not_long_enough' });
            return;
        }
        
        // Mow the grass
        currentGrassArea.grassHeight = 2; // Cut to standard height
        currentGrassArea.needsMowing = false;
        this.score += 30;
        this.showPointMessage('+30 points - Lawn mowed!');
        this.logAction('mow_success', { points: 30 });
        
        // Check if all grass areas are mowed
        if (!this.grassAreas.some(area => area.needsMowing)) {
            this.removeTask('mow_lawn');
        }
    }
    
    warnAnimals() {
        // Only warn animals that are detected (within detection range)
        const detectedAnimalsArray = Array.from(this.detectedAnimals);
        
        if (detectedAnimalsArray.length === 0) {
            this.logAction('warn_failed', { reason: 'no_animals_detected' });
            this.showPointMessage('No wild animals detected nearby!');
            return;
        }
        
        let totalPoints = 0;
        let bonusMessages = [];
        
        detectedAnimalsArray.forEach(animal => {
            animal.warningCount++;
            
            // Calculate time-sensitive bonus
            const timeSinceSpawn = Date.now() - animal.spawnTime;
            let basePoints = 15;
            let timeBonus = 0;
            
            if (timeSinceSpawn < 5000) { // Within 5 seconds
                timeBonus = 20;
                bonusMessages.push('Quick reaction bonus!');
            } else if (timeSinceSpawn < 10000) { // Within 10 seconds
                timeBonus = 15;
                bonusMessages.push('Good timing bonus!');
            } else if (timeSinceSpawn < 20000) { // Within 20 seconds
                timeBonus = 10;
                bonusMessages.push('On-time bonus!');
            }
            
            const animalPoints = basePoints + timeBonus;
            totalPoints += animalPoints;
            
            if (animal.warningCount >= 2) {
                animal.isLeaving = true;
                animal.timeUntilLeave = 300; // 5 seconds to leave
                this.animalSpawnTimes.delete(animal.id);
            }
        });
        
        this.score += totalPoints;
        
        let message = `+${totalPoints} points - Warned ${detectedAnimalsArray.length} wild animal(s)!`;
        if (bonusMessages.length > 0) {
            message += ` ${bonusMessages[0]}`;
        }
        
        this.showPointMessage(message);
        this.logAction('warn_success', { 
            animalsWarned: detectedAnimalsArray.length, 
            totalPoints: totalPoints,
            bonuses: bonusMessages 
        });
        
        // Check if all detected animals are leaving
        if (!this.detectedAnimals.size || Array.from(this.detectedAnimals).every(animal => animal.isLeaving)) {
            this.removeTask('warn_animals');
        }
    }
    
    waterPlants() {
        if (this.waterLevel < this.waterUsagePerAction) {
            this.logAction('water_failed', { reason: 'insufficient_water', waterLevel: this.waterLevel });
            this.showPointMessage('Not enough water! Return to charging station to refill.');
            return;
        }
        
        let watered = false;
        let waterUsed = 0;
        
        // Check for nearby plants
        const nearbyPlants = this.plants.filter(plant => 
            this.getDistance(this.robot.pos, plant.pos) < 50
        );
        
        if (nearbyPlants.length > 0) {
            nearbyPlants.forEach(plant => {
                if (this.waterLevel >= this.waterUsagePerAction) {
                    plant.waterLevel = Math.min(100, plant.waterLevel + 30);
                    plant.needsWater = false;
                    this.waterLevel -= this.waterUsagePerAction;
                    waterUsed += this.waterUsagePerAction;
                    watered = true;
                }
            });
            
            if (watered) {
                this.score += 5 * nearbyPlants.length;
                this.showPointMessage(`+${5 * nearbyPlants.length} points - Watered ${nearbyPlants.length} plants!`);
                                 this.logAction('water_success', { 
                     itemType: 'plants', 
                     count: nearbyPlants.length, 
                     waterUsed: waterUsed,
                     points: 5 * nearbyPlants.length 
                 });
                 
                 // Check if all plants are watered
                 if (!this.plants.some(plant => plant.needsWater)) {
                     this.removeTask('water_plants');
                 }
                 return;
            }
        }
        
        // Check if robot is on grass that needs watering
        const currentGrassArea = this.grassAreas.find(area => 
            this.robot.pos.x >= area.rect.x && 
            this.robot.pos.x <= area.rect.x + area.rect.width &&
            this.robot.pos.y >= area.rect.y && 
            this.robot.pos.y <= area.rect.y + area.rect.height
        );
        
        if (currentGrassArea) {
            if (currentGrassArea.needsWatering) {
                currentGrassArea.waterLevel = Math.min(100, currentGrassArea.waterLevel + 40);
                currentGrassArea.needsWatering = false;
                this.waterLevel -= this.waterUsagePerAction;
                this.score += 15;
                this.showPointMessage('+15 points - Lawn watered!');
                this.logAction('water_success', { 
                    itemType: 'lawn', 
                    waterUsed: this.waterUsagePerAction, 
                    points: 15 
                });
                watered = true;
                
                // Check if all lawn areas are watered
                if (!this.grassAreas.some(area => area.needsWatering)) {
                    this.removeTask('water_lawn');
                }
            } else {
                this.showPointMessage('This grass area is already well watered!');
                this.logAction('water_failed', { reason: 'grass_already_watered' });
                return;
            }
        }
        
        if (!watered) {
            this.logAction('water_failed', { reason: 'nothing_to_water' });
            this.showPointMessage('Nothing nearby needs watering!');
        }
    }
    
    goToChargingStation() {
        const chargingCenter = {
            x: this.chargingStation.rect.x + this.chargingStation.rect.width / 2,
            y: this.chargingStation.rect.y + this.chargingStation.rect.height / 2
        };
        this.setNavigationTarget(chargingCenter.x, chargingCenter.y);
        this.logAction('navigate_to_charging', { targetPos: chargingCenter });
    }
    
    goToWaterStation() {
        const waterCenter = {
            x: this.waterStation.rect.x + this.waterStation.rect.width / 2,
            y: this.waterStation.rect.y + this.waterStation.rect.height / 2
        };
        this.setNavigationTarget(waterCenter.x, waterCenter.y);
        this.logAction('navigate_to_water', { targetPos: waterCenter });
    }
    
    // Helper methods
    getDistance(pos1, pos2) {
        return Math.sqrt(Math.pow(pos1.x - pos2.x, 2) + Math.pow(pos1.y - pos2.y, 2));
    }
    
    isInHouse(pos) {
        return pos.x >= this.farmHouse.rect.x && 
               pos.x <= this.farmHouse.rect.x + this.farmHouse.rect.width &&
               pos.y >= this.farmHouse.rect.y && 
               pos.y <= this.farmHouse.rect.y + this.farmHouse.rect.height;
    }
    
    isPositionInHouse(pos) {
        return this.isInHouse(pos);
    }
    
    isInHouseThroughDoor(pos) {
        // Farm house entry - simplified
        return this.isInHouse(pos);
    }
    
    showPointMessage(message) {
        this.lastPointMessage = message;
        this.pointMessageTime = Date.now();
        console.log('🎯 ' + message);
    }
    
    findNearestBin(itemType) {
        // Only trash bin in farm setting
        return this.bins.find(bin => bin.type === 'trash');
    }
    
    getBinCenter(bin) {
        return {
            x: bin.rect.x + bin.rect.width / 2,
            y: bin.rect.y + bin.rect.height / 2
        };
    }
    
    // Task management
    addTask(taskId, description) {
        if (!this.hasTask(taskId)) {
            this.activeTasks.push({ id: taskId, description: description, startTime: Date.now() });
            this.updateTaskDisplay();
        }
    }
    
    removeTask(taskId) {
        const index = this.activeTasks.findIndex(task => task.id === taskId);
        if (index !== -1) {
            this.completedTasks.push(this.activeTasks[index]);
            this.activeTasks.splice(index, 1);
            this.updateTaskDisplay();
        }
    }
    
    hasTask(taskId) {
        return this.activeTasks.some(task => task.id === taskId);
    }
    
    updateTaskDisplay() {
        const taskEl = document.getElementById('task-instruction');
        if (this.activeTasks.length > 0) {
            taskEl.textContent = this.activeTasks[0].description;
        } else {
            taskEl.textContent = 'All tasks completed! Great job!';
        }
    }
    
    advanceDay() {
        this.dayCount++;
        this.gameTime = 0;
        this.lastPhotoTime = 0;
        
        // Daily events
        this.generateWeeds(); // New weeds grow
        this.generateTrash(); // New trash appears
        
        // Spawn new animals occasionally
        if (Math.random() < 0.6) {
            this.spawnAnimal();
        }
        
        this.logAction('day_advanced', { newDay: this.dayCount });
    }
    
    updateTimeDisplay() {
        const hours = Math.floor(this.gameTime / 60);
        const minutes = Math.floor(this.gameTime % 60);
        const timeString = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;
        
        // Update time display in UI
        document.getElementById('time').textContent = `Day ${this.dayCount} - ${timeString}`;
    }
    
    // Navigation methods (simplified versions from original)
    handleCanvasClick(event) {
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        this.setNavigationTarget(x, y);
    }
    
    setNavigationTarget(x, y) {
        this.navigationTarget = { x, y };
        this.isNavigating = true;
        this.logAction('navigation_target_set', { target: { x, y } });
    }
    
    cancelNavigation() {
        this.isNavigating = false;
        this.navigationTarget = null;
        this.logAction('navigation_cancelled');
    }
    
    updateNavigation() {
        if (!this.navigationTarget) return;
        
        const dx = this.navigationTarget.x - this.robot.pos.x;
        const dy = this.navigationTarget.y - this.robot.pos.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < 20) {
            this.isNavigating = false;
            this.navigationTarget = null;
            return;
        }
        
        const targetAngle = Math.atan2(dy, dx);
        this.robot.angle = targetAngle;
        
        const speed = 2;
        this.robot.pos.x += Math.cos(targetAngle) * speed;
        this.robot.pos.y += Math.sin(targetAngle) * speed;
        this.robot.isMoving = true;
    }
    
    handleManualMovement() {
        const moveSpeed = 2;
        const rotationSpeed = 0.08;
        let dx = 0, dy = 0;
        
        if (this.keys['ArrowLeft']) {
            this.robot.angle -= rotationSpeed;
            this.robot.isRotating = true;
        } else if (this.keys['ArrowRight']) {
            this.robot.angle += rotationSpeed;
            this.robot.isRotating = true;
        } else {
            this.robot.isRotating = false;
        }
        
        if (this.keys['ArrowUp']) {
            dx = Math.cos(this.robot.angle) * moveSpeed;
            dy = Math.sin(this.robot.angle) * moveSpeed;
            this.robot.isMoving = true;
        } else if (this.keys['ArrowDown']) {
            dx = -Math.cos(this.robot.angle) * moveSpeed;
            dy = -Math.sin(this.robot.angle) * moveSpeed;
            this.robot.isMoving = true;
        } else if (!this.robot.isRotating) {
            this.robot.isMoving = false;
        }
        
        if (dx !== 0 || dy !== 0) {
            const newX = this.robot.pos.x + dx;
            const newY = this.robot.pos.y + dy;
            
            // Boundary checking
            if (newX < 20 || newX > this.canvas.width - 20 || 
                newY < 20 || newY > this.canvas.height - 20) {
                return; // Don't move out of bounds
            }
            
            // House collision detection - farm house is accessible
            // (No special collision detection for farm house)
            
            // Check collision with bins
            for (const bin of this.bins) {
                if (newX >= bin.rect.x && newX <= bin.rect.x + bin.rect.width &&
                    newY >= bin.rect.y && newY <= bin.rect.y + bin.rect.height) {
                    return; // Block movement into bins
                }
            }
            
            // Check collision with charging station
            if (newX >= this.chargingStation.rect.x && 
                newX <= this.chargingStation.rect.x + this.chargingStation.rect.width &&
                newY >= this.chargingStation.rect.y && 
                newY <= this.chargingStation.rect.y + this.chargingStation.rect.height) {
                // Allow movement into charging station (robot can charge there)
            }
            
            // Check collision with water station
            if (newX >= this.waterStation.rect.x && 
                newX <= this.waterStation.rect.x + this.waterStation.rect.width &&
                newY >= this.waterStation.rect.y && 
                newY <= this.waterStation.rect.y + this.waterStation.rect.height) {
                // Allow movement into water station (robot can refill water there)
            }
            
            this.robot.pos.x = newX;
            this.robot.pos.y = newY;
        }
    }
    
    gameLoop() {
        this.update();
        this.render();
        requestAnimationFrame(() => this.gameLoop());
    }
    
    gameOver(reason) {
        this.gameRunning = false;
        this.logAction('game_over', { reason: reason });
        document.getElementById('gameOverModal').style.display = 'block';
    }
    
    // Utility methods from original game
    generateSessionId() {
        return 'farm_session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    logAction(actionType, data = {}) {
        const actionEntry = {
            timestamp: Date.now(),
            gameTime: this.gameTime,
            sessionId: this.gameSession.sessionId,
            level: this.level,
            dayCount: this.dayCount,
            actionType: actionType,
            batteryLevel: this.batteryLevel,
            robotState: this.robot ? {
                position: { x: this.robot.pos.x, y: this.robot.pos.y },
                angle: this.robot.angle,
                carriedItem: this.robot.carriedItem ? this.robot.carriedItem.type : null,
                isCharging: this.isCharging
            } : null,
            gameState: {
                score: this.score,
                activeTasks: this.activeTasks ? this.activeTasks.length : 0,
                animalsCount: this.animals ? this.animals.length : 0,
                trashCount: this.trash ? this.trash.length : 0,
                weedCount: this.weeds ? this.weeds.length : 0
            },
            ...data
        };
        
        this.actionLog.push(actionEntry);
        console.log('Action logged:', actionType, data);
    }
    
    logGameState() {
        const now = Date.now();
        if (now - this.lastLogTime < this.logInterval) return;
        
        const stateEntry = {
            timestamp: now,
            gameTime: this.gameTime,
            sessionId: this.gameSession.sessionId,
            level: this.level,
            dayCount: this.dayCount,
            robot: this.robot ? {
                position: { x: this.robot.pos.x, y: this.robot.pos.y },
                angle: this.robot.angle,
                batteryLevel: this.batteryLevel,
                isCharging: this.isCharging,
                carriedItem: this.robot.carriedItem ? this.robot.carriedItem.type : null
            } : null,
            environment: {
                plantsHealth: this.plants ? this.plants.map(p => ({ type: p.type, health: p.health })) : [],
                animalsCount: this.animals ? this.animals.length : 0,
                trashCount: this.trash ? this.trash.length : 0,
                weedCount: this.weeds ? this.weeds.length : 0,
                grassNeedsMowing: this.grassAreas ? this.grassAreas.some(area => area.needsMowing) : false
            },
            tasks: {
                active: this.activeTasks ? this.activeTasks.length : 0,
                completed: this.completedTasks ? this.completedTasks.length : 0
            }
        };
        
        this.stateLog.push(stateEntry);
        this.lastLogTime = now;
    }
    
    updateUI() {
        document.getElementById('score').textContent = this.score;
        document.getElementById('level').textContent = this.level;
        
        // Update battery indicator with color coding
        const batteryEl = document.getElementById('battery');
        if (batteryEl) {
            batteryEl.textContent = Math.round(this.batteryLevel) + '%';
            
            // Color code battery level
            if (this.batteryLevel > 50) {
                batteryEl.style.color = '#4CAF50'; // Green
            } else if (this.batteryLevel > 20) {
                batteryEl.style.color = '#FF9800'; // Orange
            } else {
                batteryEl.style.color = '#F44336'; // Red
            }
        }
        
        // Update water level indicator
        const waterEl = document.getElementById('water');
        if (waterEl) {
            waterEl.textContent = Math.round(this.waterLevel) + '%';
            
            // Color code water level
            if (this.waterLevel > 50) {
                waterEl.style.color = '#2196F3'; // Blue
            } else if (this.waterLevel > 20) {
                waterEl.style.color = '#FF9800'; // Orange
            } else {
                waterEl.style.color = '#F44336'; // Red
            }
        }
    }
    
    exportLogs() {
        const exportData = {
            session: this.gameSession,
            actions: this.actionLog,
            states: this.stateLog,
            photos: this.photosTaken,
            exportTime: Date.now()
        };
        
        const dataStr = JSON.stringify(exportData, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});
        const url = URL.createObjectURL(dataBlob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `farm_robot_training_data_${this.gameSession.sessionId}.json`;
        link.click();
        
        URL.revokeObjectURL(url);
    }
    
    // Modal handlers
    nextLevel() {
        this.level++;
        this.gameRunning = true;
        document.getElementById('completionModal').style.display = 'none';
        this.initializeLevel();
    }
    
    restartLevel() {
        this.gameRunning = true;
        document.getElementById('gameOverModal').style.display = 'none';
        this.initializeLevel();
    }
    
    restartGame() {
        this.level = 1;
        this.score = 0;
        this.dayCount = 1;
        this.batteryLevel = 100;
        this.gameRunning = true;
        document.getElementById('gameOverModal').style.display = 'none';
        this.initializeLevel();
    }
}

// Supporting classes for game systems
class PlantGrowthSystem {
    update(plants, deltaTime) {
        plants.forEach(plant => {
            plant.growth += 0.01 * deltaTime;
            if (plant.needsWater && Math.random() < 0.001) {
                plant.needsWater = false;
            }
        });
    }
}

class AnimalSpawnSystem {
    constructor() {
        this.lastSpawnTime = 0;
        this.spawnInterval = 30000; // 30 seconds
    }
    
    update(animals, plants) {
        const now = Date.now();
        if (now - this.lastSpawnTime > this.spawnInterval && animals.length < 3) {
            // Spawn logic would go here
            this.lastSpawnTime = now;
        }
    }
}

class WeatherSystem {
    constructor() {
        this.currentWeather = 'sunny';
        this.temperature = 72;
    }
    
    update() {
        // Weather affects plant growth and robot efficiency
    }
}

// Start the game when page loads
window.addEventListener('load', () => {
    window.game = new Game();
}); 