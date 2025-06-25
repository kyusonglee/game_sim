#!/bin/bash

# Robot Simulator Server Startup Script
# Handles port conflicts and provides helpful troubleshooting

echo "🚀 Starting Robot Simulator Server..."

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Python not found! Please install Python 3.6+ to run the server."
    exit 1
fi

# Use python3 if available, otherwise use python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# Function to kill processes on port 8000
kill_port_8000() {
    echo "🔧 Checking for processes using port 8000..."
    PIDS=$(lsof -ti:8000 2>/dev/null)
    if [ ! -z "$PIDS" ]; then
        echo "⚠️  Found processes using port 8000: $PIDS"
        echo "🔨 Killing processes..."
        echo $PIDS | xargs kill -9 2>/dev/null
        sleep 2
        echo "✅ Processes killed. Waiting for port to be released..."
        sleep 3
    else
        echo "✅ Port 8000 is available"
    fi
}

# Function to check if server files exist
check_files() {
    local missing_files=()
    
    for file in "index.html" "style.css" "game.js" "server.py"; do
        if [ ! -f "$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -ne 0 ]; then
        echo "❌ Missing required files: ${missing_files[*]}"
        echo "Please make sure you're in the correct directory with all game files."
        exit 1
    fi
}

# Main execution
echo "📁 Current directory: $(pwd)"

# Check for required files
check_files

# Handle command line arguments
case "${1:-}" in
    "--kill-port"|"-k")
        kill_port_8000
        exit 0
        ;;
    "--force"|"-f")
        echo "🔧 Force mode: Killing any processes on port 8000..."
        kill_port_8000
        ;;
    "--help"|"-h")
        echo "Robot Simulator Server Startup Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  -k, --kill-port    Kill processes using port 8000 and exit"
        echo "  -f, --force        Kill processes on port 8000 before starting server"
        echo "  -h, --help         Show this help message"
        echo ""
        echo "The server will automatically find an available port if 8000 is busy."
        exit 0
        ;;
esac

# Create training logs directory
mkdir -p training_logs

echo "🎮 Starting Robot Simulator Server..."
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

# Start the server
$PYTHON_CMD server.py

echo ""
echo "👋 Server shutdown complete!"

# Check if training logs were created
if [ -d "training_logs" ] && [ "$(ls -A training_logs)" ]; then
    echo "📊 Training data was collected in: $(pwd)/training_logs"
    echo "💡 Run 'python analyze_training_data.py' to analyze the data"
fi 