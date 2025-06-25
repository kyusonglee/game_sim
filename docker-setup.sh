#!/bin/bash

# Robot Simulator Docker Setup Script
# Creates necessary directories and sets up permissions for volume mounts

echo "🐋 Setting up Robot Simulator with Docker..."
echo "=" * 50

# Create local directories for volume mounts
echo "📁 Creating local directories..."
mkdir -p training_logs
mkdir -p data
mkdir -p data/exports
mkdir -p data/screenshots

# Set proper permissions (readable/writable by all users)
echo "🔒 Setting permissions..."
chmod 755 training_logs
chmod 755 data
chmod 755 data/exports
chmod 755 data/screenshots

# Create .gitignore for data directories if it doesn't exist
if [ ! -f .gitignore ]; then
    echo "📝 Creating .gitignore..."
    cat > .gitignore << EOF
# Training data and screenshots (too large for git)
training_logs/
data/
*.log
*.png
*.jpg
*.json

# Python cache
__pycache__/
*.pyc
*.pyo

# Docker volumes
.docker/
EOF
else
    echo "📝 .gitignore already exists"
fi

echo ""
echo "✅ Setup complete! Directory structure:"
echo "   📊 ./training_logs/     - Game session logs and data"
echo "   📸 ./data/              - Screenshots and exports"
echo "   🐋 Ready for Docker!"
echo ""
echo "🚀 To start the simulator:"
echo "   docker-compose up --build"
echo ""
echo "🔧 To view logs:"
echo "   docker-compose logs -f robot-simulator"
echo ""
echo "🛑 To stop:"
echo "   docker-compose down"
echo ""
echo "📊 To analyze training data:"
echo "   docker-compose exec robot-simulator python analyze_training_data.py --export-visual"
echo "" 