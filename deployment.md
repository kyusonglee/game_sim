# Robot Simulator - Deployment Guide

## Docker Deployment

### Prerequisites
- Docker installed on your system
- Docker Compose (optional, for easier management)

### Quick Start

#### Option 1: Using Docker Compose (Recommended)

```bash
# Build and start the application
docker-compose up --build

# Run in detached mode (background)
docker-compose up -d --build

# Stop the application
docker-compose down
```

#### Option 2: Using Docker directly

```bash
# Build the Docker image
docker build -t robot-simulator .

# Run the container
docker run -p 8000:8000 --name robot-simulator-app robot-simulator

# Run in detached mode (background)
docker run -d -p 8000:8000 --name robot-simulator-app robot-simulator

# Stop and remove the container
docker stop robot-simulator-app
docker rm robot-simulator-app
```

### Access the Application

Once deployed, access the Robot Simulator at:
- **Local**: http://localhost:8000
- **Container Health**: Check with `docker ps` or `docker-compose ps`

### Configuration

#### Environment Variables
- `PORT`: Server port (default: 8000)
- `HOST`: Server host (default: localhost)

#### Port Mapping
- The container exposes port 8000
- Map to any available host port: `-p <host-port>:8000`

### Production Deployment

#### Using Docker Compose with environment file

1. Create `.env` file:
```bash
PORT=8000
COMPOSE_PROJECT_NAME=robot-simulator
```

2. Deploy:
```bash
docker-compose -f docker-compose.yml up -d
```

#### Using Docker Swarm

```bash
# Initialize swarm (if not already done)
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml robot-simulator-stack

# Check services
docker service ls

# Remove stack
docker stack rm robot-simulator-stack
```

#### Using Kubernetes

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: robot-simulator
spec:
  replicas: 2
  selector:
    matchLabels:
      app: robot-simulator
  template:
    metadata:
      labels:
        app: robot-simulator
    spec:
      containers:
      - name: robot-simulator
        image: robot-simulator:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            memory: "128Mi"
            cpu: "100m"
---
apiVersion: v1
kind: Service
metadata:
  name: robot-simulator-service
spec:
  selector:
    app: robot-simulator
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy to Kubernetes:
```bash
kubectl apply -f k8s-deployment.yaml
```

### Monitoring and Maintenance

#### Health Checks
The container includes built-in health checks:
```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' robot-simulator-app

# View health check logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' robot-simulator-app
```

#### Logs
```bash
# View logs (Docker)
docker logs robot-simulator-app

# Follow logs (Docker)
docker logs -f robot-simulator-app

# View logs (Docker Compose)
docker-compose logs robot-simulator

# Follow logs (Docker Compose)
docker-compose logs -f robot-simulator
```

#### Updates
```bash
# Rebuild and restart (Docker Compose)
docker-compose up --build -d

# Update with new image (Docker)
docker pull robot-simulator:latest
docker stop robot-simulator-app
docker rm robot-simulator-app
docker run -d -p 8000:8000 --name robot-simulator-app robot-simulator:latest
```

### Troubleshooting

#### Common Issues

1. **Port already in use**
   ```bash
   # Check what's using port 8000
   lsof -i :8000
   
   # Use different port
   docker run -p 8080:8000 robot-simulator
   ```

2. **Container won't start**
   ```bash
   # Check logs
   docker logs robot-simulator-app
   
   # Run interactively for debugging
   docker run -it --rm robot-simulator /bin/bash
   ```

3. **Health check failing**
   ```bash
   # Check if server is responding
   docker exec robot-simulator-app curl -f http://localhost:8000 || echo "Failed"
   ```

### Security Considerations

- Container runs as non-root user (`robotapp`)
- No sensitive data stored in image
- Health checks ensure service availability
- CORS headers configured for local development

### Performance Tuning

For production environments:

```yaml
# docker-compose.yml additions
services:
  robot-simulator:
    # ... existing config
    deploy:
      resources:
        limits:
          cpus: '0.50'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
    ulimits:
      nofile:
        soft: 65536
        hard: 65536
``` 