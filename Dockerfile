# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy application files
COPY index.html .
COPY style.css .
COPY game.js .
COPY server.py .
COPY analyze_training_data.py .
COPY requirements.txt .
COPY README.md .

# Install Python dependencies for analysis
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for volume mounts with proper permissions
RUN mkdir -p /app/training_logs /app/data

# Create non-root user for security with specific UID/GID that matches host user
RUN groupadd -g 1000 robotapp && useradd -u 1000 -g robotapp -m robotapp

# Change ownership of app directory and volume mount points
RUN chown -R robotapp:robotapp /app

# Ensure the directories are writable
RUN chmod -R 755 /app/training_logs /app/data

# Switch to non-root user
USER robotapp

# Expose port 8000
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000')" || exit 1

# Run the server
CMD ["python", "server.py"] 