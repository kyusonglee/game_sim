#!/usr/bin/env python3
"""
Enhanced web server for Robot Simulator with logging capabilities.
Serves the game and handles training data collection.
"""

import http.server
import socketserver
import webbrowser
import os
import sys
import json
import urllib.parse
import socket
import time
import base64
from pathlib import Path
from datetime import datetime

# Configuration
DEFAULT_PORT = 8000
HOST = '0.0.0.0'  # Bind to all interfaces for Docker compatibility

def find_available_port(start_port=DEFAULT_PORT, max_attempts=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((HOST, port))
                return port
        except OSError:
            continue
    raise OSError(f"No available ports found in range {start_port}-{start_port + max_attempts - 1}")

class GameHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=Path(__file__).parent, **kwargs)
    
    def end_headers(self):
        # Add CORS headers for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(200)
        self.end_headers()
    
    def do_POST(self):
        """Handle POST requests for logging endpoints"""
        if self.path == '/api/log':
            self.handle_log_entry()
        elif self.path == '/api/log/batch':
            self.handle_batch_logs()
        elif self.path == '/api/log/export':
            self.handle_export_request()
        else:
            self.send_error(404, "Endpoint not found")
    
    def handle_log_entry(self):
        """Handle individual log entry"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            log_data = json.loads(post_data.decode('utf-8'))
            
            # Save log entry
            self.save_log_entry(log_data)
            
            # Send success response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = json.dumps({"status": "success", "message": "Log entry saved"})
            self.wfile.write(response.encode('utf-8'))
            
        except Exception as e:
            print(f"Error processing log entry: {e}")
            self.send_error(500, f"Error processing log entry: {str(e)}")
    
    def handle_batch_logs(self):
        """Handle batch log entries"""
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            batch_data = json.loads(post_data.decode('utf-8'))
            
            # Save batch logs
            self.save_batch_logs(batch_data)
            
            # Send success response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = json.dumps({"status": "success", "message": "Batch logs saved"})
            self.wfile.write(response.encode('utf-8'))
            
        except Exception as e:
            print(f"Error processing batch logs: {e}")
            self.send_error(500, f"Error processing batch logs: {str(e)}")
    
    def handle_export_request(self):
        """Handle training data export request"""
        try:
            # Get query parameters
            query_components = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            session_id = query_components.get('session_id', [None])[0]
            include_images = query_components.get('include_images', ['false'])[0].lower() == 'true'
            
            # Export training data
            export_data = self.export_training_data(session_id, include_images)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Disposition', f'attachment; filename="training_data_{session_id or "all"}.json"')
            self.end_headers()
            self.wfile.write(json.dumps(export_data, indent=2).encode('utf-8'))
            
        except Exception as e:
            print(f"Error exporting training data: {e}")
            self.send_error(500, f"Error exporting training data: {str(e)}")
    
    def save_screenshot(self, screenshot_data, session_id, log_type, timestamp):
        """Save screenshot as separate PNG file"""
        if not screenshot_data or not screenshot_data.startswith('data:image/png;base64,'):
            return None
        
        try:
            # Remove data URL prefix
            base64_data = screenshot_data.split(',')[1]
            image_data = base64.b64decode(base64_data)
            
            # Create screenshots directory
            screenshots_dir = Path("training_logs") / session_id / "screenshots"
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            
            # Save with timestamp and type
            filename = f"{log_type}_{timestamp}.png"
            screenshot_path = screenshots_dir / filename
            
            with open(screenshot_path, 'wb') as f:
                f.write(image_data)
            
            return str(screenshot_path.relative_to(Path("training_logs")))
            
        except Exception as e:
            print(f"Failed to save screenshot: {e}")
            return None
    
    def save_log_entry(self, log_data):
        """Save individual log entry to file"""
        logs_dir = Path("training_logs")
        logs_dir.mkdir(exist_ok=True)
        
        log_type = log_data.get('type', 'unknown')
        session_id = log_data.get('data', {}).get('sessionId', 'unknown')
        timestamp = log_data.get('data', {}).get('timestamp', int(datetime.now().timestamp() * 1000))
        
        # Create session directory
        session_dir = logs_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Extract and save screenshot separately
        log_entry = log_data['data'].copy()
        screenshot_path = None
        
        if 'screenshot' in log_entry and log_entry['screenshot']:
            screenshot_path = self.save_screenshot(
                log_entry['screenshot'], 
                session_id, 
                log_type, 
                timestamp
            )
            # Replace screenshot data with file path
            log_entry['screenshot'] = screenshot_path
        
        # Save action logs and state logs separately
        if log_type == 'action':
            log_file = session_dir / 'actions.jsonl'
        else:
            log_file = session_dir / 'states.jsonl'
        
        # Append log entry (JSONL format for easy streaming)
        with open(log_file, 'a', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'type': log_type,
                **log_entry
            }, f)
            f.write('\n')
    
    def save_batch_logs(self, batch_data):
        """Save batch of log entries"""
        logs_dir = Path("training_logs")
        logs_dir.mkdir(exist_ok=True)
        
        session_id = batch_data.get('sessionId', 'unknown')
        session_dir = logs_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Save batch to states file
        log_file = session_dir / 'states.jsonl'
        
        with open(log_file, 'a', encoding='utf-8') as f:
            for entry in batch_data.get('data', []):
                # Handle screenshots in batch logs
                if 'screenshot' in entry and entry['screenshot']:
                    timestamp = entry.get('timestamp', int(datetime.now().timestamp() * 1000))
                    screenshot_path = self.save_screenshot(
                        entry['screenshot'], 
                        session_id, 
                        'state', 
                        timestamp
                    )
                    entry['screenshot'] = screenshot_path
                
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'state',
                    **entry
                }, f)
                f.write('\n')
    
    def export_training_data(self, session_id=None, include_images=False):
        """Export training data for specified session or all sessions"""
        logs_dir = Path("training_logs")
        
        if not logs_dir.exists():
            return {"sessions": [], "total_entries": 0}
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "include_images": include_images,
            "sessions": [],
            "total_entries": 0
        }
        
        # Get session directories
        if session_id:
            session_dirs = [logs_dir / session_id] if (logs_dir / session_id).exists() else []
        else:
            session_dirs = [d for d in logs_dir.iterdir() if d.is_dir()]
        
        for session_dir in session_dirs:
            session_data = {
                "session_id": session_dir.name,
                "actions": [],
                "states": [],
                "screenshots_count": 0
            }
            
            # Count screenshots
            screenshots_dir = session_dir / "screenshots"
            if screenshots_dir.exists():
                session_data["screenshots_count"] = len(list(screenshots_dir.glob("*.png")))
            
            # Load actions
            actions_file = session_dir / 'actions.jsonl'
            if actions_file.exists():
                with open(actions_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            action = json.loads(line)
                            
                            # Optionally include image data
                            if include_images and action.get('screenshot'):
                                screenshot_path = logs_dir / action['screenshot']
                                if screenshot_path.exists():
                                    with open(screenshot_path, 'rb') as img_f:
                                        img_data = base64.b64encode(img_f.read()).decode('utf-8')
                                        action['screenshot_data'] = f"data:image/png;base64,{img_data}"
                            
                            session_data["actions"].append(action)
            
            # Load states
            states_file = session_dir / 'states.jsonl'
            if states_file.exists():
                with open(states_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            state = json.loads(line)
                            
                            # Optionally include image data
                            if include_images and state.get('screenshot'):
                                screenshot_path = logs_dir / state['screenshot']
                                if screenshot_path.exists():
                                    with open(screenshot_path, 'rb') as img_f:
                                        img_data = base64.b64encode(img_f.read()).decode('utf-8')
                                        state['screenshot_data'] = f"data:image/png;base64,{img_data}"
                            
                            session_data["states"].append(state)
            
            # Add session metadata
            session_data["action_count"] = len(session_data["actions"])
            session_data["state_count"] = len(session_data["states"])
            session_data["total_entries"] = session_data["action_count"] + session_data["state_count"]
            
            export_data["sessions"].append(session_data)
            export_data["total_entries"] += session_data["total_entries"]
        
        return export_data

class ReusableTCPServer(socketserver.TCPServer):
    """TCP Server that allows port reuse"""
    def __init__(self, server_address, RequestHandlerClass, bind_and_activate=True):
        super().__init__(server_address, RequestHandlerClass, bind_and_activate=False)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if bind_and_activate:
            try:
                self.server_bind()
                self.server_activate()
            except:
                self.server_close()
                raise

def main():
    # Check if required files exist
    required_files = ['index.html', 'style.css', 'game.js']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("Please make sure all game files are in the same directory as this server script.")
        sys.exit(1)
    
    # Create training logs directory
    logs_dir = Path("training_logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Find available port
    try:
        port = find_available_port(DEFAULT_PORT)
        if port != DEFAULT_PORT:
            print(f"⚠️  Port {DEFAULT_PORT} was busy, using port {port} instead")
    except OSError as e:
        print(f"❌ Could not find available port: {e}")
        print("💡 Try one of these solutions:")
        print("   1. Wait a few seconds and try again")
        print("   2. Kill processes using port 8000: lsof -ti:8000 | xargs kill -9")
        print("   3. Use a different port by editing the DEFAULT_PORT in this script")
        sys.exit(1)
    
    # Start the server
    try:
        with ReusableTCPServer((HOST, port), GameHandler) as httpd:
            server_url = f"http://{HOST}:{port}"
            
            print("🚀 Robot Simulator Web Server with Training Data Collection")
            print("=" * 60)
            print(f"🌐 Server running at: {server_url}")
            print(f"📁 Serving files from: {Path(__file__).parent.absolute()}")
            print(f"📊 Training logs stored in: {logs_dir.absolute()}")
            print("🎮 Open the URL above in your web browser to play!")
            print("📈 Training data will be automatically collected during gameplay")
            print("⏹️  Press Ctrl+C to stop the server")
            print("=" * 60)
            print("\n🔧 API Endpoints:")
            print(f"   POST {server_url}/api/log - Individual log entries")
            print(f"   POST {server_url}/api/log/batch - Batch log entries")
            print(f"   GET  {server_url}/api/log/export - Export training data")
            print(f"   GET  {server_url}/api/log/export?session_id=<id> - Export specific session")
            print(f"   GET  {server_url}/api/log/export?include_images=true - Include screenshots")
            print("\n💡 In-Game Features:")
            print("   • Press 'E' in game to export current session data")
            print("   • Screenshots automatically captured for all actions")
            print("   • Object positions and environment state logged")
            print("   • Room layouts and furniture positions recorded")
            print("\n📊 Data Analysis:")
            print("   python analyze_training_data.py --export-visual")
            print("   python analyze_training_data.py --export-ml")
            print("   python analyze_training_data.py --generate-report")
            print("\n🔧 Troubleshooting:")
            print("   • If port issues persist, run: lsof -ti:8000 | xargs kill -9")
            print("   • Server automatically finds available ports")
            print("   • Large screenshots may slow down logging")
            print("=" * 60)
            
            # Try to open browser automatically
            try:
                webbrowser.open(server_url)
                print("🌟 Attempting to open game in your default web browser...")
            except Exception as e:
                print(f"⚠️  Could not auto-open browser: {e}")
                print(f"Please manually open {server_url} in your browser.")
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        print(f"📊 Training data saved in: {logs_dir.absolute()}")
    except OSError as e:
        if e.errno == 48 or "Address already in use" in str(e):
            print(f"❌ Port {port} is still in use!")
            print("💡 Solutions:")
            print(f"   1. Wait 30-60 seconds and try again")
            print(f"   2. Kill processes: lsof -ti:{port} | xargs kill -9")
            print(f"   3. Restart your terminal")
            print(f"   4. Use a different port by changing DEFAULT_PORT in server.py")
        else:
            print(f"❌ Server error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 