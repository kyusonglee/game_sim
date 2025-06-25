#!/usr/bin/env python3
"""
Debug script to help troubleshoot web agent connection issues.
"""

import time
import requests
import subprocess
import threading
import socket
from pathlib import Path

def check_port_availability(port):
    """Check if a port is available or in use"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('localhost', port))
            return result != 0  # True if port is available (not in use)
    except:
        return True

def check_web_server():
    """Check if the web server is running and accessible"""
    print("🌐 Checking web server status...")
    
    # Check if port 8000 is in use
    if check_port_availability(8000):
        print("❌ Port 8000 is not in use - web server is not running")
        return False
    else:
        print("✅ Port 8000 is in use - something is running")
    
    # Try to access the web server
    try:
        response = requests.get("http://localhost:8000", timeout=5)
        if response.status_code == 200:
            print("✅ Web server is accessible")
            
            # Check if it contains the game
            if 'gameCanvas' in response.text:
                print("✅ Game canvas found in HTML")
                return True
            else:
                print("❌ Game canvas not found in HTML")
                return False
        else:
            print(f"❌ Web server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to web server")
        return False
    except requests.exceptions.Timeout:
        print("❌ Web server connection timed out")
        return False
    except Exception as e:
        print(f"❌ Error checking web server: {e}")
        return False

def check_required_files():
    """Check if all required files exist"""
    print("\n📁 Checking required files...")
    
    required_files = [
        'server.py',
        'index.html',
        'game.js',
        'style.css'
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            all_exist = False
    
    return all_exist

def start_web_server():
    """Try to start the web server"""
    print("\n🚀 Attempting to start web server...")
    
    try:
        # Try to start the server
        process = subprocess.Popen(
            ['python', 'server.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it a moment to start
        time.sleep(3)
        
        # Check if it's still running
        if process.poll() is None:
            print("✅ Web server started successfully")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Web server failed to start")
            print(f"   stdout: {stdout}")
            print(f"   stderr: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting web server: {e}")
        return None

def test_browser_setup():
    """Test if Chrome/chromedriver is properly set up"""
    print("\n🌐 Testing browser setup...")
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.get("https://www.google.com")
        title = driver.title
        driver.quit()
        
        print("✅ Chrome/chromedriver working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Browser setup issue: {e}")
        print("💡 Try: pip install chromedriver-autoinstaller")
        return False

def comprehensive_test():
    """Run comprehensive diagnostics"""
    print("🔍 Running comprehensive diagnostics...")
    print("=" * 50)
    
    # Check 1: Required files
    files_ok = check_required_files()
    
    # Check 2: Web server status
    server_running = check_web_server()
    
    # Check 3: Browser setup
    browser_ok = test_browser_setup()
    
    # Summary
    print("\n📊 Diagnostic Summary:")
    print(f"   Required files: {'✅' if files_ok else '❌'}")
    print(f"   Web server: {'✅' if server_running else '❌'}")
    print(f"   Browser setup: {'✅' if browser_ok else '❌'}")
    
    if not server_running and files_ok:
        print("\n🚀 Attempting to start web server...")
        server_process = start_web_server()
        
        if server_process:
            print("⏳ Waiting for server to be ready...")
            time.sleep(3)
            
            # Re-check server
            server_running = check_web_server()
            if server_running:
                print("✅ Web server is now running!")
                return server_process
            else:
                print("❌ Server started but not accessible")
                server_process.terminate()
                return None
    
    return None

def manual_setup_guide():
    """Print manual setup instructions"""
    print("\n📋 Manual Setup Guide:")
    print("=" * 30)
    
    print("\n1. 🗂️  Check you're in the right directory:")
    print("   ls -la")
    print("   # Should see: server.py, index.html, game.js, style.css")
    
    print("\n2. 🐍 Install required packages:")
    print("   pip install selenium chromedriver-autoinstaller opencv-python")
    
    print("\n3. 🌐 Start web server manually:")
    print("   python server.py")
    print("   # Should see: 'Server running at http://localhost:8000'")
    
    print("\n4. 🌍 Test in browser:")
    print("   Open http://localhost:8000 in Chrome")
    print("   # Should see the robot game interface")
    
    print("\n5. 🤖 Test web agent:")
    print("   python test_web_agent.py")

def quick_fix():
    """Try to automatically fix common issues"""
    print("\n🔧 Attempting quick fixes...")
    
    # Try to install missing dependencies
    try:
        import subprocess
        subprocess.check_call(['pip', 'install', 'chromedriver-autoinstaller'], 
                            capture_output=True)
        print("✅ Installed chromedriver-autoinstaller")
    except:
        print("❌ Failed to install chromedriver-autoinstaller")
    
    # Try to start server if files exist
    if check_required_files():
        server_process = start_web_server()
        return server_process
    
    return None

def main():
    """Main diagnostic function"""
    print("🔍 Web Agent Diagnostic Tool")
    print("=" * 40)
    
    # Run comprehensive test
    server_process = comprehensive_test()
    
    if server_process:
        print(f"\n🎉 Setup successful! Server running with PID {server_process.pid}")
        print("\nNow try running the web agent:")
        print("   python test_web_agent.py")
        print("   python run_agent.py web-agent")
        
        input("\nPress Enter to stop the server...")
        server_process.terminate()
        print("🛑 Server stopped")
    else:
        print("\n❌ Setup failed. Here's what to do:")
        manual_setup_guide()
        
        response = input("\n🔧 Try automatic fix? (y/n): ").lower()
        if response == 'y':
            quick_fix()

if __name__ == "__main__":
    main() 