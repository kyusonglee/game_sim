#!/usr/bin/env python3
"""
Test script to verify automatic server startup functionality
"""

import sys
import time
from run_agents import check_server_running, ensure_server_running

def test_server_autostart():
    """Test the automatic server startup"""
    print("🧪 Testing Automatic Server Startup")
    print("=" * 50)
    
    game_url = "http://localhost:8000"
    
    # Check initial state
    print(f"1. Checking if server is already running at {game_url}...")
    if check_server_running(game_url):
        print("✅ Server is already running")
    else:
        print("❌ Server is not running")
    
    # Test ensure_server_running function
    print(f"\n2. Testing ensure_server_running()...")
    try:
        actual_url = ensure_server_running(game_url)
        print(f"✅ Function returned: {actual_url}")
        
        # Verify server is now running
        print(f"\n3. Verifying server is accessible...")
        if check_server_running(actual_url):
            print("✅ Server is now accessible!")
            
            # Test basic HTTP request
            import urllib.request
            try:
                response = urllib.request.urlopen(actual_url, timeout=5)
                print(f"✅ HTTP response code: {response.code}")
                print("✅ Server auto-start test PASSED!")
            except Exception as e:
                print(f"❌ HTTP request failed: {e}")
                
        else:
            print("❌ Server is still not accessible")
            
    except Exception as e:
        print(f"❌ ensure_server_running() failed: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Test completed")

if __name__ == "__main__":
    test_server_autostart() 