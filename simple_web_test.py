#!/usr/bin/env python3
"""
Simple test to verify web agent can connect to the game
"""

import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import chromedriver_autoinstaller

def simple_connection_test():
    """Test basic connection to the web game"""
    print("🧪 Simple Web Connection Test")
    print("=" * 30)
    
    # Auto-install chromedriver
    chromedriver_autoinstaller.install()
    
    driver = None
    try:
        # Setup Chrome
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1200,800")
        
        print("🌐 Starting Chrome...")
        driver = webdriver.Chrome(options=chrome_options)
        
        print("📡 Connecting to game...")
        driver.get("http://localhost:8000")
        
        print("⏳ Waiting for game canvas...")
        canvas = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "gameCanvas"))
        )
        print("✅ Canvas found!")
        
        print("⏳ Waiting for game object...")
        game_ready = WebDriverWait(driver, 10).until(
            lambda d: d.execute_script("return typeof window.game !== 'undefined'")
        )
        print("✅ Game object ready!")
        
        # Test basic game state extraction
        print("📊 Testing game state...")
        state = driver.execute_script("""
            return {
                hasGame: typeof window.game !== 'undefined',
                hasRobot: window.game && window.game.robot ? true : false,
                robotPos: window.game && window.game.robot ? 
                    [window.game.robot.pos.x, window.game.robot.pos.y] : null
            };
        """)
        
        print(f"   Game loaded: {state['hasGame']}")
        print(f"   Robot exists: {state['hasRobot']}")
        print(f"   Robot position: {state['robotPos']}")
        
        if state['hasGame'] and state['hasRobot']:
            print("✅ Everything looks good!")
            
            # Test a simple key press
            print("🎮 Testing key press...")
            driver.execute_script("""
                var event = new KeyboardEvent('keydown', {
                    code: 'ArrowLeft',
                    key: 'ArrowLeft',
                    bubbles: true
                });
                document.dispatchEvent(event);
                if (window.game && window.game.keys) {
                    window.game.keys['ArrowLeft'] = true;
                }
            """)
            
            time.sleep(0.5)
            
            driver.execute_script("""
                var event = new KeyboardEvent('keyup', {
                    code: 'ArrowLeft', 
                    key: 'ArrowLeft',
                    bubbles: true
                });
                document.dispatchEvent(event);
                if (window.game && window.game.keys) {
                    window.game.keys['ArrowLeft'] = false;
                }
            """)
            
            print("✅ Key press test completed!")
            return True
        else:
            print("❌ Game not properly loaded")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        if driver:
            print("🧹 Closing browser...")
            driver.quit()

if __name__ == "__main__":
    success = simple_connection_test()
    if success:
        print("\n🎉 Connection test passed! Web agent should work now.")
        print("Try running: python test_web_agent.py")
    else:
        print("\n❌ Connection test failed. Check the error messages above.") 