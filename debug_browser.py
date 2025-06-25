#!/usr/bin/env python3
"""
Debug script to see what's happening in the browser
"""

import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import chromedriver_autoinstaller

def debug_browser():
    """Open browser and debug what's happening"""
    print("🔍 Browser Debug Session")
    print("=" * 30)
    
    # Auto-install chromedriver
    chromedriver_autoinstaller.install()
    
    # Setup Chrome with debugging
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1200,800")
    # Don't run headless so we can see what's happening
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        print("🌐 Loading game...")
        driver.get("http://localhost:8000")
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "gameCanvas"))
        )
        print("✅ Canvas element found")
        
        # Check for JavaScript errors
        print("\n📜 Checking for JavaScript errors...")
        logs = driver.get_log('browser')
        if logs:
            print("Found browser console messages:")
            for log in logs:
                print(f"   {log['level']}: {log['message']}")
        else:
            print("   No console errors found")
        
        # Check game object status every second for 10 seconds
        print("\n🎮 Monitoring game initialization...")
        for i in range(10):
            try:
                status = driver.execute_script("""
                    return {
                        windowGame: typeof window.game,
                        gameExists: window.game ? true : false,
                        gameConstructor: typeof Game,
                        documentReady: document.readyState,
                        canvasFound: document.getElementById('gameCanvas') ? true : false,
                        scriptsLoaded: {
                            gameJs: Array.from(document.scripts).some(s => s.src.includes('game.js'))
                        }
                    };
                """)
                print(f"   Second {i+1}: Game={status['windowGame']}, Ready={status['documentReady']}, Canvas={status['canvasFound']}")
                
                if status['gameExists']:
                    print("✅ Game object found!")
                    
                    # Get more details about the game
                    game_details = driver.execute_script("""
                        return {
                            hasRobot: window.game.robot ? true : false,
                            hasRooms: window.game.rooms ? window.game.rooms.length : 0,
                            hasObjects: window.game.objects ? window.game.objects.length : 0,
                            gameRunning: window.game.gameRunning,
                            robotPos: window.game.robot ? [window.game.robot.pos.x, window.game.robot.pos.y] : null
                        };
                    """)
                    
                    print(f"   Game details: {game_details}")
                    break
                    
                time.sleep(1)
            except Exception as e:
                print(f"   Second {i+1}: Error checking game status: {e}")
                time.sleep(1)
        
        print(f"\n🌍 Current URL: {driver.current_url}")
        print(f"📜 Page title: {driver.title}")
        
        # Check if we can see the game visually
        print("\n📸 Taking screenshot...")
        driver.save_screenshot("debug_screenshot.png")
        print("   Screenshot saved as debug_screenshot.png")
        
        input("\n⏸️  Browser is open. Check the game manually and press Enter to continue...")
        
    finally:
        driver.quit()
        print("🧹 Browser closed")

if __name__ == "__main__":
    debug_browser() 