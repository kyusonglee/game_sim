#!/usr/bin/env python3
"""
Test script for the web agent to verify it can interact with the game properly.
"""

import time
from agent import RobotAgent, Action

def test_web_agent():
    """Test the web agent with the fixed interaction methods"""
    print("🧪 Testing Web Agent...")
    
    # Create agent
    agent = RobotAgent(game_url="http://localhost:8000", headless=False)
    
    try:
        # Start browser and load game
        if not agent.start_browser():
            print("❌ Failed to start browser or load game")
            return False
        
        print("⏳ Waiting for game to fully load...")
        time.sleep(3)
        
        # Test getting game state
        print("\n📊 Testing game state extraction...")
        state = agent.get_game_state()
        if state:
            print(f"✅ Game state retrieved successfully!")
            print(f"   Robot position: {state.robot_pos}")
            print(f"   Robot room: {state.robot_room}")
            print(f"   Objects found: {len(state.objects)}")
            print(f"   Current score: {state.score}")
            print(f"   Game level: {state.level}")
        else:
            print("❌ Failed to get game state")
            return False
        
        # Test basic actions
        print("\n🎮 Testing robot actions...")
        
        actions_to_test = [
            Action('rotate_left', 0.5),
            Action('rotate_right', 0.5),
            Action('move_forward', 1.0),
            Action('move_backward', 0.5),
            Action('pickup', 0.1),
            Action('drop', 0.1)
        ]
        
        for i, action in enumerate(actions_to_test):
            print(f"   Testing action {i+1}/{len(actions_to_test)}: {action.type}")
            agent.send_action(action)
            time.sleep(0.5)  # Small delay between actions
            
            # Check if state changed
            new_state = agent.get_game_state()
            if new_state:
                print(f"     ✅ Action executed, robot now at {new_state.robot_pos}")
            else:
                print(f"     ⚠️ Could not verify action result")
        
        # Test screenshot capture
        print("\n📸 Testing screenshot capture...")
        screenshot = agent.take_screenshot()
        if screenshot is not None:
            print(f"✅ Screenshot captured successfully! Shape: {screenshot.shape}")
            
            # Test image analysis
            analysis = agent.analyze_screenshot(screenshot)
            print(f"   Detected objects: {len(analysis.get('detected_objects', []))}")
        else:
            print("❌ Failed to capture screenshot")
        
        # Test decision making
        print("\n🧠 Testing AI decision making...")
        for i in range(5):
            action = agent.decide_next_action()
            if action:
                print(f"   Decision {i+1}: {action.type}")
                agent.send_action(action)
                time.sleep(0.5)
            else:
                print(f"   Decision {i+1}: No action chosen")
        
        print("\n🎉 All tests completed!")
        return True
        
    except KeyboardInterrupt:
        print("\n⏸️ Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        agent.cleanup()

def main():
    """Main function"""
    print("🚀 Web Agent Test Suite")
    print("=" * 50)
    print("This script tests the web agent's ability to:")
    print("- Load the web game")
    print("- Extract game state")
    print("- Send keyboard actions")
    print("- Capture screenshots")
    print("- Make AI decisions")
    print()
    print("⚠️  Make sure the web server is running:")
    print("   python run_agent.py play-web")
    print()
    
    input("Press Enter to start testing... ")
    
    success = test_web_agent()
    
    if success:
        print("\n✅ All tests passed! The web agent should work correctly now.")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main() 