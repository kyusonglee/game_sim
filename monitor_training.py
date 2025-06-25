#!/usr/bin/env python3
"""
Monitor GPU utilization during PPO training phases
"""

import time
import subprocess
import sys
import re

def get_gpu_usage():
    """Get current GPU memory usage"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        
        gpu_info = []
        for i, line in enumerate(lines):
            memory_used, memory_total, utilization = line.split(', ')
            gpu_info.append({
                'id': i,
                'memory_used_mb': int(memory_used),
                'memory_total_mb': int(memory_total),
                'utilization_percent': int(utilization),
                'memory_used_gb': int(memory_used) / 1024,
                'memory_total_gb': int(memory_total) / 1024,
                'memory_percent': (int(memory_used) / int(memory_total)) * 100
            })
        
        return gpu_info
    except Exception as e:
        print(f"Error getting GPU usage: {e}")
        return []

def monitor_training(interval=10):
    """Monitor GPU usage during training"""
    print("🚀 GPU Training Monitor")
    print("Monitoring GPU usage during PPO training phases...")
    print("Expected: Low usage during rollout, high usage during policy updates")
    print("-" * 80)
    
    max_usage_seen = [0, 0]  # Track max usage for each GPU
    policy_update_detected = False
    
    while True:
        try:
            gpu_info = get_gpu_usage()
            
            if not gpu_info:
                time.sleep(interval)
                continue
                
            current_time = time.strftime("%H:%M:%S")
            
            # Check for policy update phase (high GPU usage)
            total_memory_gb = sum(gpu['memory_used_gb'] for gpu in gpu_info)
            high_usage = any(gpu['memory_used_gb'] > 5.0 for gpu in gpu_info)
            very_high_usage = any(gpu['memory_used_gb'] > 15.0 for gpu in gpu_info)
            
            if very_high_usage and not policy_update_detected:
                print(f"\n🔥 POLICY UPDATE PHASE DETECTED! 🔥")
                policy_update_detected = True
            
            print(f"[{current_time}] ", end="")
            
            for gpu in gpu_info:
                memory_gb = gpu['memory_used_gb']
                utilization = gpu['utilization_percent']
                
                # Track maximum usage
                if memory_gb > max_usage_seen[gpu['id']]:
                    max_usage_seen[gpu['id']] = memory_gb
                
                # Color coding for different phases
                if memory_gb > 15.0:
                    phase = "🔥 POLICY UPDATE"
                elif memory_gb > 5.0:
                    phase = "⚡ HIGH COMPUTE"
                elif memory_gb > 1.0:
                    phase = "🔄 ROLLOUT"
                else:
                    phase = "💤 IDLE"
                
                print(f"GPU{gpu['id']}: {memory_gb:.1f}GB ({memory_gb/gpu['memory_total_gb']*100:.1f}%) "
                      f"Util:{utilization:2d}% [{phase}] ", end="")
            
            print(f"| Total: {total_memory_gb:.1f}GB | Max seen: {max_usage_seen}")
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print(f"\n📊 Training Monitor Summary:")
            print(f"Maximum GPU usage achieved:")
            for i, max_mem in enumerate(max_usage_seen):
                print(f"  GPU {i}: {max_mem:.1f}GB ({max_mem/24:.1f}% of TITAN RTX)")
            
            if policy_update_detected:
                print("✅ Policy update phases were detected!")
            else:
                print("⚠️ No high-utilization policy update phases detected yet.")
                print("   This could mean:")
                print("   - Training just started (still in rollout collection)")
                print("   - Using smaller buffer/batch sizes")
                print("   - GPU optimization not applied")
            
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(interval)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        interval = int(sys.argv[1])
    else:
        interval = 5  # Check every 5 seconds
    
    monitor_training(interval) 