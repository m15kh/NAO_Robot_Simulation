#!/usr/bin/env python3
# filepath: /home/liam/resana/webot_python/release_webcam.py

import os
import subprocess
import sys
import time

def find_webcam_processes():
    """Find processes that might be using webcam devices."""
    try:
        # List processes using video devices
        result = subprocess.run(
            ["fuser", "-v", "/dev/video*"], 
            capture_output=True, 
            text=True
        )
        return result.stdout
    except Exception as e:
        print(f"Error checking webcam processes: {e}")
        return ""

def kill_webcam_processes():
    """Kill all processes using webcam devices."""
    try:
        subprocess.run(["sudo", "fuser", "-k", "/dev/video*"])
        print("Killed processes using webcam devices")
        return True
    except Exception as e:
        print(f"Failed to kill webcam processes: {e}")
        return False

def reset_usb_devices():
    """Reset USB devices that might be webcams."""
    try:
        # Needs sudo privileges
        print("Attempting to reset USB devices (may require sudo)...")
        
        # List USB devices
        result = subprocess.run(
            ["lsusb"], 
            capture_output=True, 
            text=True
        )
        
        # Look for potential webcam devices
        for line in result.stdout.splitlines():
            if "Camera" in line or "cam" in line.lower() or "video" in line.lower():
                parts = line.split()
                if len(parts) >= 2:
                    bus = parts[1]
                    device = parts[3].rstrip(':')
                    path = f"/dev/bus/usb/{bus}/{device}"
                    print(f"Resetting device at {path}")
                    try:
                        subprocess.run(["sudo", "udevadm", "trigger", "--action=remove", f"--property-match=DEVNAME={path}"])
                        time.sleep(1)
                        subprocess.run(["sudo", "udevadm", "trigger", "--action=add", f"--property-match=DEVNAME={path}"])
                    except Exception as e:
                        print(f"Error resetting device: {e}")
        
        return True
    except Exception as e:
        print(f"Failed to reset USB devices: {e}")
        return False

if __name__ == "__main__":
    print("Current processes using webcam devices:")
    processes = find_webcam_processes()
    if processes.strip():
        print(processes)
        
        user_input = input("Do you want to kill all processes using webcam devices? (y/n): ")
        if user_input.lower() == 'y':
            kill_webcam_processes()
    else:
        print("No processes found using webcam devices")
    
    user_input = input("Do you want to try resetting USB devices? (y/n): ")
    if user_input.lower() == 'y':
        reset_usb_devices()
    
    print("\nAdditional troubleshooting tips:")
    print("1. Try unplugging and reconnecting your webcam")
    print("2. Check if your webcam is recognized with: ls -l /dev/video*")
    print("3. Check device permissions with: ls -la /dev/video*")
    print("4. Verify camera compatibility with: v4l2-ctl --list-devices")
    print("5. Install v4l2 tools if needed: sudo apt install v4l-utils")