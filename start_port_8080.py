#!/usr/bin/env python3
"""
Start Email Guardian on port 8080
"""
import os
import sys
import subprocess

def main():
    print("Starting Email Guardian on port 8080...")
    print("=" * 50)
    
    # Change to the project directory
    os.chdir('/home/runner/workspace')
    
    # Start the application
    try:
        subprocess.run([sys.executable, 'main.py'], check=True)
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error starting application: {e}")

if __name__ == "__main__":
    main()