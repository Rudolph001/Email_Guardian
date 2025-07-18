
#!/usr/bin/env python3
"""
Email Guardian - Quick Installer
One-click installation and setup for Email Guardian.
"""

import subprocess
import sys
import os

def main():
    """Quick installer that runs setup and provides instructions"""
    print("=" * 60)
    print("    EMAIL GUARDIAN - QUICK INSTALLER")
    print("=" * 60)
    
    # Check if setup.py exists
    if not os.path.exists('setup.py'):
        print("✗ setup.py not found in current directory")
        print("Please make sure you're in the Email Guardian directory")
        sys.exit(1)
    
    # Run setup
    try:
        print("Running setup...")
        subprocess.run([sys.executable, 'setup.py'], check=True)
        
        print("\n" + "=" * 60)
        print("    INSTALLATION COMPLETE!")
        print("=" * 60)
        print("\nTo start Email Guardian:")
        print("  python run.py")
        print("\nFor development mode:")
        print("  python run.py --dev")
        print("\nFor production mode:")
        print("  python run.py --prod")
        print("\nThe application will be available at http://localhost:5000")
        print("=" * 60)
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Installation failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n✗ Installation cancelled by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
