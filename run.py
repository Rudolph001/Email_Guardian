#!/usr/bin/env python3
"""
Email Guardian - Local Application Runner
Run this script to start the Email Guardian application locally.
"""

import sys
import os
import subprocess
import time

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False

    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_missing_module(module_name, package_name=None):
    """Install a specific missing module"""
    if package_name is None:
        package_name = module_name

    print(f"Installing {package_name}...")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', package_name
        ], check=True, capture_output=True, text=True)
        print(f"✓ Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def verify_and_install_modules():
    """Verify that all required modules are available and install if missing"""
    # Module mapping: import_name -> package_name
    module_map = {
        'flask': 'flask>=2.3.0',
        'flask_sqlalchemy': 'flask-sqlalchemy>=3.0.0',
        'flask_login': 'flask-login>=0.6.0',
        'pandas': 'pandas>=2.0.0',
        'numpy': 'numpy>=1.24.0',
        'sklearn': 'scikit-learn>=1.3.0',
        'sqlalchemy': 'sqlalchemy>=2.0.0',
        'werkzeug': 'werkzeug>=2.3.0',
        'email_validator': 'email-validator>=2.0.0'
    }

    all_available = True

    for module_name, package_name in module_map.items():
        try:
            __import__(module_name)
            print(f"✓ {module_name} available")
        except ImportError:
            print(f"❌ {module_name} missing - installing...")
            if install_missing_module(module_name, package_name):
                # Try importing again after installation
                try:
                    __import__(module_name)
                    print(f"✓ {module_name} now available")
                except ImportError:
                    print(f"❌ {module_name} still not available after installation")
                    all_available = False
            else:
                all_available = False

    return all_available

def start_application():
    """Start the Email Guardian application"""
    print("\n" + "="*50)
    print("Starting Email Guardian Application...")
    print("="*50)

    try:
        # Add current directory to Python path to ensure local imports work
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        # Import and start the application
        print("Loading application modules...")
        from app import app
        print("✓ Application modules loaded successfully")
        print("\n🚀 Starting server on http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        print("-" * 50)

        app.run(host='0.0.0.0', port=5000, debug=True)

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nThis might be due to missing dependencies.")
        print("Please run setup.py first to install all dependencies.")
        return False
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("Email Guardian - Local Application Runner")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        input("Press Enter to exit...")
        return

    # Verify and install modules
    print("\nChecking dependencies...")
    if not verify_and_install_modules():
        print("\n❌ Some dependencies could not be installed.")
        print("Please run setup.py manually to resolve dependency issues.")
        input("Press Enter to exit...")
        return

    print("\n✓ All dependencies verified")

    # Start the application
    start_application()

if __name__ == "__main__":
    main()