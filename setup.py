
#!/usr/bin/env python3
"""
Email Guardian - Setup Script
Installs all dependencies and prepares the application for local use.
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install Python dependencies"""
    dependencies = [
        "flask>=3.1.1",
        "flask-sqlalchemy>=3.1.1",
        "pandas>=2.3.1",
        "numpy>=2.3.1",
        "scikit-learn>=1.7.1",
        "sqlalchemy>=2.0.41",
        "werkzeug>=3.1.3",
        "email-validator>=2.2.0",
        "gunicorn>=23.0.0"
    ]
    
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep.split('>=')[0]}"):
            return False
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'uploads', 'instance']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory exists: {directory}")

def initialize_data_files():
    """Initialize data files if they don't exist"""
    data_files = {
        'data/sessions.json': '{}',
        'data/whitelists.json': '{"domains": [], "email_addresses": []}',
        'data/attachment_keywords.json': '{"high_risk": [], "medium_risk": [], "low_risk": []}'
    }
    
    for file_path, default_content in data_files.items():
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write(default_content)
            print(f"✓ Created data file: {file_path}")
        else:
            print(f"✓ Data file exists: {file_path}")

def main():
    """Main setup function"""
    print("=" * 60)
    print("    EMAIL GUARDIAN - LOCAL SETUP")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    print("\n" + "=" * 40)
    print("INSTALLING DEPENDENCIES")
    print("=" * 40)
    
    if not install_dependencies():
        print("\n✗ Failed to install dependencies")
        sys.exit(1)
    
    # Create directories
    print("\n" + "=" * 40)
    print("CREATING DIRECTORIES")
    print("=" * 40)
    create_directories()
    
    # Initialize data files
    print("\n" + "=" * 40)
    print("INITIALIZING DATA FILES")
    print("=" * 40)
    initialize_data_files()
    
    # Success message
    print("\n" + "=" * 60)
    print("    SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nTo start the application, run:")
    print("  python run.py")
    print("\nOr for development mode:")
    print("  python run.py --dev")
    print("\nThe application will be available at:")
    print("  http://localhost:5000")
    print("=" * 60)

if __name__ == "__main__":
    main()
