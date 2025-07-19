
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
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    
    # Special handling for Python 3.13+
    if version.major == 3 and version.minor >= 13:
        print("⚠ Python 3.13+ detected - using compatible package versions")
    
    return True

def upgrade_pip():
    """Upgrade pip to latest version"""
    print("\nUpgrading pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True, text=True)
        print("✓ pip upgraded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠ Could not upgrade pip: {e}")
        return False

def install_dependencies():
    """Install Python dependencies with version compatibility"""
    version = sys.version_info
    
    # Updated dependencies for better compatibility
    dependencies = [
        "flask>=3.1.1",
        "flask-sqlalchemy>=3.1.1",
        "flask-login>=0.6.3",
        "werkzeug>=3.1.3",
        "pandas>=2.3.1",
        "numpy>=2.3.1",
        "scikit-learn>=1.7.1",
        "sqlalchemy>=2.0.41",
        "email-validator>=2.2.0",
        "gunicorn>=23.0.0",
        "psycopg2-binary>=2.9.10",
        "openai>=1.0.0",
        "anthropic>=0.7.0",
        "networkx>=3.0",
        "textblob>=0.17.0",
        "vaderSentiment>=3.3.2"
    ]
    
    # Upgrade pip first
    upgrade_pip()
    
    # Install dependencies one by one for better error tracking
    for dep in dependencies:
        package_name = dep.split('>=')[0]
        print(f"\nInstalling {package_name}...")
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", dep
            ], check=True, capture_output=True, text=True)
            print(f"✓ {package_name} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package_name}")
            print(f"Error: {e.stderr}")
            
            # Try installing without version constraints
            try:
                print(f"Trying to install {package_name} without version constraints...")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package_name
                ], check=True, capture_output=True, text=True)
                print(f"✓ {package_name} installed without version constraints")
            except subprocess.CalledProcessError as e2:
                print(f"✗ Failed to install {package_name} even without constraints")
                print(f"Error: {e2.stderr}")
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
        'data/attachment_keywords.json': '{"high_risk": [], "medium_risk": [], "low_risk": []}',
        'data/rules.json': '{"rules": []}'
    }
    
    for file_path, default_content in data_files.items():
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write(default_content)
            print(f"✓ Created data file: {file_path}")
        else:
            print(f"✓ Data file exists: {file_path}")

def test_imports():
    """Test if all required modules can be imported"""
    required_modules = [
        'flask', 'pandas', 'numpy', 'sklearn', 'sqlalchemy', 'werkzeug',
        'openai', 'anthropic', 'networkx', 'textblob', 'vaderSentiment'
    ]
    
    print("\nTesting module imports...")
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module} import successful")
        except ImportError as e:
            print(f"✗ {module} import failed: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠ Some modules failed to import: {failed_imports}")
        print("The application might not work correctly.")
        return False
    
    print("✓ All modules imported successfully")
    return True

def main():
    """Main setup function"""
    print("=" * 60)
    print("    EMAIL GUARDIAN - LOCAL SETUP")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python executable: {sys.executable}")
    
    # Check if this is an update (data directory exists)
    if os.path.exists('data') and os.listdir('data'):
        print("\n" + "!" * 60)
        print("    EXISTING DATA DETECTED")
        print("!" * 60)
        print("Your existing sessions and uploads will be preserved.")
        print("The .gitignore file protects your data from being overwritten.")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    print("\n" + "=" * 40)
    print("INSTALLING DEPENDENCIES")
    print("=" * 40)
    
    if not install_dependencies():
        print("\n✗ Failed to install some dependencies")
        print("You can try running the application anyway, but it may not work correctly.")
    
    # Test imports
    print("\n" + "=" * 40)
    print("TESTING IMPORTS")
    print("=" * 40)
    test_imports()
    
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
    print("    SETUP COMPLETED!")
    print("=" * 60)
    print("\nTo start the application, run:")
    print("  python run.py")
    print("\nOr for development mode:")
    print("  python run.py --dev")
    print("\nThe application will be available at:")
    print("  http://localhost:8080")
    print("\n" + "=" * 60)
    print("    DATA PROTECTION INFO")
    print("=" * 60)
    print("✓ Your uploads and processed data are protected by .gitignore")
    print("✓ Git pull will NOT overwrite your existing work")
    print("✓ Only application code gets updated from GitHub")
    print("\nNote: If you encountered any dependency installation errors,")
    print("the application might still work with the available packages.")
    print("=" * 60)

if __name__ == "__main__":
    main()
