
#!/usr/bin/env python3
"""
Email Guardian - Quick Installation Script
This script will install all dependencies and prepare the application.
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install dependencies with proper error handling"""
    dependencies = [
        'flask>=3.1.1',
        'flask-sqlalchemy>=3.1.1', 
        'flask-login>=0.6.3',
        'werkzeug>=3.1.3',
        'pandas>=2.3.1',
        'numpy>=2.3.1',
        'scikit-learn>=1.7.1',
        'sqlalchemy>=2.0.41',
        'email-validator>=2.2.0',
        'gunicorn>=23.0.0',
        'psycopg2-binary>=2.9.10',
        'openai>=1.0.0',
        'anthropic>=0.7.0',
        'networkx>=3.0',
        'textblob>=0.17.0',
        'vaderSentiment>=3.3.2'
    ]

    print("Installing Python dependencies...")

    # Upgrade pip first
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                      check=True, capture_output=True, text=True)
        print("✓ pip upgraded")
    except subprocess.CalledProcessError:
        print("⚠ Could not upgrade pip, continuing...")

    # Install each dependency
    failed_installs = []
    for dep in dependencies:
        package_name = dep.split('>=')[0]
        print(f"Installing {package_name}...")

        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                          check=True, capture_output=True, text=True)
            print(f"✓ {package_name} installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package_name}")
            failed_installs.append(package_name)

    return failed_installs

def test_imports():
    """Test if all required modules can be imported"""
    required_modules = [
        ('flask', 'Flask'),
        ('flask_sqlalchemy', 'Flask-SQLAlchemy'),
        ('flask_login', 'Flask-Login'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-learn'),
        ('sqlalchemy', 'SQLAlchemy'),
        ('werkzeug', 'Werkzeug'),
        ('email_validator', 'Email-Validator'),
        ('openai', 'OpenAI'),
        ('anthropic', 'Anthropic'),
        ('networkx', 'NetworkX'),
        ('textblob', 'TextBlob'),
        ('vaderSentiment', 'VaderSentiment')
    ]

    print("\nTesting module imports...")
    failed_imports = []

    for module, display_name in required_modules:
        try:
            __import__(module)
            print(f"✓ {display_name} import successful")
        except ImportError as e:
            print(f"❌ {display_name} import failed: {e}")
            failed_imports.append(display_name)

    return failed_imports

def main():
    print("Email Guardian - Quick Installation")
    print("=" * 40)

    # Check Python version
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        return 1

    # Install dependencies
    failed_installs = install_dependencies()

    if failed_installs:
        print(f"\n⚠ Some packages failed to install: {', '.join(failed_installs)}")
        print("The application might still work, but some features may be unavailable.")

    # Test imports
    failed_imports = test_imports()

    if failed_imports:
        print(f"\n❌ Some modules failed to import: {', '.join(failed_imports)}")
        print("Please try running this script again or install the missing packages manually.")
        return 1

    # Create necessary directories
    print("\nCreating application directories...")
    os.makedirs('data', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('instance', exist_ok=True)
    print("✓ Directories created")

    print("\n" + "=" * 40)
    print("✓ Installation completed successfully!")
    print("\nTo start the application, run:")
    print("  python run.py")
    print("\nThe application will be available at:")
    print("  http://localhost:8080")
    print("=" * 40)

    return 0

if __name__ == "__main__":
    sys.exit(main())
