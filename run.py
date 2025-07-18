
#!/usr/bin/env python3
"""
Email Guardian - Run Script
Starts the Email Guardian application locally.
"""

import os
import sys
import argparse
import subprocess
import webbrowser
import time
import threading

def check_dependencies():
    """Check if required dependencies are installed"""
    required_modules = [
        'flask', 'pandas', 'numpy', 'sklearn', 'sqlalchemy', 'werkzeug'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print("✗ Missing dependencies:")
        for module in missing:
            print(f"  - {module}")
        print("\nRun 'python setup.py' to install dependencies")
        return False
    
    print("✓ All dependencies are installed")
    return True

def check_data_files():
    """Check if data files exist"""
    required_files = [
        'data/sessions.json',
        'data/whitelists.json', 
        'data/attachment_keywords.json'
    ]
    
    missing = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing.append(file_path)
    
    if missing:
        print("✗ Missing data files:")
        for file_path in missing:
            print(f"  - {file_path}")
        print("\nRun 'python setup.py' to initialize data files")
        return False
    
    print("✓ All data files are present")
    return True

def open_browser_delayed(url, delay=2):
    """Open browser after a delay"""
    time.sleep(delay)
    try:
        webbrowser.open(url)
        print(f"✓ Opened browser at {url}")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print(f"Please open {url} manually")

def run_development_server(host='127.0.0.1', port=5000, debug=True, open_browser=True):
    """Run the Flask development server"""
    print("\n" + "=" * 50)
    print("    STARTING EMAIL GUARDIAN")
    print("=" * 50)
    print(f"Mode: {'Development' if debug else 'Production'}")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"URL: http://{host}:{port}")
    print("=" * 50)
    
    if open_browser:
        # Start browser opening in background
        browser_thread = threading.Thread(
            target=open_browser_delayed, 
            args=(f"http://{host}:{port}",)
        )
        browser_thread.daemon = True
        browser_thread.start()
    
    # Set environment variables
    os.environ['FLASK_APP'] = 'main.py'
    os.environ['FLASK_ENV'] = 'development' if debug else 'production'
    
    try:
        # Import and run the app
        from app import app
        app.run(host=host, port=port, debug=debug, use_reloader=debug)
    except KeyboardInterrupt:
        print("\n\n✓ Application stopped by user")
    except Exception as e:
        print(f"\n✗ Error starting application: {e}")
        sys.exit(1)

def run_production_server(host='0.0.0.0', port=5000, workers=4):
    """Run with Gunicorn for production"""
    print("\n" + "=" * 50)
    print("    STARTING EMAIL GUARDIAN (PRODUCTION)")
    print("=" * 50)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Workers: {workers}")
    print("=" * 50)
    
    try:
        cmd = [
            'gunicorn',
            '--bind', f'{host}:{port}',
            '--workers', str(workers),
            '--worker-class', 'sync',
            '--timeout', '120',
            '--keep-alive', '2',
            '--max-requests', '1000',
            '--max-requests-jitter', '100',
            'main:app'
        ]
        
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\n✓ Application stopped by user")
    except FileNotFoundError:
        print("\n✗ Gunicorn not found. Running with Flask development server instead...")
        run_development_server(host=host, port=port, debug=False, open_browser=False)
    except Exception as e:
        print(f"\n✗ Error starting application: {e}")
        sys.exit(1)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Email Guardian - Email Security Analysis Tool')
    parser.add_argument('--dev', action='store_true', help='Run in development mode')
    parser.add_argument('--prod', action='store_true', help='Run in production mode with Gunicorn')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--workers', type=int, default=4, help='Number of Gunicorn workers (default: 4)')
    parser.add_argument('--no-browser', action='store_true', help='Don\'t open browser automatically')
    
    args = parser.parse_args()
    
    print("Email Guardian - Email Security Analysis Tool")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check data files
    if not check_data_files():
        sys.exit(1)
    
    # Determine run mode
    if args.prod:
        run_production_server(host=args.host, port=args.port, workers=args.workers)
    else:
        # Default to development mode, or if --dev is specified
        debug_mode = args.dev or not args.prod
        run_development_server(
            host=args.host, 
            port=args.port, 
            debug=debug_mode,
            open_browser=not args.no_browser
        )

if __name__ == "__main__":
    main()
