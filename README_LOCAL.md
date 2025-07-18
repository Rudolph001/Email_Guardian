
# Email Guardian - Local Installation Guide

## Quick Start

### 1. One-Click Installation
```bash
python install.py
```

### 2. Manual Installation
```bash
# Install dependencies and setup
python setup.py

# Start the application
python run.py
```

## Installation Options

### Option 1: Quick Installer
```bash
python install.py
```
This runs the complete setup and provides usage instructions.

### Option 2: Step-by-Step Setup
```bash
# 1. Install dependencies and create directories
python setup.py

# 2. Start in development mode (recommended for local use)
python run.py --dev

# 3. Or start in production mode
python run.py --prod
```

## Usage

### Development Mode (Recommended for Local)
```bash
python run.py --dev
```
- Auto-reloads on code changes
- Debug mode enabled
- Opens browser automatically
- Runs on http://localhost:5000

### Production Mode
```bash
python run.py --prod
```
- Uses Gunicorn if available
- Optimized for performance
- No auto-reload

### Custom Configuration
```bash
# Custom host and port
python run.py --host 0.0.0.0 --port 8080

# Don't open browser automatically
python run.py --no-browser

# Production with custom workers
python run.py --prod --workers 8
```

## Requirements

- Python 3.8 or higher
- Internet connection (for initial dependency installation)

## Platform Support

- ✅ Windows 10/11
- ✅ macOS 10.15+
- ✅ Linux (Ubuntu, CentOS, etc.)

## Troubleshooting

### Python Not Found
Make sure Python is installed and added to your PATH:
- Windows: Download from python.org
- macOS: Use Homebrew `brew install python3`
- Linux: Use your package manager `sudo apt install python3 python3-pip`

### Permission Errors
On macOS/Linux, you might need to use:
```bash
python3 setup.py
python3 run.py
```

### Dependency Installation Issues
If pip installation fails, try:
```bash
python -m pip install --upgrade pip
python setup.py
```

## File Structure After Installation

```
email-guardian/
├── data/                 # Application data
├── uploads/              # File uploads
├── instance/             # Database files
├── static/               # CSS, JS, images
├── templates/            # HTML templates
├── setup.py              # Installation script
├── run.py                # Application runner
├── install.py            # Quick installer
└── main.py               # Application entry point
```

## Next Steps

1. Start the application: `python run.py`
2. Open http://localhost:5000 in your browser
3. Upload a Tessian CSV file to begin analysis
4. Use the dashboard to review and manage email security cases

## Support

For issues or questions, check the application logs or create an issue in the repository.
