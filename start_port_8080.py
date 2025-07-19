#!/usr/bin/env python3
"""
Start the Email Guardian application on port 8080
"""
import os
import sys
from app import app

if __name__ == '__main__':
    print("Starting Email Guardian on port 8080...")
    app.run(host='0.0.0.0', port=8080, debug=True)