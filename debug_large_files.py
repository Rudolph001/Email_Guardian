#!/usr/bin/env python3
"""
Debug script for large file processing issues
Run this to diagnose problems with large CSV uploads
"""

import os
import json
import gzip
from session_manager import SessionManager

def diagnose_large_file_issues():
    """Diagnose issues with large file processing"""
    print("=== EMAIL GUARDIAN LARGE FILE DIAGNOSTICS ===\n")
    
    # Check upload folder
    upload_folder = 'uploads'
    if os.path.exists(upload_folder):
        files = os.listdir(upload_folder)
        print(f"Files in uploads folder: {len(files)}")
        
        # Check for large files
        large_files = []
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(upload_folder, file)
                file_size = os.path.getsize(file_path)
                file_size_mb = file_size / (1024 * 1024)
                if file_size_mb > 10:
                    large_files.append((file, file_size_mb))
                print(f"  {file}: {file_size_mb:.2f} MB")
        
        if large_files:
            print(f"\nLarge files (>10MB) found: {len(large_files)}")
            for file, size in large_files:
                print(f"  {file}: {size:.2f} MB")
    else:
        print("No uploads folder found")
    
    # Check session files
    print(f"\n=== SESSION DATA ANALYSIS ===")
    
    session_manager = SessionManager()
    sessions_file = 'data/sessions.json'
    compressed_file = sessions_file + '.gz'
    
    # Check if compressed session file exists
    if os.path.exists(compressed_file):
        print(f"Compressed sessions file found: {compressed_file}")
        try:
            with gzip.open(compressed_file, 'rt', encoding='utf-8') as f:
                sessions_data = json.load(f)
            print(f"Sessions in compressed file: {len(sessions_data)}")
            
            # Analyze sessions
            for session_id, session in sessions_data.items():
                processed_count = len(session.get('processed_data', []))
                total_records = session.get('total_records', 0)
                print(f"  Session {session_id}: {processed_count}/{total_records} records processed")
                
        except Exception as e:
            print(f"Error reading compressed sessions: {e}")
    
    # Check regular sessions file
    if os.path.exists(sessions_file):
        try:
            with open(sessions_file, 'r') as f:
                sessions_data = json.load(f)
            print(f"Sessions in regular file: {len(sessions_data)}")
            
            # Analyze sessions
            for session_id, session in sessions_data.items():
                processed_count = len(session.get('processed_data', []))
                total_records = session.get('total_records', 0)
                print(f"  Session {session_id}: {processed_count}/{total_records} records processed")
                
        except Exception as e:
            print(f"Error reading regular sessions: {e}")
    else:
        print("No regular sessions file found")
    
    # Memory usage recommendations
    print(f"\n=== RECOMMENDATIONS FOR LARGE FILES ===")
    print("1. Files >10MB will use chunked processing (2500 records per chunk)")
    print("2. Session data >5MB will be compressed automatically")  
    print("3. For files with 10,000+ records, expect 2-5 minutes processing time")
    print("4. Check browser developer console for any frontend timeout errors")
    print("5. If data appears missing, check the session logs for save errors")

if __name__ == "__main__":
    diagnose_large_file_issues()