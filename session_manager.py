import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
import uuid

class SessionManager:
    """Manage upload sessions and data persistence"""

    def __init__(self):
        self.sessions_file = 'data/sessions.json'
        self.whitelists_file = 'data/whitelists.json'
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def initialize_data_files():
        """Initialize data files if they don't exist"""
        sessions_file = 'data/sessions.json'
        whitelists_file = 'data/whitelists.json'

        if not os.path.exists(sessions_file):
            with open(sessions_file, 'w') as f:
                json.dump({}, f, indent=2)

        if not os.path.exists(whitelists_file):
            default_whitelists = {
                'domains': [
                    'company.com',
                    'corporate.com',
                    'internal.com',
                    'trusted-partner.com'
                ],
                'updated_at': datetime.now().isoformat()
            }
            with open(whitelists_file, 'w') as f:
                json.dump(default_whitelists, f, indent=2)

    def load_sessions(self) -> Dict:
        """Load all sessions from file"""
        try:
            if os.path.exists(self.sessions_file):
                with open(self.sessions_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Error loading sessions: {str(e)}")
            return {}

    def save_sessions(self, sessions: Dict) -> bool:
        """Save sessions to file"""
        try:
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions, f, indent=2, default=str)
            return True
        except Exception as e:
            self.logger.error(f"Error saving sessions: {str(e)}")
            return False

    def create_session(self, session_id: str, filename: str, csv_headers: List[str]) -> Dict:
        """Create a new processing session"""
        try:
            sessions = self.load_sessions()

            session_data = {
                'session_id': session_id,
                'filename': filename,
                'csv_headers': csv_headers,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'status': 'active',
                'total_records': 0,
                'processed_records': 0,
                'filtered_records': 0,
                'cases': {},
                'processed_data': []
            }

            sessions[session_id] = session_data

            if self.save_sessions(sessions):
                return {'success': True, 'session': session_data}
            else:
                return {'success': False, 'error': 'Failed to save session'}

        except Exception as e:
            self.logger.error(f"Error creating session: {str(e)}")
            return {'success': False, 'error': str(e)}

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get a specific session"""
        sessions = self.load_sessions()
        return sessions.get(session_id)

    def get_all_sessions(self) -> List[Dict]:
        """Get all sessions for display"""
        sessions = self.load_sessions()
        return list(sessions.values())

    def update_session_data(self, session_id: str, processed_data: List[Dict]) -> bool:
        """Update session with processed data"""
        try:
            sessions = self.load_sessions()

            if session_id in sessions:
                sessions[session_id]['processed_data'] = processed_data
                sessions[session_id]['processed_records'] = len(processed_data)
                sessions[session_id]['updated_at'] = datetime.now().isoformat()

                self.logger.info(f"Updating session {session_id} with {len(processed_data)} processed records")

                if self.save_sessions(sessions):
                    # Verify the data was saved
                    saved_session = self.get_session(session_id)
                    if saved_session and 'processed_data' in saved_session:
                        self.logger.info(f"Verified: {len(saved_session['processed_data'])} records saved to session {session_id}")
                    else:
                        self.logger.error(f"Failed to verify saved data for session {session_id}")

                    return True
                else:
                     return False

            return False

        except Exception as e:
            self.logger.error(f"Error updating session data: {str(e)}")
            return False

    def get_processed_data(self, session_id: str) -> List[Dict]:
        """Get processed data for a session"""
        try:
            session_data = self.get_session(session_id)
            self.logger.info(f"Session data keys: {list(session_data.keys()) if session_data else 'No session data'}")

            if session_data and 'processed_data' in session_data:
                processed_data = session_data['processed_data']
                self.logger.info(f"Found {len(processed_data)} processed records in session {session_id}")
                return processed_data
            else:
                self.logger.warning(f"No processed_data found in session {session_id}")
                return []
        except Exception as e:
            self.logger.error(f"Error getting processed data: {str(e)}")
            return []

    def update_case_status(self, session_id: str, record_id: int, status: str) -> Dict:
        """Update case status (cleared, escalated)"""
        try:
            sessions = self.load_sessions()

            if session_id in sessions:
                if 'cases' not in sessions[session_id]:
                    sessions[session_id]['cases'] = {}

                sessions[session_id]['cases'][str(record_id)] = {
                    'status': status,
                    'updated_at': datetime.now().isoformat()
                }

                sessions[session_id]['updated_at'] = datetime.now().isoformat()

                if self.save_sessions(sessions):
                    return {'success': True}
                else:
                    return {'success': False, 'error': 'Failed to save session'}

            return {'success': False, 'error': 'Session not found'}

        except Exception as e:
            self.logger.error(f"Error updating case status: {str(e)}")
            return {'success': False, 'error': str(e)}

    def generate_draft_email(self, session_id: str, record_id: int) -> Dict:
        """Generate draft email for escalation"""
        try:
            session = self.get_session(session_id)
            if not session:
                return {'success': False, 'error': 'Session not found'}

            processed_data = session.get('processed_data', [])
            if record_id >= len(processed_data):
                return {'success': False, 'error': 'Record not found'}

            record = processed_data[record_id]

            # Generate draft email content
            draft = {
                'subject': f'Email Security Alert - Escalation Required',
                'body': f"""
Email Security Alert - Escalation Required

Record Details:
- User: {record.get('sender', 'Unknown')}
- Subject: {record.get('subject', 'No subject')}
- Time: {record.get('_time', 'Unknown')}
- Recipients: {record.get('recipients', 'Unknown')}
- Attachments: {record.get('attachments', 'None')}

Risk Factors:
- Leaver Status: {record.get('leaver', 'No')}
- Wordlist Match (Subject): {record.get('wordlist_subject', 'No')}
- Wordlist Match (Attachment): {record.get('wordlist_attachment', 'No')}
- Status: {record.get('status', 'Unknown')}

Please review this email for potential data exfiltration concerns.

Generated by Email Guardian System
Session: {session_id}
                """.strip(),
                'generated_at': datetime.now().isoformat()
            }

            return {'success': True, 'draft': draft}

        except Exception as e:
            self.logger.error(f"Error generating draft email: {str(e)}")
            return {'success': False, 'error': str(e)}

    def load_whitelists(self) -> Dict:
        """Load whitelist data"""
        try:
            if os.path.exists(self.whitelists_file):
                with open(self.whitelists_file, 'r') as f:
                    return json.load(f)
            return {'domains': [], 'updated_at': datetime.now().isoformat()}
        except Exception as e:
            self.logger.error(f"Error loading whitelists: {str(e)}")
            return {'domains': [], 'updated_at': datetime.now().isoformat()}

    def save_whitelists(self, whitelists: Dict) -> bool:
        """Save whitelist data"""
        try:
            with open(self.whitelists_file, 'w') as f:
                json.dump(whitelists, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Error saving whitelists: {str(e)}")
            return False

    def get_whitelists(self) -> Dict:
        """Get whitelist data"""
        return self.load_whitelists()

    def update_whitelist(self, domains: List[str]) -> Dict:
        """Update whitelist domains"""
        try:
            whitelists = {
                'domains': domains,
                'updated_at': datetime.now().isoformat()
            }

            if self.save_whitelists(whitelists):
                return {'success': True}
            else:
                return {'success': False, 'error': 'Failed to save whitelists'}

        except Exception as e:
            self.logger.error(f"Error updating whitelist: {str(e)}")
            return {'success': False, 'error': str(e)}

    def get_session_stats(self, session_id: str) -> Dict:
        """Get session statistics for visualization"""
        session = self.get_session(session_id)
        if not session:
            return {}

        processed_data = session.get('processed_data', [])
        cases = session.get('cases', {})

        # Calculate statistics
        stats = {
            'total_records': len(processed_data),
            'cases_cleared': sum(1 for case in cases.values() if case.get('status') == 'clear'),
            'cases_escalated': sum(1 for case in cases.values() if case.get('status') == 'escalate'),
            'cases_open': len(processed_data) - len(cases),
            'processing_date': session.get('created_at', ''),
            'filename': session.get('filename', '')
        }

        return stats

    def delete_session(self, session_id: str) -> Dict:
        """Delete a processing session"""
        try:
            sessions = self.load_sessions()
            
            if session_id not in sessions:
                return {'success': False, 'error': 'Session not found'}
            
            # Remove the session
            del sessions[session_id]
            
            if self.save_sessions(sessions):
                self.logger.info(f"Successfully deleted session {session_id}")
                return {'success': True}
            else:
                return {'success': False, 'error': 'Failed to save sessions after deletion'}
                
        except Exception as e:
            self.logger.error(f"Error deleting session {session_id}: {str(e)}")
            return {'success': False, 'error': str(e)}

    def export_session(self, session_id: str) -> Dict:
        """Export complete session data"""
        session = self.get_session(session_id)
        if not session:
            return {}

        return {
            'session_metadata': {
                'session_id': session_id,
                'filename': session.get('filename'),
                'created_at': session.get('created_at'),
                'total_records': session.get('total_records'),
                'processed_records': session.get('processed_records')
            },
            'processed_data': session.get('processed_data', []),
            'cases': session.get('cases', {}),
            'csv_headers': session.get('csv_headers', []),
            'exported_at': datetime.now().isoformat()
        }
    
    def update_processed_data(self, session_id: str, processed_data: List[Dict]) -> Dict:
        """Update processed data for a session"""
        try:
            sessions = self.load_sessions()
            
            if session_id not in sessions:
                return {'success': False, 'error': 'Session not found'}
            
            sessions[session_id]['processed_data'] = processed_data
            sessions[session_id]['updated_at'] = datetime.now().isoformat()
            
            if self.save_sessions(sessions):
                self.logger.info(f"Updated processed data for session {session_id}")
                return {'success': True}
            else:
                return {'success': False, 'error': 'Failed to save session data'}
                
        except Exception as e:
            self.logger.error(f"Error updating processed data for session {session_id}: {str(e)}")
            return {'success': False, 'error': str(e)}