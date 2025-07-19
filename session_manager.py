"""
The code ensures that whitelist domains are stored and compared in lowercase for consistency.
"""
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
        """Load all sessions from file, prioritizing regular file for new sessions"""
        try:
            sessions = {}

            # First load from regular sessions file (has latest sessions)
            if os.path.exists(self.sessions_file):
                with open(self.sessions_file, 'r') as f:
                    sessions = json.load(f)
                self.logger.debug(f"Loaded {len(sessions)} sessions from regular file")

            # Then load from compressed file if it exists (older large sessions)
            compressed_file = self.sessions_file + '.gz'
            if os.path.exists(compressed_file):
                import gzip
                try:
                    with gzip.open(compressed_file, 'rt', encoding='utf-8') as f:
                        compressed_sessions = json.load(f)

                    # Merge compressed sessions with regular sessions (regular takes priority)
                    for session_id, session_data in compressed_sessions.items():
                        if session_id not in sessions:
                            sessions[session_id] = session_data

                    self.logger.debug(f"Merged {len(compressed_sessions)} sessions from compressed file")
                except Exception as e:
                    self.logger.error(f"Error loading compressed sessions: {e}")

            # Load separately stored data if it exists
            data_dir = 'data/session_data'
            if os.path.exists(data_dir):
                for session_id in sessions:
                    data_file = os.path.join(data_dir, f"{session_id}_data.json.gz")
                    if os.path.exists(data_file):
                        try:
                            import gzip
                            with gzip.open(data_file, 'rt', encoding='utf-8') as f:
                                processed_data = json.load(f)
                            sessions[session_id]['processed_data'] = processed_data
                            self.logger.debug(f"Loaded {len(processed_data)} records from {data_file}")
                        except Exception as e:
                            self.logger.error(f"Error loading processed data for session {session_id}: {e}")
                            sessions[session_id]['processed_data'] = []
                    else:
                        # Ensure processed_data exists even if no separate file
                        if 'processed_data' not in sessions[session_id]:
                            sessions[session_id]['processed_data'] = []

            return sessions
        except Exception as e:
            self.logger.error(f"Error loading sessions: {str(e)}")
            return {}

    def save_sessions(self, sessions: Dict) -> bool:
        """Save sessions to file with separate storage for large datasets"""
        try:
            import gzip
            import json
            import os

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.sessions_file), exist_ok=True)

            # Check if we need separate storage for large sessions
            json_str = json.dumps(sessions, default=str, separators=(',', ':'))  # Compact JSON

            if len(json_str) > 5 * 1024 * 1024:  # 5MB threshold for better memory management
                self.logger.info(f"Large session data detected ({len(json_str)/1024/1024:.2f}MB), using separate storage")

                # Save each session's processed_data separately for large datasets
                metadata_sessions = {}
                data_dir = 'data/session_data'
                os.makedirs(data_dir, exist_ok=True)

                for session_id, session_data in sessions.items():
                    # Create metadata without processed_data
                    metadata = {k: v for k, v in session_data.items() if k != 'processed_data'}
                    metadata_sessions[session_id] = metadata

                    # Save processed_data separately if it exists
                    if 'processed_data' in session_data and session_data['processed_data'] is not None and len(session_data['processed_data']) > 0:
                        data_file = os.path.join(data_dir, f"{session_id}_data.json.gz")
                        try:
                            with gzip.open(data_file, 'wt', encoding='utf-8') as f:
                                json.dump(session_data['processed_data'], f, default=str, separators=(',', ':'))
                            self.logger.info(f"Saved {len(session_data['processed_data'])} records to {data_file}")
                        except Exception as e:
                            self.logger.error(f"Failed to save processed data for session {session_id}: {e}")
                            return False

                # Save metadata file
                with open(self.sessions_file, 'w') as f:
                    json.dump(metadata_sessions, f, indent=2, default=str)

            else:
                # Save normally for smaller datasets
                with open(self.sessions_file, 'w') as f:
                    json.dump(sessions, f, indent=2, default=str)

            return True

        except Exception as e:
            self.logger.error(f"Error saving sessions: {str(e)}")
            return False

    def create_session(self, session_id: str, filename: str, csv_headers: List[str]) -> Dict:
        """Create a new processing session"""
        try:
            # Always save new sessions to regular file first, not compressed
            sessions = {}
            if os.path.exists(self.sessions_file):
                with open(self.sessions_file, 'r') as f:
                    sessions = json.load(f)

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

            # Save directly to regular sessions file
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions, f, indent=2, default=str)

            self.logger.info(f"Created new session {session_id} and saved to regular sessions file")
            return {'success': True, 'session': session_data}

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

    def update_session_data(self, session_id: str, processed_data: List[Dict], processing_stats: Dict = None) -> Dict:
        """Update session with processed data and processing statistics"""
        try:
            sessions = self.load_sessions()

            # If session doesn't exist in loaded sessions, try to reload from file
            if session_id not in sessions:
                self.logger.warning(f"Session {session_id} not found in loaded sessions, reloading...")
                # Try loading from regular file first (for new sessions)
                if os.path.exists(self.sessions_file):
                    with open(self.sessions_file, 'r') as f:
                        regular_sessions = json.load(f)
                    if session_id in regular_sessions:
                        sessions.update(regular_sessions)
                        self.logger.info(f"Found session {session_id} in regular sessions file")

            if session_id in sessions:
                # Filter out None values from processed_data
                filtered_data = [record for record in processed_data if record is not None]
                sessions[session_id]['processed_data'] = filtered_data
                sessions[session_id]['processed_records'] = len(filtered_data)
                sessions[session_id]['updated_at'] = datetime.now().isoformat()

                # Add processing statistics if provided
                if processing_stats:
                    sessions[session_id]['processing_stats'] = processing_stats
                    sessions[session_id]['total_records'] = processing_stats.get('total_records', len(processed_data))
                    sessions[session_id]['whitelist_filtered'] = processing_stats.get('whitelist_filtered', 0)
                    sessions[session_id]['escalated_records'] = processing_stats.get('escalated_records', 0)
                    sessions[session_id]['case_management_records'] = processing_stats.get('case_management_records', 0)

                self.logger.info(f"Updating session {session_id} with {len(processed_data)} processed records")

                # For large datasets, use separate storage immediately
                if len(json.dumps(sessions, default=str, separators=(',', ':'))) > 5 * 1024 * 1024:
                    self.logger.info(f"Large session detected, using separate storage for {session_id}")
                    success = self._save_large_session(session_id, sessions[session_id])
                    if success:
                        return {'success': True}
                    else:
                        return {'success': False, 'error': 'Failed to save large session'}
                else:
                    if self.save_sessions(sessions):
                        # Verify the data was saved
                        saved_session = self.get_session(session_id)
                        if saved_session and 'processed_data' in saved_session:
                            self.logger.info(f"Verified: {len(saved_session['processed_data'])} records saved to session {session_id}")
                        else:
                            self.logger.error(f"Failed to verify saved data for session {session_id}")

                        return {'success': True}
                    else:
                         return {'success': False, 'error': 'Failed to save session'}

            return {'success': False, 'error': 'Session not found'}

        except Exception as e:
            self.logger.error(f"Error updating session data: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _save_large_session(self, session_id: str, session_data: Dict) -> bool:
        """Save a single large session using separate storage"""
        try:
            import gzip
            import os

            # Save processed_data separately
            data_dir = 'data/session_data'
            os.makedirs(data_dir, exist_ok=True)

            if 'processed_data' in session_data and session_data['processed_data'] is not None and len(session_data['processed_data']) > 0:
                data_file = os.path.join(data_dir, f"{session_id}_data.json.gz")
                with gzip.open(data_file, 'wt', encoding='utf-8') as f:
                    json.dump(session_data['processed_data'], f, default=str, separators=(',', ':'))
                self.logger.info(f"Saved {len(session_data['processed_data'])} records to {data_file}")

            # Save metadata to regular sessions file
            regular_sessions = {}
            if os.path.exists(self.sessions_file):
                with open(self.sessions_file, 'r') as f:
                    regular_sessions = json.load(f)

            # Create metadata without processed_data
            metadata = {k: v for k, v in session_data.items() if k != 'processed_data'}
            regular_sessions[session_id] = metadata

            with open(self.sessions_file, 'w') as f:
                json.dump(regular_sessions, f, indent=2, default=str)

            self.logger.info(f"Session {session_id} metadata saved to regular sessions file")
            return True

        except Exception as e:
            self.logger.error(f"Error saving large session {session_id}: {e}")
            return False

    def _normalize_field_names(self, data: List[Dict]) -> List[Dict]:
        """Normalize field names to lowercase for consistent access"""
        normalized_data = []
        for record in data:
            if isinstance(record, dict):
                normalized_record = {}
                for key, value in record.items():
                    # Keep original key but also add lowercase version for compatibility
                    normalized_record[key] = value
                    normalized_key = str(key).lower().strip()
                    if normalized_key != key:
                        normalized_record[normalized_key] = value
                normalized_data.append(normalized_record)
            else:
                normalized_data.append(record)
        return normalized_data

    def get_processed_data(self, session_id: str, page: int = 1, per_page: int = 50, filters: Dict = None) -> Dict:
        """Get processed data for a session with pagination and filtering for faster loading"""
        try:
            session_data = self.get_session(session_id)
            self.logger.info(f"Session data keys: {list(session_data.keys()) if session_data else 'No session data'}")

            # Try to get processed_data from session first
            processed_data = []
            if session_data and 'processed_data' in session_data:
                processed_data = session_data['processed_data']
                if processed_data is None:
                    processed_data = []
                elif not isinstance(processed_data, list):
                    self.logger.warning(f"processed_data is not a list, got {type(processed_data)}")
                    processed_data = []
                else:
                    # Filter out None entries
                    processed_data = [record for record in processed_data if record is not None]

            # If not found or empty, try loading from separate compressed file
            if not processed_data:
                data_dir = 'data/session_data'
                data_file = os.path.join(data_dir, f"{session_id}_data.json.gz")

                if os.path.exists(data_file):
                    try:
                        import gzip
                        with gzip.open(data_file, 'rt', encoding='utf-8') as f:
                            processed_data = json.load(f)
                        self.logger.info(f"Loaded {len(processed_data)} records from compressed file for session {session_id}")
                    except Exception as e:
                        self.logger.error(f"Error loading compressed data file: {e}")

            if not processed_data:
                return {'data': [], 'total': 0, 'page': page, 'per_page': per_page, 'total_pages': 0}

            # Apply filters for better performance
            filtered_data = processed_data
            if filters:
                filtered_data = self._apply_filters(processed_data, filters, session_data)

            # Calculate pagination
            total = len(filtered_data)
            total_pages = (total + per_page - 1) // per_page if total > 0 else 0
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page

            # Return paginated results
            paginated_data = filtered_data[start_idx:end_idx]

            self.logger.info(f"Returning page {page} with {len(paginated_data)} records (total: {total})")

            return {
                'data': paginated_data,
                'total': total,
                'page': page,
                'per_page': per_page,
                'total_pages': total_pages,
                'has_prev': page > 1,
                'has_next': page < total_pages
            }

        except Exception as e:
            self.logger.error(f"Error getting processed data: {str(e)}")
            return {'data': [], 'total': 0, 'page': page, 'per_page': per_page, 'total_pages': 0}

    def _apply_filters(self, data: List[Dict], filters: Dict, session_data: Dict = None) -> List[Dict]:
        """Apply filters to data for faster processing"""
        filtered_data = data

        try:
            # Dashboard type filter - apply this first to get the correct dataset
            if filters.get('dashboard_type'):
                session_cases = session_data.get('cases', {}) if session_data else {}
                if filters['dashboard_type'] == 'escalation':
                    # Only escalated cases
                    escalated_data = []
                    for i, d in enumerate(data):
                        if d is None:
                            continue
                        record_id = d.get('record_id', i)
                        case_info = session_cases.get(str(record_id), {})
                        if case_info.get('status', '').lower() == 'escalate':
                            escalated_data.append(d)
                    filtered_data = escalated_data
                elif filters['dashboard_type'] == 'case_management':
                    # Exclude escalated cases
                    case_mgmt_data = []
                    for i, d in enumerate(data):
                        if d is None:
                            continue
                        record_id = d.get('record_id', i)
                        case_info = session_cases.get(str(record_id), {})
                        if case_info.get('status', '').lower() != 'escalate':
                            case_mgmt_data.append(d)
                    filtered_data = case_mgmt_data

            # Risk level filter
            if filters.get('risk_filter') and filters['risk_filter'] != 'all':
                risk_filter = filters['risk_filter'].strip()
                filtered_data = [d for d in filtered_data if d and d.get('ml_risk_level', '').strip().lower() == risk_filter.lower()]

            # Rule filter
            if filters.get('rule_filter') and filters['rule_filter'] != 'all':
                if filters['rule_filter'] == 'matched':
                    filtered_data = [d for d in filtered_data if d and d.get('rule_results', {}).get('matched_rules')]
                elif filters['rule_filter'] == 'unmatched':
                    filtered_data = [d for d in filtered_data if d and not d.get('rule_results', {}).get('matched_rules')]

            # Status filter
            if filters.get('status_filter') and filters['status_filter'] != 'all':
                status_filter = filters['status_filter'].strip()
                filtered_data = [d for d in filtered_data if d and d.get('status', '').strip().lower() == status_filter.lower()]

            # Search filter - faster string matching
            if filters.get('search') and filters['search'].strip():
                search_term = filters['search'].lower().strip()
                filtered_data = [d for d in filtered_data if d and (
                               search_term in str(d.get('sender', '')).lower() or
                               search_term in str(d.get('subject', '')).lower() or
                               search_term in str(d.get('recipients', '')).lower())]

        except Exception as e:
            self.logger.error(f"Error applying filters: {str(e)}")
            return data

        return filtered_data

    def get_processed_data_legacy(self, session_id: str) -> List[Dict]:
        """Legacy method - get all processed data (for backward compatibility)"""
        result = self.get_processed_data(session_id, page=1, per_page=999999)
        return result.get('data', [])

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
            sender_email = record.get('sender', 'Unknown')
            draft = {
                'subject': f'Email Security Alert - Escalation Required',
                'to': sender_email,
                'cc': '',
                'body': f"""
Dear {sender_email},

This is an automated security alert regarding a recent email activity that has been flagged for review.

Email Details:
- From: {sender_email}
- Subject: {record.get('subject', 'No subject')}
- Date/Time: {record.get('_time', 'Unknown')}
- Recipients: {record.get('recipients', 'Unknown')}
- Attachments: {record.get('attachments', 'None')}

Security Assessment:
- Risk Level: {record.get('ml_risk_level', 'Unknown')}
- Leaver Status: {record.get('leaver', 'No')}
- Wordlist Match (Subject): {record.get('wordlist_subject', 'No')}
- Wordlist Match (Attachment): {record.get('wordlist_attachment', 'No')}
- Status: {record.get('status', 'Unknown')}

This email has been escalated for security review due to potential data exfiltration concerns. Please contact the security team if you have any questions about this automated alert.

Best regards,
Email Guardian Security System
Session ID: {session_id}
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

    def get_whitelists(self):
        """Get current whitelist domains"""
        try:
            with open(self.whitelists_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'domains': [],
                'updated_at': datetime.now().isoformat()
            }

    def get_attachment_keywords(self):
        """Get current attachment analysis keywords"""
        try:
            with open('data/attachment_keywords.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default keywords for attachment classification
            default_keywords = {
                'business_keywords': [
                    'contract', 'invoice', 'proposal', 'presentation', 'report', 'memo',
                    'budget', 'forecast', 'analysis', 'summary', 'minutes', 'agenda',
                    'specification', 'requirements', 'policy', 'procedure', 'guidelines',
                    'training', 'manual', 'documentation', 'quarterly', 'annual',
                    'financial', 'legal', 'compliance', 'audit', 'review'
                ],
                'personal_keywords': [
                    'family', 'personal', 'private', 'photo', 'picture', 'vacation',
                    'holiday', 'birthday', 'wedding', 'baby', 'child', 'kids',
                    'home', 'house', 'car', 'pet', 'hobby', 'friend', 'resume',
                    'cv', 'portfolio', 'diary', 'journal', 'medical', 'health',
                    'insurance', 'tax', 'bank', 'credit', 'loan', 'mortgage'
                ],
                'suspicious_keywords': [
                    'confidential', 'secret', 'classified', 'proprietary', 'internal',
                    'restricted', 'sensitive', 'backup', 'copy', 'duplicate',
                    'final', 'urgent', 'immediate', 'asap', 'password', 'login',
                    'access', 'key', 'token', 'credential', 'database', 'db'
                ],
                'updated_at': datetime.now().isoformat()
            }
            self.update_attachment_keywords(default_keywords)
            return default_keywords

    def update_attachment_keywords(self, keywords_data):
        """Update attachment analysis keywords"""
        try:
            keywords_data['updated_at'] = datetime.now().isoformat()
            os.makedirs('data', exist_ok=True)
            with open('data/attachment_keywords.json', 'w') as f:
                json.dump(keywords_data, f, indent=2)
            return {'success': True, 'message': 'Attachment keywords updated successfully'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def update_whitelist(self, domains: List[str]) -> Dict:
        """Update whitelist domains"""
        try:
            # Clean domains - remove empty strings, convert to lowercase, and remove duplicates
            cleaned_domains = list(set([domain.strip().lower() for domain in domains if domain.strip()]))

            whitelists_data = {
                'domains': cleaned_domains,
                'updated_at': datetime.now().isoformat()
            }

            with open(self.whitelists_file, 'w') as f:
                json.dump(whitelists_data, f, indent=2)

            self.logger.info(f"Updated whitelist with {len(cleaned_domains)} domains")
            return {'success': True, 'domains_count': len(cleaned_domains)}

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
        """Delete a session and all its data from all storage locations"""
        try:
            import os

            # Delete from regular sessions file
            regular_sessions = {}
            if os.path.exists(self.sessions_file):
                with open(self.sessions_file, 'r') as f:
                    regular_sessions = json.load(f)

                if session_id in regular_sessions:
                    del regular_sessions[session_id]
                    with open(self.sessions_file, 'w') as f:
                        json.dump(regular_sessions, f, indent=2, default=str)
                    self.logger.info(f"Deleted session {session_id} from regular sessions file")

            # Delete from compressed sessions file
            compressed_file = self.sessions_file + '.gz'
            if os.path.exists(compressed_file):
                import gzip
                try:
                    with gzip.open(compressed_file, 'rt', encoding='utf-8') as f:
                        compressed_sessions = json.load(f)

                    if session_id in compressed_sessions:
                        del compressed_sessions[session_id]
                        with gzip.open(compressed_file, 'wt', encoding='utf-8') as f:
                            json.dump(compressed_sessions, f, default=str, separators=(',', ':'))
                        self.logger.info(f"Deleted session {session_id} from compressed sessions file")
                except Exception as e:
                    self.logger.error(f"Error updating compressed sessions: {e}")

            # Delete separately stored data file
            data_dir = 'data/session_data'
            if os.path.exists(data_dir):
                data_file = os.path.join(data_dir, f"{session_id}_data.json.gz")
                if os.path.exists(data_file):
                    os.remove(data_file)
                    self.logger.info(f"Deleted session data file: {data_file}")

            self.logger.info(f"Successfully deleted session {session_id} from all storage locations")
            return {'success': True}

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
