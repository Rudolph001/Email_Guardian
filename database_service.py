"""
High-performance database service for Email Guardian
Replaces JSON-based storage with PostgreSQL for much faster operations
"""

from app import db
from models import EmailRecord, ProcessingSession, Rule
from sqlalchemy import func, and_, or_
from datetime import datetime
import json
import logging
import uuid

class DatabaseService:
    """High-performance database operations for Email Guardian"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_session(self, filename: str) -> str:
        """Create a new processing session"""
        session_id = str(uuid.uuid4())
        
        session = ProcessingSession(
            session_id=session_id,
            filename=filename,
            status='processing'
        )
        
        db.session.add(session)
        db.session.commit()
        
        self.logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> dict:
        """Get session information"""
        session = ProcessingSession.query.filter_by(session_id=session_id).first()
        if not session:
            return None
        return session.to_dict()
    
    def get_all_sessions(self) -> list:
        """Get all processing sessions"""
        sessions = ProcessingSession.query.order_by(ProcessingSession.created_at.desc()).all()
        return [session.to_dict() for session in sessions]
    
    def update_session_stats(self, session_id: str, stats: dict):
        """Update session processing statistics"""
        session = ProcessingSession.query.filter_by(session_id=session_id).first()
        if session:
            session.total_records = stats.get('total_records', 0)
            session.processed_records = stats.get('processed_records', 0)
            session.filtered_records = stats.get('filtered_records', 0)
            session.escalated_records = stats.get('escalated_records', 0)
            session.status = stats.get('status', 'active')
            session.processing_stats = json.dumps(stats.get('processing_stats', {}))
            session.updated_at = datetime.utcnow()
            db.session.commit()
    
    def save_email_record(self, session_id: str, record_data: dict) -> int:
        """Save a single email record to database"""
        try:
            # Extract key fields for indexing
            sender = record_data.get('sender', '')
            if isinstance(sender, list):
                sender = ', '.join(sender) if sender else ''
            
            recipients = record_data.get('recipients', '')
            if isinstance(recipients, list):
                recipients = ', '.join(recipients) if recipients else ''
            
            # Parse timestamp
            timestamp = None
            if record_data.get('timestamp'):
                try:
                    if isinstance(record_data['timestamp'], str):
                        timestamp = datetime.fromisoformat(record_data['timestamp'].replace('Z', '+00:00'))
                    elif isinstance(record_data['timestamp'], datetime):
                        timestamp = record_data['timestamp']
                except:
                    pass
            
            # Determine email size category
            email_size = self._categorize_email_size(record_data)
            
            # Check for attachments and links
            has_attachments = self._has_attachments(record_data)
            has_links = self._has_links(record_data)
            
            # Process rule matches
            rule_results = record_data.get('rule_results', {})
            rule_matches = rule_results.get('matched_rules', [])
            has_rule_matches = bool(rule_matches)
            
            # Determine dashboard type
            dashboard_type = 'escalation' if record_data.get('case_status') == 'escalate' else 'case_management'
            
            email_record = EmailRecord(
                session_id=session_id,
                record_id=record_data.get('record_id', str(uuid.uuid4())),
                sender=sender[:255] if sender else '',  # Truncate if too long
                recipients=recipients,
                subject=record_data.get('subject', '')[:1000] if record_data.get('subject') else '',  # Truncate
                timestamp=timestamp,
                ml_score=record_data.get('ml_score'),
                risk_level=record_data.get('ml_risk_level', ''),
                domain_classification=record_data.get('domain_classification', ''),
                case_status=record_data.get('case_status', 'open'),
                dashboard_type=dashboard_type,
                has_attachments=has_attachments,
                attachment_types=json.dumps(record_data.get('attachment_types', [])),
                attachment_classification=record_data.get('attachment_classification', ''),
                rule_matches=json.dumps(rule_matches),
                has_rule_matches=has_rule_matches,
                email_size=email_size,
                has_links=has_links,
                original_data=json.dumps(record_data.get('original_data', {})),
                processed_data=json.dumps(record_data)
            )
            
            db.session.add(email_record)
            db.session.flush()  # Get the ID without committing
            
            return email_record.id
            
        except Exception as e:
            self.logger.error(f"Error saving email record: {str(e)}")
            db.session.rollback()
            raise
    
    def save_email_records_batch(self, session_id: str, records: list):
        """Save multiple email records in a batch for better performance"""
        try:
            email_records = []
            for record_data in records:
                # Extract key fields for indexing (same logic as save_email_record)
                sender = record_data.get('sender', '')
                if isinstance(sender, list):
                    sender = ', '.join(sender) if sender else ''
                
                recipients = record_data.get('recipients', '')
                if isinstance(recipients, list):
                    recipients = ', '.join(recipients) if recipients else ''
                
                # Parse timestamp
                timestamp = None
                if record_data.get('timestamp'):
                    try:
                        if isinstance(record_data['timestamp'], str):
                            timestamp = datetime.fromisoformat(record_data['timestamp'].replace('Z', '+00:00'))
                        elif isinstance(record_data['timestamp'], datetime):
                            timestamp = record_data['timestamp']
                    except:
                        pass
                
                # Determine categories
                email_size = self._categorize_email_size(record_data)
                has_attachments = self._has_attachments(record_data)
                has_links = self._has_links(record_data)
                
                # Process rule matches
                rule_results = record_data.get('rule_results', {})
                rule_matches = rule_results.get('matched_rules', [])
                has_rule_matches = bool(rule_matches)
                
                # Determine dashboard type
                dashboard_type = 'escalation' if record_data.get('case_status') == 'escalate' else 'case_management'
                
                email_record = EmailRecord(
                    session_id=session_id,
                    record_id=record_data.get('record_id', str(uuid.uuid4())),
                    sender=sender[:255] if sender else '',
                    recipients=recipients,
                    subject=record_data.get('subject', '')[:1000] if record_data.get('subject') else '',
                    timestamp=timestamp,
                    ml_score=record_data.get('ml_score'),
                    risk_level=record_data.get('ml_risk_level', ''),
                    domain_classification=record_data.get('domain_classification', ''),
                    case_status=record_data.get('case_status', 'open'),
                    dashboard_type=dashboard_type,
                    has_attachments=has_attachments,
                    attachment_types=json.dumps(record_data.get('attachment_types', [])),
                    attachment_classification=record_data.get('attachment_classification', ''),
                    rule_matches=json.dumps(rule_matches),
                    has_rule_matches=has_rule_matches,
                    email_size=email_size,
                    has_links=has_links,
                    original_data=json.dumps(record_data.get('original_data', {})),
                    processed_data=json.dumps(record_data)
                )
                
                email_records.append(email_record)
            
            # Bulk insert for much better performance
            db.session.bulk_save_objects(email_records)
            db.session.commit()
            
            self.logger.info(f"Saved {len(email_records)} email records for session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Error saving email records batch: {str(e)}")
            db.session.rollback()
            raise
    
    def get_session_records(self, session_id: str, dashboard_type: str = None, 
                           filters: dict = None, page: int = 1, per_page: int = 50) -> dict:
        """Get email records for a session with efficient filtering and pagination"""
        try:
            query = EmailRecord.query.filter_by(session_id=session_id)
            
            # Filter by dashboard type
            if dashboard_type:
                query = query.filter(EmailRecord.dashboard_type == dashboard_type)
            
            # Apply filters
            if filters:
                if filters.get('risk_filter') and filters['risk_filter'] != 'all':
                    query = query.filter(EmailRecord.risk_level == filters['risk_filter'])
                
                if filters.get('status_filter') and filters['status_filter'] != 'all':
                    query = query.filter(EmailRecord.case_status == filters['status_filter'])
                
                if filters.get('rule_filter'):
                    if filters['rule_filter'] == 'matched':
                        query = query.filter(EmailRecord.has_rule_matches == True)
                    elif filters['rule_filter'] == 'unmatched':
                        query = query.filter(EmailRecord.has_rule_matches == False)
                
                if filters.get('search'):
                    search_term = f"%{filters['search']}%"
                    query = query.filter(
                        or_(
                            EmailRecord.sender.ilike(search_term),
                            EmailRecord.recipients.ilike(search_term),
                            EmailRecord.subject.ilike(search_term)
                        )
                    )
                
                if filters.get('ml_score_min') is not None:
                    query = query.filter(EmailRecord.ml_score >= filters['ml_score_min'])
                
                if filters.get('ml_score_max') is not None:
                    query = query.filter(EmailRecord.ml_score <= filters['ml_score_max'])
                
                if filters.get('has_attachments') is not None:
                    query = query.filter(EmailRecord.has_attachments == filters['has_attachments'])
                
                if filters.get('has_links') is not None:
                    query = query.filter(EmailRecord.has_links == filters['has_links'])
                
                if filters.get('email_size'):
                    query = query.filter(EmailRecord.email_size == filters['email_size'])
            
            # Count total records for pagination
            total = query.count()
            
            # Apply sorting - default by ML score descending for case management
            if dashboard_type == 'case_management':
                query = query.order_by(EmailRecord.ml_score.desc().nullslast())
            else:
                query = query.order_by(EmailRecord.created_at.desc())
            
            # Apply pagination
            paginated = query.paginate(
                page=page, 
                per_page=per_page, 
                error_out=False
            )
            
            # Convert to dictionaries
            records = [record.to_dict() for record in paginated.items]
            
            return {
                'records': records,
                'total': total,
                'page': page,
                'per_page': per_page,
                'pages': paginated.pages,
                'has_prev': paginated.has_prev,
                'has_next': paginated.has_next
            }
            
        except Exception as e:
            self.logger.error(f"Error getting session records: {str(e)}")
            raise
    
    def update_case_status(self, session_id: str, record_id: str, status: str):
        """Update case status for a specific record"""
        try:
            record = EmailRecord.query.filter_by(
                session_id=session_id, 
                record_id=record_id
            ).first()
            
            if record:
                record.case_status = status
                record.dashboard_type = 'escalation' if status == 'escalate' else 'case_management'
                record.updated_at = datetime.utcnow()
                db.session.commit()
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error updating case status: {str(e)}")
            db.session.rollback()
            raise
    
    def get_session_statistics(self, session_id: str) -> dict:
        """Get fast statistics for a session using database aggregation"""
        try:
            # Get basic counts
            total_records = EmailRecord.query.filter_by(session_id=session_id).count()
            escalated_count = EmailRecord.query.filter_by(
                session_id=session_id, 
                dashboard_type='escalation'
            ).count()
            
            # Get risk level distribution
            risk_stats = db.session.query(
                EmailRecord.risk_level,
                func.count(EmailRecord.id)
            ).filter_by(session_id=session_id).group_by(EmailRecord.risk_level).all()
            
            # Get domain classification distribution
            domain_stats = db.session.query(
                EmailRecord.domain_classification,
                func.count(EmailRecord.id)
            ).filter_by(session_id=session_id).group_by(EmailRecord.domain_classification).all()
            
            return {
                'total_records': total_records,
                'escalated_count': escalated_count,
                'risk_distribution': dict(risk_stats),
                'domain_distribution': dict(domain_stats)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting session statistics: {str(e)}")
            return {}
    
    def _categorize_email_size(self, record_data: dict) -> str:
        """Categorize email size based on content length"""
        content_length = 0
        
        # Estimate size based on subject and content
        if record_data.get('subject'):
            content_length += len(str(record_data['subject']))
        
        if record_data.get('message_content'):
            content_length += len(str(record_data['message_content']))
        
        if content_length < 1000:
            return 'small'
        elif content_length < 5000:
            return 'medium'
        else:
            return 'large'
    
    def _has_attachments(self, record_data: dict) -> bool:
        """Check if email has attachments"""
        attachments = record_data.get('attachments', '')
        if not attachments or attachments == '-' or attachments == '':
            return False
        return True
    
    def _has_links(self, record_data: dict) -> bool:
        """Check if email has links"""
        content = str(record_data.get('message_content', ''))
        subject = str(record_data.get('subject', ''))
        
        # Simple link detection
        link_indicators = ['http://', 'https://', 'www.', '.com', '.org', '.net']
        full_text = content + ' ' + subject
        
        return any(indicator in full_text.lower() for indicator in link_indicators)
    
    def delete_session(self, session_id: str):
        """Delete a session and all its records"""
        try:
            # Delete all email records for this session
            EmailRecord.query.filter_by(session_id=session_id).delete()
            
            # Delete the session
            ProcessingSession.query.filter_by(session_id=session_id).delete()
            
            db.session.commit()
            self.logger.info(f"Deleted session {session_id} and all its records")
            
        except Exception as e:
            self.logger.error(f"Error deleting session: {str(e)}")
            db.session.rollback()
            raise