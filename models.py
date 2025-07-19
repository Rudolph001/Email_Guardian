from app import db
from datetime import datetime
from flask_login import UserMixin
import json

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class EmailRecord(db.Model):
    __tablename__ = 'email_records'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(64), nullable=False, index=True)
    record_id = db.Column(db.String(64), nullable=False)
    
    # Email core fields for fast filtering
    sender = db.Column(db.String(255), index=True)
    recipients = db.Column(db.Text)
    subject = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, index=True)
    
    # Analysis fields
    ml_score = db.Column(db.Float, index=True)
    risk_level = db.Column(db.String(20), index=True)
    domain_classification = db.Column(db.String(20), index=True)
    case_status = db.Column(db.String(20), default='open', index=True)  # open, cleared, escalated
    dashboard_type = db.Column(db.String(20), default='case_management', index=True)  # case_management, escalation
    
    # Attachment fields
    has_attachments = db.Column(db.Boolean, default=False, index=True)
    attachment_types = db.Column(db.Text)  # JSON array
    attachment_classification = db.Column(db.String(50))
    
    # Rule matching
    rule_matches = db.Column(db.Text)  # JSON array of matched rules
    has_rule_matches = db.Column(db.Boolean, default=False, index=True)
    
    # Size and links
    email_size = db.Column(db.String(20))  # small, medium, large
    has_links = db.Column(db.Boolean, default=False, index=True)
    
    # Full data storage (compressed)
    original_data = db.Column(db.Text)  # JSON string of original CSV row
    processed_data = db.Column(db.Text)  # JSON string of processed data
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        data = {
            'id': self.id,
            'record_id': self.record_id,
            'session_id': self.session_id,
            'sender': self.sender,
            'recipients': self.recipients,
            'subject': self.subject,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'ml_score': self.ml_score,
            'risk_level': self.risk_level,
            'domain_classification': self.domain_classification,
            'case_status': self.case_status,
            'dashboard_type': self.dashboard_type,
            'has_attachments': self.has_attachments,
            'attachment_classification': self.attachment_classification,
            'has_rule_matches': self.has_rule_matches,
            'email_size': self.email_size,
            'has_links': self.has_links,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
        
        # Parse JSON fields
        if self.attachment_types:
            try:
                data['attachment_types'] = json.loads(self.attachment_types)
            except:
                data['attachment_types'] = []
        else:
            data['attachment_types'] = []
            
        if self.rule_matches:
            try:
                data['rule_matches'] = json.loads(self.rule_matches)
            except:
                data['rule_matches'] = []
        else:
            data['rule_matches'] = []
            
        return data

class ProcessingSession(db.Model):
    __tablename__ = 'processing_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(64), unique=True, nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False)
    total_records = db.Column(db.Integer, default=0)
    processed_records = db.Column(db.Integer, default=0)
    filtered_records = db.Column(db.Integer, default=0)
    escalated_records = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default='active', index=True)  # active, completed, archived
    csv_headers = db.Column(db.Text)  # JSON string of CSV headers
    processing_stats = db.Column(db.Text)  # JSON string of processing statistics
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        data = {
            'session_id': self.session_id,
            'filename': self.filename,
            'total_records': self.total_records,
            'processed_records': self.processed_records,
            'filtered_records': self.filtered_records,
            'escalated_records': self.escalated_records,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
        
        # Parse JSON fields
        if self.csv_headers:
            try:
                data['csv_headers'] = json.loads(self.csv_headers)
            except:
                data['csv_headers'] = []
        else:
            data['csv_headers'] = []
            
        if self.processing_stats:
            try:
                data['processing_stats'] = json.loads(self.processing_stats)
            except:
                data['processing_stats'] = {}
        else:
            data['processing_stats'] = {}
            
        return data

class Rule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    conditions = db.Column(db.Text)  # JSON string of conditions
    actions = db.Column(db.Text)  # JSON string of actions
    priority = db.Column(db.Integer, default=1)
    active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
