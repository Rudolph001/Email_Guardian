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
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(64), nullable=False)
    original_data = db.Column(db.Text)  # JSON string of original CSV row
    processed_data = db.Column(db.Text)  # JSON string of processed data
    ml_score = db.Column(db.Float)
    risk_level = db.Column(db.String(20))
    domain_classification = db.Column(db.String(20))
    case_status = db.Column(db.String(20), default='open')  # open, cleared, escalated
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ProcessingSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(64), unique=True, nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    total_records = db.Column(db.Integer, default=0)
    processed_records = db.Column(db.Integer, default=0)
    filtered_records = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default='active')  # active, completed, archived
    csv_headers = db.Column(db.Text)  # JSON string of CSV headers
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

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
