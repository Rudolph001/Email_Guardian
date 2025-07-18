from flask import render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from app import app, db
from models import EmailRecord, ProcessingSession, Rule
from session_manager import SessionManager
from data_processor import DataProcessor
from rule_engine import RuleEngine
from ml_engine import MLEngine
import os
import pandas as pd
import json
import uuid
from datetime import datetime

@app.route('/')
def index():
    """Main dashboard with session overview"""
    session_manager = SessionManager()
    sessions = session_manager.get_all_sessions()
    return render_template('index.html', sessions=sessions)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle CSV file upload and initial processing"""
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and file.filename.lower().endswith('.csv'):
        try:
            # Generate unique session ID
            session_id = str(uuid.uuid4())[:8]
            filename = secure_filename(file.filename)
            
            # Save uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
            file.save(file_path)
            
            # Process CSV file
            processor = DataProcessor()
            result = processor.process_csv(file_path, session_id, filename)
            
            if result['success']:
                flash(f'Successfully processed {result["total_records"]} records', 'success')
                return redirect(url_for('dashboard', session_id=session_id))
            else:
                flash(f'Error processing file: {result["error"]}', 'error')
                return redirect(url_for('index'))
                
        except Exception as e:
            flash(f'Error uploading file: {str(e)}', 'error')
            return redirect(url_for('index'))
    else:
        flash('Please upload a CSV file', 'error')
        return redirect(url_for('index'))

@app.route('/dashboard/<session_id>')
def dashboard(session_id):
    """Main dashboard for a specific session"""
    session_manager = SessionManager()
    session_data = session_manager.get_session(session_id)
    if not session_data:
        flash('Session not found', 'error')
        return redirect(url_for('index'))
    
    # Get processed data
    processed_data = session_manager.get_processed_data(session_id)
    app.logger.info(f"Retrieved {len(processed_data) if processed_data else 0} processed records for session {session_id}")
    
    # Debug: Log first record if available
    if processed_data:
        app.logger.info(f"Sample record keys: {list(processed_data[0].keys()) if processed_data[0] else 'Empty record'}")
    
    # Get ML insights
    ml_engine = MLEngine()
    try:
        if processed_data:
            ml_insights = ml_engine.get_insights(processed_data)
            app.logger.info(f"ML insights generated: {ml_insights.get('total_emails', 0)} emails analyzed")
        else:
            ml_insights = {
                'total_emails': 0,
                'risk_distribution': {},
                'anomaly_summary': {
                    'high_anomaly_count': 0,
                    'anomaly_percentage': 0
                },
                'pattern_summary': {
                    'total_patterns': 0,
                    'critical_patterns': 0,
                    'high_patterns': 0
                },
                'recommendations': []
            }
    except Exception as e:
        app.logger.error(f"Error getting ML insights: {str(e)}")
        ml_insights = {
            'total_emails': len(processed_data) if processed_data else 0,
            'risk_distribution': {},
            'anomaly_summary': {
                'high_anomaly_count': 0,
                'anomaly_percentage': 0
            },
            'pattern_summary': {
                'total_patterns': 0,
                'critical_patterns': 0,
                'high_patterns': 0
            },
            'recommendations': []
        }
    
    # Get rule results
    rule_engine = RuleEngine()
    try:
        rule_results = rule_engine.get_rule_results(session_id)
    except Exception as e:
        app.logger.error(f"Error getting rule results: {str(e)}")
        rule_results = {}
    
    # Ensure processed_data is a list
    if not isinstance(processed_data, list):
        processed_data = []
    
    return render_template('dashboard.html', 
                         session=session_data,
                         processed_data=processed_data,
                         ml_insights=ml_insights,
                         rule_results=rule_results)

@app.route('/cases/<session_id>')
def case_management(session_id):
    """Case management dashboard for detailed email event viewing"""
    session_manager = SessionManager()
    session_data = session_manager.get_session(session_id)
    if not session_data:
        flash('Session not found', 'error')
        return redirect(url_for('index'))
    
    # Get processed data
    processed_data = session_manager.get_processed_data(session_id)
    
    # Get filter parameters
    risk_filter = request.args.get('risk_filter', 'all')
    rule_filter = request.args.get('rule_filter', 'all')
    status_filter = request.args.get('status_filter', 'all')
    search_query = request.args.get('search', '')
    
    # Apply filters
    filtered_data = processed_data if processed_data else []
    
    if risk_filter != 'all':
        filtered_data = [d for d in filtered_data if d.get('ml_risk_level', '').lower() == risk_filter.lower()]
    
    if rule_filter != 'all':
        if rule_filter == 'matched':
            filtered_data = [d for d in filtered_data if d.get('rule_results', {}).get('matched_rules')]
        elif rule_filter == 'unmatched':
            filtered_data = [d for d in filtered_data if not d.get('rule_results', {}).get('matched_rules')]
    
    if status_filter != 'all':
        filtered_data = [d for d in filtered_data if d.get('status', '').lower() == status_filter.lower()]
    
    if search_query:
        search_lower = search_query.lower()
        filtered_data = [d for d in filtered_data if 
                        search_lower in d.get('sender', '').lower() or 
                        search_lower in d.get('subject', '').lower() or 
                        search_lower in d.get('recipients', '').lower()]
    
    # Get unique values for filter dropdowns
    risk_levels = list(set(d.get('ml_risk_level', 'Unknown') for d in processed_data if processed_data))
    statuses = list(set(d.get('status', 'Active') for d in processed_data if processed_data))
    
    return render_template('case_management.html', 
                         session=session_data,
                         cases=filtered_data,
                         risk_levels=risk_levels,
                         statuses=statuses,
                         current_filters={
                             'risk_filter': risk_filter,
                             'rule_filter': rule_filter,
                             'status_filter': status_filter,
                             'search': search_query
                         })

@app.route('/api/case/<session_id>/<record_id>')
def get_case_details(session_id, record_id):
    """API endpoint to get detailed case information for popup"""
    session_manager = SessionManager()
    processed_data = session_manager.get_processed_data(session_id)
    
    if not processed_data:
        return jsonify({'error': 'Session not found'}), 404
    
    # Find the specific record
    case_data = None
    for record in processed_data:
        if record.get('record_id') == record_id:
            case_data = record
            break
    
    if not case_data:
        return jsonify({'error': 'Case not found'}), 404
    
    return jsonify(case_data)

@app.route('/api/case/<session_id>/<record_id>/update', methods=['POST'])
def update_case_status(session_id, record_id):
    """API endpoint to update case status"""
    session_manager = SessionManager()
    processed_data = session_manager.get_processed_data(session_id)
    
    if not processed_data:
        return jsonify({'error': 'Session not found'}), 404
    
    new_status = request.json.get('status')
    notes = request.json.get('notes', '')
    
    # Find and update the record
    updated = False
    for record in processed_data:
        if record.get('record_id') == record_id:
            record['status'] = new_status
            record['notes'] = notes
            record['updated_at'] = datetime.now().isoformat()
            updated = True
            break
    
    if updated:
        # Save updated data back to session
        session_manager.update_processed_data(session_id, processed_data)
        return jsonify({'success': True, 'message': 'Case updated successfully'})
    else:
        return jsonify({'error': 'Case not found'}), 404

@app.route('/admin')
def admin():
    """Admin panel for managing whitelists and settings"""
    session_manager = SessionManager()
    whitelists = session_manager.get_whitelists()
    sessions = session_manager.get_all_sessions()
    return render_template('admin.html', whitelists=whitelists, sessions=sessions)

@app.route('/rules')
def rules():
    """Rules management interface"""
    rules = RuleEngine.get_all_rules()
    return render_template('rules.html', rules=rules)

@app.route('/rules/create', methods=['POST'])
def create_rule():
    """Create a new rule"""
    try:
        rule_data = {
            'name': request.form.get('name'),
            'description': request.form.get('description'),
            'conditions': json.loads(request.form.get('conditions', '[]')),
            'actions': json.loads(request.form.get('actions', '[]')),
            'priority': int(request.form.get('priority', 1)),
            'active': request.form.get('active') == 'on'
        }
        
        rule_engine = RuleEngine()
        result = rule_engine.create_rule(rule_data)
        
        if result['success']:
            flash('Rule created successfully', 'success')
        else:
            flash(f'Error creating rule: {result["error"]}', 'error')
            
    except Exception as e:
        flash(f'Error creating rule: {str(e)}', 'error')
    
    return redirect(url_for('rules'))

@app.route('/rules/<int:rule_id>/delete', methods=['POST'])
def delete_rule(rule_id):
    """Delete a rule"""
    rule_engine = RuleEngine()
    result = rule_engine.delete_rule(rule_id)
    
    if result['success']:
        flash('Rule deleted successfully', 'success')
    else:
        flash(f'Error deleting rule: {result["error"]}', 'error')
    
    return redirect(url_for('rules'))

@app.route('/case/<session_id>/<int:record_id>/action', methods=['POST'])
def case_action(session_id, record_id):
    """Handle case actions (clear, escalate)"""
    action = request.form.get('action')
    session_manager = SessionManager()
    
    if action in ['clear', 'escalate']:
        result = session_manager.update_case_status(session_id, record_id, action)
        
        if result['success']:
            flash(f'Case {action}d successfully', 'success')
            
            # If escalating, prepare draft email
            if action == 'escalate':
                draft_email = session_manager.generate_draft_email(session_id, record_id)
                # In a real implementation, this would integrate with Outlook
                flash(f'Draft email prepared for escalation', 'info')
        else:
            flash(f'Error {action}ing case: {result["error"]}', 'error')
    
    return redirect(url_for('dashboard', session_id=session_id))

@app.route('/admin/whitelist', methods=['POST'])
def update_whitelist():
    """Update whitelist domains"""
    domains = request.form.get('domains', '').split('\n')
    domains = [domain.strip() for domain in domains if domain.strip()]
    
    session_manager = SessionManager()
    result = session_manager.update_whitelist(domains)
    
    if result['success']:
        flash('Whitelist updated successfully', 'success')
    else:
        flash(f'Error updating whitelist: {result["error"]}', 'error')
    
    return redirect(url_for('admin'))

@app.route('/delete_session/<session_id>', methods=['POST'])
def delete_session(session_id):
    """Delete a processing session"""
    session_manager = SessionManager()
    result = session_manager.delete_session(session_id)
    
    if result['success']:
        flash(f'Session {session_id} deleted successfully', 'success')
    else:
        flash(f'Error deleting session: {result["error"]}', 'error')
    
    return redirect(url_for('admin'))

@app.route('/export/<session_id>')
def export_session(session_id):
    """Export session data as JSON"""
    session_manager = SessionManager()
    session_data = session_manager.get_session(session_id)
    if not session_data:
        flash('Session not found', 'error')
        return redirect(url_for('index'))
    
    export_data = session_manager.export_session(session_id)
    
    # Create export file
    export_filename = f"email_guardian_export_{session_id}.json"
    export_path = os.path.join(app.config['UPLOAD_FOLDER'], export_filename)
    
    with open(export_path, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], export_filename, as_attachment=True)

@app.route('/api/ml_insights/<session_id>')
def api_ml_insights(session_id):
    """API endpoint for ML insights (for Chart.js)"""
    session_manager = SessionManager()
    processed_data = session_manager.get_processed_data(session_id)
    ml_engine = MLEngine()
    insights = ml_engine.get_insights(processed_data)
    return jsonify(insights)

@app.route('/api/session_stats/<session_id>')
def api_session_stats(session_id):
    """API endpoint for session statistics (for Chart.js)"""
    session_manager = SessionManager()
    stats = session_manager.get_session_stats(session_id)
    return jsonify(stats)

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500
