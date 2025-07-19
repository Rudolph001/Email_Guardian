from flask import render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from app import app, db
from models import EmailRecord, ProcessingSession, Rule
from session_manager import SessionManager
from data_processor import DataProcessor
from rule_engine import RuleEngine
from ml_engine import MLEngine
from advanced_ml_engine import AdvancedMLEngine
from domain_manager import DomainManager
import os
import pandas as pd
import numpy as np
import json
import uuid
from datetime import datetime

def reprocess_all_sessions():
    """Reprocess all existing sessions with current whitelist and rules"""
    try:
        session_manager = SessionManager()
        all_sessions = session_manager.get_all_sessions()

        if not all_sessions:
            return {'success': True, 'sessions_count': 0}

        reprocessed_count = 0
        failed_sessions = []

        for session in all_sessions:
            session_id = session.get('session_id')
            if not session_id:
                continue

            try:
                # Get original file path
                original_filename = session.get('filename', '')
                if not original_filename:
                    continue

                # Find the original uploaded file
                upload_folder = app.config.get('UPLOAD_FOLDER', 'uploads')
                if not os.path.exists(upload_folder):
                    os.makedirs(upload_folder)
                original_file_path = None

                # Try to find the original file
                for filename in os.listdir(upload_folder):
                    if filename.startswith(session_id) and filename.endswith('.csv'):
                        original_file_path = os.path.join(upload_folder, filename)
                        break

                if not original_file_path or not os.path.exists(original_file_path):
                    app.logger.warning(f"Original file not found for session {session_id}")
                    continue

                # Reprocess the session
                processor = DataProcessor()
                result = processor.process_csv(original_file_path, session_id, original_filename)

                if result['success']:
                    reprocessed_count += 1
                    app.logger.info(f"Successfully reprocessed session {session_id}")
                else:
                    failed_sessions.append(session_id)
                    app.logger.error(f"Failed to reprocess session {session_id}: {result.get('error', 'Unknown error')}")

            except Exception as e:
                failed_sessions.append(session_id)
                app.logger.error(f"Error reprocessing session {session_id}: {str(e)}")

        if failed_sessions:
            return {
                'success': False,
                'sessions_count': reprocessed_count,
                'error': f'Failed to reprocess sessions: {", ".join(failed_sessions)}'
            }
        else:
            return {'success': True, 'sessions_count': reprocessed_count}

    except Exception as e:
        app.logger.error(f"Error in reprocess_all_sessions: {str(e)}")
        return {'success': False, 'sessions_count': 0, 'error': str(e)}

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
        return jsonify({'success': False, 'error': 'No file selected'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})

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
                return jsonify({
                    'success': True, 
                    'session_id': session_id, 
                    'total_records': result.get('total_records', 0),
                    'message': f'File uploaded and processing started! Session ID: {session_id}'
                })
            else:
                # Enhanced error response with detailed information
                error_response = {
                    'success': False, 
                    'error': result.get('error', 'Unknown error occurred'),
                    'error_type': result.get('error_type', 'unknown')
                }

                # Add validation details if available
                if 'validation_details' in result:
                    validation_details = result['validation_details']
                    error_response['validation_errors'] = validation_details.get('validation_errors', [])
                    error_response['total_errors'] = validation_details.get('total_errors', 0)
                    error_response['missing_columns'] = validation_details.get('missing_columns', [])
                    error_response['available_columns'] = validation_details.get('available_columns', [])

                return jsonify(error_response)

        except Exception as e:
            app.logger.error(f"Upload exception: {str(e)}")
            return jsonify({
                'success': False, 
                'error': f'Error uploading file: {str(e)}',
                'error_type': 'upload_exception'
            })

        except Exception as e:
            flash(f'Error uploading file: {str(e)}', 'error')
            return redirect(url_for('index'))
    else:
        flash('Please upload a CSV file', 'error')
        return redirect(url_for('index'))

@app.route('/dashboard/<session_id>')
def dashboard(session_id):
    """Main dashboard for a specific session showing processing summary"""
    session_manager = SessionManager()
    session_data = session_manager.get_session(session_id)
    if not session_data:
        flash('Session not found', 'error')
        return redirect(url_for('index'))

    # Get all processed data to calculate accurate statistics
    result = session_manager.get_processed_data(session_id, page=1, per_page=999999)
    processed_data = result.get('data', [])
    total_records = result.get('total', 0)
    app.logger.info(f"Session {session_id} has {total_records} processed records")

    # Separate data based on manual escalation status
    escalated_data = []
    case_management_data = []
    session_cases = session_data.get('cases', {}) if session_data else {}

    # Process all records to separate escalated vs case management
    for i, record in enumerate(processed_data):
        if record is None:
            continue

        record_id = record.get('record_id', i)
        case_info = session_cases.get(str(record_id), {})
        case_status = case_info.get('status', '').lower()

        if case_status == 'escalate':
            record['status'] = 'Escalated'
            escalated_data.append(record)
        else:
            case_management_data.append(record)

    # Count escalated cases
    escalated_count = len(escalated_data)
    case_management_count = len(case_management_data)

    # Get processing statistics
    processing_stats = session_data.get('processing_stats', {})
    processing_steps = processing_stats.get('processing_steps', [])

    # Calculate risk level distributions from actual data
    risk_distribution = {}
    rule_matches_count = 0

    all_data = escalated_data + case_management_data
    for record in all_data:
        if record and isinstance(record, dict):
            # Count risk levels
            risk_level = record.get('ml_risk_level', 'Low')
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1

            # Count rule matches
            rule_results = record.get('rule_results', {})
            if rule_results.get('matched_rules'):
                rule_matches_count += 1

    # Get ML insights with actual data
    ml_insights = {
        'total_emails': len(all_data),
        'risk_distribution': risk_distribution,
        'rule_matches_count': rule_matches_count,
        'anomaly_summary': {
            'high_anomaly_count': len([r for r in all_data if r.get('ml_anomaly_score', 0) > 0.7]),
            'anomaly_percentage': (len([r for r in all_data if r.get('ml_anomaly_score', 0) > 0.7]) / len(all_data) * 100) if all_data else 0
        },
        'pattern_summary': {
            'total_patterns': len(set([r.get('ml_cluster', -1) for r in all_data if r.get('ml_cluster', -1) >= 0])),
            'critical_patterns': risk_distribution.get('Critical', 0),
            'high_patterns': risk_distribution.get('High', 0)
        },
        'recommendations': []
    }

    return render_template('dashboard.html',
                         session=session_data,
                         escalated_data=escalated_data,
                         case_management_data=case_management_data,
                         processing_steps=processing_steps,
                         ml_insights=ml_insights,
                         escalated_count=escalated_count,
                         case_management_count=case_management_count,
                         total_records=total_records)

@app.route('/cases/<session_id>')
def case_management(session_id):
    """Case management dashboard for detailed email event viewing"""
    session_manager = SessionManager()
    session_data = session_manager.get_session(session_id)
    if not session_data:
        flash('Session not found', 'error')
        return redirect(url_for('index'))

    # Get pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 50, type=int), 200)  # Limit max per page

    # Get processed data with optimized filtering and pagination
    filters = {
        'dashboard_type': 'case_management',
        'session_id': session_id,
        'risk_filter': request.args.get('risk_filter', 'all'),
        'rule_filter': request.args.get('rule_filter', 'all'),
        'status_filter': request.args.get('status_filter', 'all'),
        'search': request.args.get('search', '').strip()
    }

    result = session_manager.get_processed_data(session_id, page=page, per_page=per_page, filters=filters)

    # Extract paginated data and metadata
    processed_data = result.get('data', [])
    total_records = result.get('total', 0)
    total_pages = result.get('total_pages', 0)

    # Debug logging for filtering
    app.logger.info(f"Case management filters applied: {filters}")
    app.logger.info(f"Filtered results: {total_records} records from {len(processed_data)} shown")

    # Count escalated cases from session data for badge
    session_cases = session_data.get('cases', {})
    escalated_count = sum(1 for case in session_cases.values() if case.get('status', '').lower() == 'escalate')

    # Ensure records have proper IDs and status for display
    for i, record in enumerate(processed_data):
        if 'record_id' not in record:
            record['record_id'] = str(i)

        # Set status for display
        record_id = record.get('record_id', i)
        case_info = session_cases.get(str(record_id), {})
        case_status = case_info.get('status', '').lower()

        if case_status and case_status != 'escalate':
            record['status'] = case_status.title()
        elif 'status' not in record:
            record['status'] = 'Active'

    # Filtering is now handled efficiently in the session manager
    # Extract filter parameters for template context
    risk_filter = request.args.get('risk_filter', 'all')
    rule_filter = request.args.get('rule_filter', 'all')
    status_filter = request.args.get('status_filter', 'all')
    search_query = request.args.get('search', '')

    # Data is already filtered and paginated by session manager for performance
    filtered_data = processed_data

    # Pagination info from result
    has_prev = result.get('has_prev', False)
    has_next = result.get('has_next', False)

    # Get unique values for filter dropdowns (exclude 'Escalated' since those cases are in escalation dashboard)
    risk_levels = list(set(d.get('ml_risk_level', 'Unknown') for d in processed_data if processed_data))
    statuses = list(set(d.get('status', 'Active') for d in processed_data if processed_data and d.get('status', 'Active').lower() != 'escalated'))

    return render_template('case_management.html', 
                         session=session_data,
                         cases=filtered_data,
                         escalated_count=escalated_count,
                         risk_levels=['Critical', 'High', 'Medium', 'Low'],
                         statuses=['Active', 'Cleared'],
                         total_cases=total_records,
                         # Pagination info (optimized)
                         page=page,
                         per_page=per_page,
                         total_filtered=total_records,
                         total_pages=total_pages,
                         has_prev=has_prev,
                         has_next=has_next,
                         current_filters={
                             'risk_filter': risk_filter,
                             'rule_filter': rule_filter,
                             'status_filter': status_filter,
                             'search': search_query
                         })

@app.route('/escalations/<session_id>')
def escalation_dashboard(session_id):
    """Escalation dashboard for managing escalated cases"""
    session_manager = SessionManager()
    session_data = session_manager.get_session(session_id)
    if not session_data:
        flash('Session not found', 'error')
        return redirect(url_for('index'))

    # Get processed data using optimized method for escalations
    result = session_manager.get_processed_data(session_id, filters={'dashboard_type': 'escalation', 'session_id': session_id})
    processed_data = result.get('data', [])

    # Debug logging
    app.logger.info(f"Escalation dashboard - Total records: {len(processed_data) if processed_data else 0}")

    # Filter only manually escalated cases - check case status in session data
    escalated_cases = []
    if processed_data:
        session_cases = session_data.get('cases', {})
        for i, d in enumerate(processed_data):
            # Skip None records
            if d is None:
                continue

            # Check if this specific record was manually escalated
            record_id = d.get('record_id', i)
            case_info = session_cases.get(str(record_id), {})
            case_status = case_info.get('status', '').lower()

            # Only include records that were manually escalated via the escalate button
            if case_status == 'escalate':
                # Ensure record has an ID for actions
                if 'record_id' not in d:
                    d['record_id'] = str(i)
                d['status'] = 'Escalated'  # Set display status
                escalated_cases.append(d)

    app.logger.info(f"Escalation dashboard - Escalated cases found: {len(escalated_cases)}")

    # Get filter parameters
    risk_filter = request.args.get('risk_filter', 'all')
    rule_filter = request.args.get('rule_filter', 'all')
    priority_filter = request.args.get('priority_filter', 'all')
    search_query = request.args.get('search', '')

    # Apply additional filters
    filtered_data = escalated_cases

    if risk_filter != 'all':
        filtered_data = [d for d in filtered_data if d.get('ml_risk_level', '').lower() == risk_filter.lower()]

    if rule_filter != 'all':
        if rule_filter == 'matched':
            filtered_data = [d for d in filtered_data if d.get('rule_results', {}).get('matched_rules')]
        elif rule_filter == 'unmatched':
            filtered_data = [d for d in filtered_data if not d.get('rule_results', {}).get('matched_rules')]

    if priority_filter != 'all':
        if priority_filter == 'high':
            filtered_data = [d for d in filtered_data if d.get('ml_risk_level', '').lower() in ['critical', 'high']]
        elif priority_filter == 'medium':
            filtered_data = [d for d in filtered_data if d.get('ml_risk_level', '').lower() == 'medium']
        elif priority_filter == 'low':
            filtered_data = [d for d in filtered_data if d.get('ml_risk_level', '').lower() == 'low']

    if search_query:
        search_lower = search_query.lower()
        filtered_data = [d for d in filtered_data if 
                        search_lower in d.get('sender', '').lower() or 
                        search_lower in d.get('subject', '').lower() or 
                        search_lower in d.get('recipients', '').lower()]

    app.logger.info(f"Escalation dashboard - Filtered escalations: {len(filtered_data)}")

    # Get unique values for filter dropdowns
    risk_levels = list(set(d.get('ml_risk_level', 'Unknown') for d in escalated_cases if escalated_cases))
    priorities = ['High', 'Medium', 'Low']

    return render_template('escalation_dashboard.html', 
                         session=session_data,
                         escalations=filtered_data,
                         risk_levels=risk_levels,
                         priorities=priorities,
                         current_filters={
                             'risk_filter': risk_filter,
                             'rule_filter': rule_filter,
                             'priority_filter': priority_filter,
                             'search': search_query
                         })

@app.route('/api/case/<session_id>/<record_id>')
def get_case_details(session_id, record_id):
    """API endpoint to get detailed case information for popup"""
    session_manager = SessionManager()
    result = session_manager.get_processed_data(session_id, page=1, per_page=999999)  # Get all data for search
    processed_data = result.get('data', [])

    if not processed_data:
        return jsonify({'error': 'Session not found'}), 404

    # Find the specific record - try multiple ID formats
    case_data = None
    for i, record in enumerate(processed_data):
        # Check various possible record ID formats
        if (record.get('record_id') == record_id or 
            str(record.get('record_id')) == record_id or
            str(i) == record_id):
            case_data = record
            # Ensure record has a record_id for future references
            if 'record_id' not in record:
                record['record_id'] = str(i)
            break

    if not case_data:
        app.logger.error(f"Record ID {record_id} not found in session {session_id}")
        app.logger.error(f"Available records: {len(processed_data)}")
        app.logger.error(f"Available record IDs: {[r.get('record_id', 'None') for r in processed_data[:5]]}")
        return jsonify({'error': 'Case not found'}), 404

    # Clean NaN values before JSON serialization
    def clean_nan_values(data):
        if isinstance(data, dict):
            return {k: clean_nan_values(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [clean_nan_values(item) for item in data]
        elif isinstance(data, float) and (np.isnan(data) or pd.isna(data)):
            return None
        elif data is np.nan:
            return None
        else:
            return data

    cleaned_case_data = clean_nan_values(case_data)

    # Add detailed ML explanations for better user understanding
    ml_explanations = generate_ml_explanations(cleaned_case_data)
    cleaned_case_data['ml_explanations'] = ml_explanations

    return jsonify(cleaned_case_data)

def generate_ml_explanations(case_data):
    """Generate user-friendly explanations for ML analysis results"""
    explanations = {
        'risk_analysis': [],
        'anomaly_factors': [],
        'attachment_analysis': [],
        'behavioral_patterns': [],
        'recommendations': []
    }

    # Risk Level Analysis
    risk_level = case_data.get('ml_risk_level', 'Unknown')
    anomaly_score = case_data.get('ml_anomaly_score', 0)

    # Handle null anomaly scores
    if anomaly_score is None:
        anomaly_score = 0

    try:
        anomaly_score = float(anomaly_score)
    except (ValueError, TypeError):
        anomaly_score = 0

    if risk_level == 'Critical':
        explanations['risk_analysis'].append("This email shows multiple high-risk indicators that require immediate attention.")
    elif risk_level == 'High':
        explanations['risk_analysis'].append("Several concerning patterns detected that warrant investigation.")
    elif risk_level == 'Medium':
        explanations['risk_analysis'].append("Some unusual characteristics found, but within acceptable risk tolerance.")
    elif risk_level == 'Low':
        explanations['risk_analysis'].append("Email appears normal with minimal security concerns.")

    # Anomaly Score Explanation
    if anomaly_score > 0.8:
        explanations['anomaly_factors'].append("Highly unusual email patterns detected - significantly different from normal communication.")
        explanations['anomaly_factors'].append("Factors: Unusual timing, recipient patterns, or content characteristics.")
    elif anomaly_score > 0.6:
        explanations['anomaly_factors'].append("Moderately unusual patterns detected - some characteristics deviate from normal behavior.")
        explanations['anomaly_factors'].append("Factors: Uncommon sender-recipient combinations or timing patterns.")
    elif anomaly_score > 0.3:
        explanations['anomaly_factors'].append("Minor deviations from typical email patterns detected.")
        explanations['anomaly_factors'].append("Factors: Slightly unusual timing or communication patterns.")
    else:
        explanations['anomaly_factors'].append("Email follows typical communication patterns with no significant anomalies.")

    # Advanced Attachment Risk Analysis
    attachment_type = case_data.get('attachment_classification', 'Unknown')
    has_attachments = case_data.get('has_attachments', False)
    attachment_risk_score = case_data.get('attachment_risk_score', 0.0)
    attachment_risk_level = case_data.get('attachment_risk_level', 'Unknown')
    attachment_risk_factors = case_data.get('attachment_risk_factors', [])
    malicious_indicators = case_data.get('attachment_malicious_indicators', [])
    exfiltration_risk = case_data.get('attachment_exfiltration_risk', 'None')

    if has_attachments:
        # Primary risk assessment
        if attachment_risk_level == 'Critical Risk':
            explanations['attachment_analysis'].append("🔴 CRITICAL ATTACHMENT RISK - Immediate investigation required!")
            explanations['attachment_analysis'].append(f"Risk Score: {attachment_risk_score}/100 - Very high probability of malicious intent")
        elif attachment_risk_level == 'High Risk':
            explanations['attachment_analysis'].append("🟠 HIGH ATTACHMENT RISK - Priority investigation recommended")
            explanations['attachment_analysis'].append(f"Risk Score: {attachment_risk_score}/100 - Significant security concerns detected")
        elif attachment_risk_level == 'Medium Risk':
            explanations['attachment_analysis'].append("🟡 MEDIUM ATTACHMENT RISK - Review recommended")
            explanations['attachment_analysis'].append(f"Risk Score: {attachment_risk_score}/100 - Some concerning patterns identified")
        elif attachment_risk_level == 'Low Risk':
            explanations['attachment_analysis'].append("🟢 LOW ATTACHMENT RISK - Minimal security concerns")
            explanations['attachment_analysis'].append(f"Risk Score: {attachment_risk_score}/100 - Generally safe but monitor")
        else:
            # Legacy classification for low-risk files
            if attachment_type == 'Personal':
                explanations['attachment_analysis'].append("⚠️ Personal attachments detected - may indicate data exfiltration risk.")
                explanations['attachment_analysis'].append("File names suggest personal documents (family photos, personal files, etc.)")
            elif attachment_type == 'Business':
                explanations['attachment_analysis'].append("✅ Business-related attachments detected - appropriate for corporate communication.")
                explanations['attachment_analysis'].append("File names suggest business documents (contracts, reports, invoices, etc.)")
            elif attachment_type == 'Mixed':
                explanations['attachment_analysis'].append("⚠️ Mixed attachment types - combination of business and personal files.")
                explanations['attachment_analysis'].append("Review individual files to ensure appropriate business use.")

        # Data exfiltration risk assessment
        if exfiltration_risk == 'High':
            explanations['attachment_analysis'].append("💀 HIGH DATA EXFILTRATION RISK - Contains patterns indicative of data theft")
        elif exfiltration_risk == 'Medium':
            explanations['attachment_analysis'].append("⚠️ MEDIUM DATA EXFILTRATION RISK - Monitor for unauthorized data transfer")
        elif exfiltration_risk == 'Low':
            explanations['attachment_analysis'].append("⚡ LOW DATA EXFILTRATION RISK - Some data transfer indicators present")

        # Specific risk factors
        if attachment_risk_factors:
            explanations['attachment_analysis'].append("📋 Risk Factors Identified:")
            for factor in attachment_risk_factors[:5]:  # Show top 5 risk factors
                explanations['attachment_analysis'].append(f"  • {factor}")

        # Malicious indicators
        if malicious_indicators:
            explanations['attachment_analysis'].append("🚨 Malicious Indicators:")
            for indicator in malicious_indicators[:3]:  # Show top 3 indicators
                explanations['attachment_analysis'].append(f"  • {indicator}")

    else:
        explanations['attachment_analysis'].append("📧 No attachments detected - text-only communication.")

    # Behavioral Patterns
    cluster = case_data.get('ml_cluster', -1)
    if cluster >= 0:
        explanations['behavioral_patterns'].append(f"Email belongs to communication cluster #{cluster}")
        explanations['behavioral_patterns'].append("This represents a group of emails with similar characteristics and patterns.")
    else:
        explanations['behavioral_patterns'].append("Email doesn't fit into any established communication pattern.")
        explanations['behavioral_patterns'].append("This uniqueness may indicate unusual or suspicious activity.")

    # Domain Analysis
    domain_classification = case_data.get('domain_classification', 'Unknown')
    if domain_classification == 'Personal':
        explanations['behavioral_patterns'].append("⚠️ Personal email domain used for business communication.")
        explanations['behavioral_patterns'].append("This may violate company policies and increase security risks.")
    elif domain_classification == 'Corporate':
        explanations['behavioral_patterns'].append("✅ Corporate email domain - appropriate for business use.")

    # Recommendations based on analysis
    if risk_level in ['Critical', 'High']:
        explanations['recommendations'].append("🔴 Immediate Action Required")
        explanations['recommendations'].append("• Review email content and attachments immediately")
        explanations['recommendations'].append("• Contact sender to verify legitimate business purpose")
        explanations['recommendations'].append("• Consider blocking similar communications until verified")
    elif risk_level == 'Medium':
        explanations['recommendations'].append("🟡 Monitor and Review")
        explanations['recommendations'].append("• Schedule review within 24 hours")
        explanations['recommendations'].append("• Document findings for pattern analysis")
    else:
        explanations['recommendations'].append("🟢 Standard Processing")
        explanations['recommendations'].append("• No immediate action required")
        explanations['recommendations'].append("• Continue routine monitoring")

    return explanations

@app.route('/api/case/<session_id>/<record_id>/update', methods=['POST'])
def update_case_status(session_id, record_id):
    """API endpoint to update case status"""
    session_manager = SessionManager()
    processed_data = session_manager.get_processed_data(session_id)

    if not processed_data:
        return jsonify({'error': 'Session not found'}), 404

    new_status = request.json.get('status')
    notes = request.json.get('notes', '')

    # Find and update the record - try multiple ID formats
    updated = False
    for i, record in enumerate(processed_data):
        # Check various possible record ID formats
        if (record.get('record_id') == record_id or 
            str(record.get('record_id')) == record_id or
            str(i) == record_id):
            record['status'] = new_status
            record['notes'] = notes
            record['updated_at'] = datetime.now().isoformat()
            updated = True

            # Also update in session cases data for proper tracking
            if new_status.lower() == 'escalate':
                session_manager.update_case_status(session_id, i, 'escalate')
                app.logger.info(f"Case {record_id} escalated in session {session_id}")

            break

    if updated:
        # Save updated data back to session
        session_manager.update_processed_data(session_id, processed_data)
        return jsonify({'success': True, 'message': 'Case updated successfully'})
    else:
        app.logger.error(f"Record ID {record_id} not found in session {session_id}")
        app.logger.error(f"Available record IDs: {[r.get('record_id') for r in processed_data]}")
        return jsonify({'error': 'Case not found'}), 404

@app.route('/api/case/<session_id>/<record_id>/escalate', methods=['POST'])
def escalate_case(session_id, record_id):
    """API endpoint specifically for escalating cases"""
    session_manager = SessionManager()
    processed_data = session_manager.get_processed_data(session_id)

    if not processed_data:
        return jsonify({'error': 'Session not found'}), 404

    # Find the record and escalate it
    updated = False
    for i, record in enumerate(processed_data):
        if (record.get('record_id') == record_id or 
            str(record.get('record_id')) == record_id or
            str(i) == record_id):

            # Update the session case status
            result = session_manager.update_case_status(session_id, i, 'escalate')

            if result['success']:
                # Also update the record itself
                record['status'] = 'Escalated'
                record['updated_at'] = datetime.now().isoformat()
                session_manager.update_processed_data(session_id, processed_data)

                app.logger.info(f"Successfully escalated case {record_id} (index {i}) in session {session_id}")
                return jsonify({
                    'success': True, 
                    'message': 'Case escalated successfully',
                    'redirect_url': url_for('escalation_dashboard', session_id=session_id)
                })
            else:
                return jsonify({'error': 'Failed to escalate case'}), 500

    app.logger.error(f"Record ID {record_id} not found in session {session_id}")
    return jsonify({'error': 'Case not found'}), 404

@app.route('/api/escalation/<session_id>/<record_id>/resolve', methods=['POST'])
def resolve_escalation(session_id, record_id):
    """API endpoint to resolve escalated cases"""
    session_manager = SessionManager()
    processed_data = session_manager.get_processed_data(session_id)

    if not processed_data:
        return jsonify({'error': 'Session not found'}), 404

    resolution = request.json.get('resolution')
    resolution_notes = request.json.get('resolution_notes', '')
    new_status = request.json.get('new_status', 'Resolved')

    # Find and update the record - try multiple ID formats
    updated = False
    for i, record in enumerate(processed_data):
        # Check various possible record ID formats
        if (record.get('record_id') == record_id or 
            str(record.get('record_id')) == record_id or
            str(i) == record_id):
            record['status'] = new_status
            record['resolution'] = resolution
            record['resolution_notes'] = resolution_notes
            record['resolved_at'] = datetime.now().isoformat()
            record['updated_at'] = datetime.now().isoformat()
            updated = True
            break

    if updated:
        # Save updated data back to session
        session_manager.update_processed_data(session_id, processed_data)
        return jsonify({'success': True, 'message': 'Escalation resolved successfully'})
    else:
        app.logger.error(f"Escalation ID {record_id} not found in session {session_id}")
        app.logger.error(f"Available record IDs: {[r.get('record_id') for r in processed_data]}")
        return jsonify({'error': 'Escalation not found'}), 404

@app.route('/api/escalation/<session_id>/<record_id>/generate-email', methods=['POST'])
def generate_escalation_email(session_id, record_id):
    """API endpoint to generate escalation email"""
    session_manager = SessionManager()
    processed_data = session_manager.get_processed_data(session_id)

    if not processed_data:
        return jsonify({'error': 'Session not found'}), 404

    # Find the record
    found_record = None
    record_index = None
    for i, record in enumerate(processed_data):
        if (record.get('record_id') == record_id or
            str(record.get('record_id')) == record_id or
            str(i) == record_id):
            found_record = record
            record_index = i
            break

    if not found_record:
        return jsonify({'error': 'Escalation not found'}), 404

    # Generate the draft email
    result = session_manager.generate_draft_email(session_id, record_index)

    if result['success']:
        return jsonify({
            'success': True,
            'draft': result['draft'],
            'message': 'Draft email generated successfully'
        })
    else:
        return jsonify({'error': result['error']}), 500

@app.route('/admin')
def admin():
    """Admin panel for managing whitelists and settings"""
    session_manager = SessionManager()
    domain_manager = DomainManager()
    whitelists = session_manager.get_whitelists()
    attachment_keywords = session_manager.get_attachmentkeywords()
    sessions = session_manager.get_all_sessions()
    domain_classifications = domain_manager.get_domains()

    # Debug logging to check domain classifications
    app.logger.info(f"Domain classifications loaded: {domain_classifications}")

    # Ensure we have default structure if data is missing
    if not domain_classifications or not isinstance(domain_classifications, dict):
        app.logger.warning("Domain classifications empty or invalid, initializing defaults")
        domain_manager.initialize_domains_file()
        domain_classifications = domain_manager.get_domains()

    # Ensure all required keys exist with proper structure
    required_keys = ['trusted', 'corporate', 'personal', 'public', 'suspicious']
    for key in required_keys:
        if key not in domain_classifications or not isinstance(domain_classifications[key], list):
            domain_classifications[key] = []

    # Additional debug logging
    app.logger.info(f"Final domain classifications structure: {domain_classifications}")
    app.logger.info(f"Trusted domains count: {len(domain_classifications.get('trusted', []))}")

    return render_template('admin.html',
                         whitelists=whitelists,
                         attachment_keywords=attachment_keywords,
                         sessions=sessions,
                         domain_classifications=domain_classifications)

@app.route('/rules')
def rules():
    """Rules management interface"""
    rules = RuleEngine.get_all_rules()
    return render_template('rules.html', rules=rules)

@app.route('/rules/create', methods=['POST'])
def create_rule():
    """Create a new rule and reprocess existing sessions"""
    try:
        rule_name = request.form.get('name', '').strip()

        # Check if rule name is provided
        if not rule_name:
            flash('Rule name is required.', 'error')
            return redirect(url_for('rules'))

        # Check for duplicate rule names to prevent accidental duplicates
        rule_engine = RuleEngine()
        existing_rules = rule_engine.get_all_rules()

        for existing_rule in existing_rules:
            if existing_rule.get('name', '').lower() == rule_name.lower():
                flash(f'A rule with the name "{rule_name}" already exists. Please choose a different name.', 'error')
                return redirect(url_for('rules'))

        rule_data = {
            'name': rule_name,
            'description': request.form.get('description', '').strip(),
            'conditions': json.loads(request.form.get('conditions', '[]')),
            'actions': json.loads(request.form.get('actions', '[]')),
            'priority': int(request.form.get('priority', 1)),
            'active': request.form.get('active') == 'on'
        }

        # Validate that conditions and actions are not empty
        if not rule_data['conditions']:
            flash('At least one condition is required for the rule.', 'error')
            return redirect(url_for('rules'))

        if not rule_data['actions']:
            flash('At least one action is required for the rule.', 'error')
            return redirect(url_for('rules'))

        result = rule_engine.create_rule(rule_data)

        if result['success']:
            # Automatically reprocess all existing sessions with the new rule
            reprocess_result = reprocess_all_sessions()

            if reprocess_result['success']:
                flash(f'Rule created successfully! Reprocessed {reprocess_result["sessions_count"]} sessions with new rule.', 'success')
            else:
                flash(f'Rule created, but some sessions failed to reprocess: {reprocess_result["error"]}', 'warning')
        else:
            flash(f'Error creating rule: {result["error"]}', 'error')

    except json.JSONDecodeError as e:
        flash('Invalid rule data format. Please check your conditions and actions.', 'error')
    except ValueError as e:
        flash('Invalid priority value. Please select a valid priority.', 'error')
    except Exception as e:
        flash(f'Error creating rule: {str(e)}', 'error')

    return redirect(url_for('rules'))

@app.route('/rules/<int:rule_id>/update', methods=['POST'])
def update_rule(rule_id):
    """Update an existing rule and reprocess existing sessions"""
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
        result = rule_engine.update_rule(rule_id, rule_data)

        if result['success']:
            # Automatically reprocess all existing sessions with the updated rule
            reprocess_result = reprocess_all_sessions()

            if reprocess_result['success']:
                flash(f'Rule updated successfully! Reprocessed {reprocess_result["sessions_count"]} sessions with updated rule.', 'success')
            else:
                flash(f'Rule updated, but some sessions failed to reprocess: {reprocess_result["error"]}', 'warning')
        else:
            flash(f'Error updating rule: {result["error"]}', 'error')

    except Exception as e:
        flash(f'Error updating rule: {str(e)}', 'error')

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

                # Log escalation for debugging
                app.logger.info(f"Escalated case {record_id} in session {session_id}")
        else:
            flash(f'Error {action}ing case: {result["error"]}', 'error')

    return redirect(url_for('dashboard', session_id=session_id))

@app.route('/admin/whitelist', methods=['POST'])
def update_whitelist():
    """Update whitelist domains and reprocess existing sessions"""
    domains = request.form.get('domains', '').split('\n')
    # Convert all domains to lowercase for consistency
    domains = [domain.strip().lower() for domain in domains if domain.strip()]

    session_manager = SessionManager()
    result = session_manager.update_whitelist(domains)

    if result['success']:
        # Automatically reprocess all existing sessions with the new whitelist
        reprocess_result = reprocess_all_sessions()

        if reprocess_result['success']:
            flash(f'Whitelist updated successfully! Reprocessed {reprocess_result["sessions_count"]} sessions with new whitelist.', 'success')
        else:
            flash(f'Whitelist updated, but some sessions failed to reprocess: {reprocess_result["error"]}', 'warning')
    else:
        flash(f'Error updating whitelist: {result["error"]}', 'error')

    return redirect(url_for('admin'))

@app.route('/admin/reprocess-sessions', methods=['POST'])
def manual_reprocess_sessions():
    """Manual endpoint to reprocess all sessions"""
    try:
        reprocess_result = reprocess_all_sessions()

        if reprocess_result['success']:
            flash(f'Successfully reprocessed {reprocess_result["sessions_count"]} sessions with current whitelist and rules.', 'success')
        else:
            flash(f'Some sessions failed to reprocess: {reprocess_result["error"]}', 'error')

    except Exception as e:
        flash(f'Error reprocessing sessions: {str(e)}', 'error')

    return redirect(url_for('admin'))

@app.route('/analytics/whitelist/<session_id>')
def whitelist_analysis(session_id):
    """Detailed whitelist analysis view"""
    session_manager = SessionManager()
    session_data = session_manager.get_session(session_id)
    if not session_data:
        flash('Session not found', 'error')
        return redirect(url_for('index'))

    # Get whitelist data
    whitelists = session_manager.get_whitelists()
    all_processed_data = session_manager.get_processed_data(session_id)

    # Calculate whitelist impact
    whitelist_stats = {
        'total_whitelisted_domains': len(whitelists.get('domains', [])),
        'filtered_emails': session_data.get('whitelist_filtered', 0),
        'analyzed_emails': len(all_processed_data) if all_processed_data else 0,
        'domains': whitelists.get('domains', [])
    }

    return render_template('whitelist_analysis.html',
                         session=session_data,
                         whitelist_stats=whitelist_stats)

@app.route('/analytics/time/<session_id>')
def time_analysis(session_id):
    """Detailed time analysis view"""
    session_manager = SessionManager()
    session_data = session_manager.get_session(session_id)
    if not session_data:
        flash('Session not found', 'error')
        return redirect(url_for('index'))

    all_processed_data = session_manager.get_processed_data(session_id)

    return render_template('time_analysis.html',
                         session=session_data,
                         processed_data=all_processed_data)

@app.route('/analytics/senders/<session_id>')
def sender_analysis(session_id):
    """Detailed sender analysis view"""
    session_manager = SessionManager()
    session_data = session_manager.get_session(session_id)
    if not session_data:
        flash('Session not found', 'error')
        return redirect(url_for('index'))

    all_processed_data = session_manager.get_processed_data(session_id)

    return render_template('sender_analysis.html',
                         session=session_data,
                         processed_data=all_processed_data)

@app.route('/admin/attachment-keywords', methods=['POST'])
def update_attachment_keywords():
    """Update attachment analysis keywords"""
    business_keywords = request.form.get('business_keywords', '').split('\n')
    personal_keywords = request.form.get('personal_keywords', '').split('\n')
    suspicious_keywords = request.form.get('suspicious_keywords', '').split('\n')

    # Clean and filter keywords
    business_keywords = [kw.strip().lower() for kw in business_keywords if kw.strip()]
    personal_keywords = [kw.strip().lower() for kw in personal_keywords if kw.strip()]
    suspicious_keywords = [kw.strip().lower() for kw in suspicious_keywords if kw.strip()]

    keywords_data = {
        'business_keywords': business_keywords,
        'personal_keywords': personal_keywords,
        'suspicious_keywords': suspicious_keywords
    }

    session_manager = SessionManager()
    result = session_manager.update_attachment_keywords(keywords_data)

    if result['success']:
        flash('Attachment keywords updated successfully', 'success')
    else:
        flash(f'Error updating attachment keywords: {result["error"]}', 'error')

    return redirect(url_for('admin'))

@app.route('/delete_session/<session_id>', methods=['POST'])
def delete_session(session_id):
    """Delete a processing session"""
    try:
        session_manager = SessionManager()
        result = session_manager.delete_session(session_id)

        if result['success']:
            app.logger.info(f'Session {session_id} deleted successfully')
            # Use a simple return instead of flash to avoid session issues
            return redirect(url_for('admin'))
        else:
            app.logger.error(f'Error deleting session: {result["error"]}')
            return redirect(url_for('admin'))
    except Exception as e:
        app.logger.error(f'Unexpected error deleting session {session_id}: {str(e)}')
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

    # Add BAU analysis with proper structure
    insights['bau_analysis'] = {
        'bau_candidates_count': len(insights.get('bau_analysis', {}).get('bau_candidates', [])),
        'bau_percentage': insights.get('bau_analysis', {}).get('bau_percentage', 0),
        'high_volume_pairs': insights.get('bau_analysis', {}).get('high_volume_pairs', [])[:10],
        'bau_candidates': insights.get('bau_analysis', {}).get('bau_candidates', []),
        'unique_domains': insights.get('bau_analysis', {}).get('unique_domains', 0),
        'recommendations': insights.get('bau_analysis', {}).get('recommendations', [])
    }
    return jsonify(insights)

@app.route('/api/session_stats/<session_id>')
def api_session_stats(session_id):
    """API endpoint for session statistics (for Chart.js)"""
    session_manager = SessionManager()
    stats = session_manager.get_session_stats(session_id)
    return jsonify(stats)

@app.route('/api/debug_data/<session_id>')
def debug_data(session_id):
    """Debug endpoint to show raw data values"""
    try:
        session_manager = SessionManager()
        processed_data = session_manager.get_processed_data(session_id)

        if not processed_data:
            return jsonify({'error': 'No data found'}), 404

        # Show first few records with attachment data
        debug_info = []
        for i, record in enumerate(processed_data[:5]):  # Show first 5 records
            debug_info.append({
                'record_index': i,
                'sender': record.get('sender', 'N/A'),
                'subject': record.get('subject', 'N/A'),
                'attachments_raw': record.get('attachments', 'N/A'),
                'attachments_type': str(type(record.get('attachments', ''))),
                'has_attachments': record.get('has_attachments', False),
                'attachment_classification': record.get('attachment_classification', 'N/A'),
                'ml_risk_level': record.get('ml_risk_level', 'N/A')
            })

        # Also show attachment keywords
        keywords_data = session_manager.get_attachment_keywords()

        return jsonify({
            'debug_records': debug_info,
            'business_keywords': keywords_data.get('business_keywords', [])[:10],
            'personal_keywords': keywords_data.get('personal_keywords', [])[:10],
            'total_records': len(processed_data)
        })

    except Exception as e:
        app.logger.error(f"Error getting debug data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/processing_errors/<session_id>')
def get_processing_errors(session_id):
    """API endpoint to get detailed processing errors for case management"""
    try:
        data_processor = DataProcessor()
        error_details = data_processor.get_processing_errors(session_id)

        return jsonify(error_details)

    except Exception as e:
        app.logger.error(f"Error getting processing errors for session {session_id}: {str(e)}")
        return jsonify({
            'errors': [{'error_type': 'api_error', 'error': str(e), 'field': 'system', 'value': 'N/A'}],
            'processing_failed': True,
            'total_errors': 1
        }), 500

@app.route('/api/advanced_ml_analysis/<session_id>')
def advanced_ml_analysis(session_id):
    """Advanced ML analysis with justification sentiment, domain behavior, and communication patterns"""
    try:
        session_manager = SessionManager()
        processed_data = session_manager.get_processed_data(session_id)
        
        if not processed_data:
            return jsonify({'error': 'Session not found'}), 404
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(processed_data)
        
        if df.empty:
            return jsonify({'error': 'No data available'}), 404
        
        # Run comprehensive ML analysis
        advanced_ml_engine = AdvancedMLEngine()
        analysis_results = advanced_ml_engine.analyze_comprehensive_email_data(df)
        
        # Generate insights report
        insights_report = advanced_ml_engine.generate_ml_insights_report(analysis_results)
        
        return jsonify({
            'analysis_results': analysis_results,
            'insights_report': insights_report,
            'total_records_analyzed': len(df),
            'analysis_timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error in advanced ML analysis: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/justification_analysis/<session_id>')
def justification_analysis(session_id):
    """Detailed analysis of justification field content"""
    try:
        session_manager = SessionManager()
        processed_data = session_manager.get_processed_data(session_id)
        
        if not processed_data:
            return jsonify({'error': 'Session not found'}), 404
        
        df = pd.DataFrame(processed_data)
        
        if 'justification' not in df.columns:
            return jsonify({'error': 'No justification field found'}), 404
        
        advanced_ml_engine = AdvancedMLEngine()
        justification_results = advanced_ml_engine._analyze_justifications(df)
        
        return jsonify({
            'justification_analysis': justification_results,
            'total_justifications': len(df),
            'analysis_timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error in justification analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recipient_domain_intelligence/<session_id>')
def recipient_domain_intelligence(session_id):
    """Advanced recipient domain behavior analysis"""
    try:
        session_manager = SessionManager()
        processed_data = session_manager.get_processed_data(session_id)
        
        if not processed_data:
            return jsonify({'error': 'Session not found'}), 404
        
        df = pd.DataFrame(processed_data)
        
        advanced_ml_engine = AdvancedMLEngine()
        domain_results = advanced_ml_engine._analyze_recipient_domains(df)
        
        return jsonify({
            'domain_intelligence': domain_results,
            'total_records': len(df),
            'analysis_timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error in domain intelligence: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/communication_patterns/<session_id>')
def communication_patterns_analysis(session_id):
    """Email communication patterns and network analysis"""
    try:
        session_manager = SessionManager()
        processed_data = session_manager.get_processed_data(session_id)
        
        if not processed_data:
            return jsonify({'error': 'Session not found'}), 404
        
        df = pd.DataFrame(processed_data)
        
        advanced_ml_engine = AdvancedMLEngine()
        comm_results = advanced_ml_engine._analyze_communication_patterns(df)
        network_results = advanced_ml_engine._analyze_email_networks(df)
        
        return jsonify({
            'communication_patterns': comm_results,
            'network_analysis': network_results,
            'total_records': len(df),
            'analysis_timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error in communication patterns analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/temporal_anomalies/<session_id>')
def temporal_anomalies_analysis(session_id):
    """Temporal pattern analysis and anomaly detection"""
    try:
        session_manager = SessionManager()
        processed_data = session_manager.get_processed_data(session_id)
        
        if not processed_data:
            return jsonify({'error': 'Session not found'}), 404
        
        df = pd.DataFrame(processed_data)
        
        advanced_ml_engine = AdvancedMLEngine()
        temporal_results = advanced_ml_engine._analyze_temporal_patterns(df)
        
        return jsonify({
            'temporal_analysis': temporal_results,
            'total_records': len(df),
            'analysis_timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error in temporal analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/behavioral_clustering/<session_id>')
def behavioral_clustering_analysis(session_id):
    """User behavioral clustering and pattern recognition"""
    try:
        session_manager = SessionManager()
        processed_data = session_manager.get_processed_data(session_id)
        
        if not processed_data:
            return jsonify({'error': 'Session not found'}), 404
        
        df = pd.DataFrame(processed_data)
        
        advanced_ml_engine = AdvancedMLEngine()
        clustering_results = advanced_ml_engine._cluster_user_behavior(df)
        
        return jsonify({
            'behavioral_clustering': clustering_results,
            'total_records': len(df),
            'analysis_timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error in behavioral clustering: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/bau_analysis/<session_id>')
def bau_analysis(session_id):
    """API endpoint for BAU (Business As Usual) analysis"""
    try:
        session_manager = SessionManager()
        result = session_manager.get_processed_data(session_id, page=1, per_page=999999)
        processed_data = result.get('data', [])

        if not processed_data:
            return jsonify({
                'error': 'No data found',
                'bau_candidates': [],
                'bau_percentage': 0,
                'high_volume_pairs': [],
                'unique_domains': 0,
                'recommendations': ['No processed data available for this session']
            }), 404

        # Debug logging
        app.logger.info(f"BAU Analysis for session {session_id}: {len(processed_data)} records")

        # Sample a few records for debugging
        if len(processed_data) > 0:
            sample_record = processed_data[0]
            app.logger.info(f"Sample record keys: {list(sample_record.keys()) if isinstance(sample_record, dict) else 'Not a dict'}")
            app.logger.info(f"Sample sender: {sample_record.get('sender', 'N/A') if isinstance(sample_record, dict) else 'N/A'}")
            app.logger.info(f"Sample recipients: {sample_record.get('recipients', 'N/A') if isinstance(sample_record, dict) else 'N/A'}")



        # Convert to DataFrame for BAU analysis
        df = pd.DataFrame(processed_data)
        app.logger.info(f"DataFrame shape: {df.shape}")
        app.logger.info(f"DataFrame columns: {list(df.columns)}")

        ml_engine = MLEngine()
        bau_results = ml_engine.detect_bau_emails(df)

        # Add debug information
        bau_results['debug_info'] = {
            'total_records': len(processed_data),
            'dataframe_shape': df.shape,
            'has_sender_column': 'sender' in df.columns,
            'has_recipients_column': 'recipients' in df.columns,
            'session_id': session_id
        }

        app.logger.info(f"BAU Results: {bau_results.get('bau_percentage', 0)}% BAU, {len(bau_results.get('bau_candidates', []))} candidates")

        return jsonify(bau_results)

    except Exception as e:
        app.logger.error(f"Error getting BAU analysis for session {session_id}: {str(e)}")
        import traceback
        app.logger.error(f"BAU analysis traceback: {traceback.format_exc()}")
        return jsonify({
            'error': str(e),
            'bau_candidates': [],
            'bau_percentage': 0,
            'high_volume_pairs': [],
            'unique_domains': 0,
            'recommendations': [f'Error in BAU analysis: {str(e)}']
        }), 500

@app.route('/api/attachment_risk_analytics/<session_id>')
def attachment_risk_analytics(session_id):
    """API endpoint for detailed attachment risk analytics"""
    try:
        session_manager = SessionManager()
        result = session_manager.get_processed_data(session_id, page=1, per_page=999999)
        processed_data = result.get('data', [])

        app.logger.info(f"Processing attachment risk analytics for session {session_id} with {len(processed_data) if processed_data else 0} records")

        # Initialize analytics structure
        risk_analytics = {
            'total_attachments': 0,
            'total_emails_with_attachments': 0,
            'risk_distribution': {},
            'top_risk_factors': {},
            'malicious_indicators': [],
            'exfiltration_threats': [],
            'high_risk_emails': [],
            'average_risk_score': 0.0,
            'critical_risk_count': 0,
            'high_risk_count': 0,
            'medium_risk_count': 0,
            'low_risk_count': 0
        }

        if not processed_data:
            app.logger.warning(f"No processed data found for session {session_id}")
            return jsonify(risk_analytics)

        total_risk_score = 0
        risk_score_count = 0

        for record in processed_data:
            if record is None:
                continue
                
            # Check if email has attachments
            has_attachments = record.get('has_attachments', False)
            if has_attachments:
                risk_analytics['total_emails_with_attachments'] += 1
                
                # Count total attachments
                attachment_count = record.get('attachment_count', 1)
                if isinstance(attachment_count, (int, float)) and attachment_count > 0:
                    risk_analytics['total_attachments'] += int(attachment_count)
                else:
                    risk_analytics['total_attachments'] += 1

                # Risk level distribution
                risk_level = record.get('attachment_risk_level', 'Low Risk')
                if risk_level:
                    risk_analytics['risk_distribution'][risk_level] = risk_analytics['risk_distribution'].get(risk_level, 0) + 1
                    
                    # Count by simplified categories
                    if 'Critical' in risk_level:
                        risk_analytics['critical_risk_count'] += 1
                    elif 'High' in risk_level:
                        risk_analytics['high_risk_count'] += 1
                    elif 'Medium' in risk_level:
                        risk_analytics['medium_risk_count'] += 1
                    else:
                        risk_analytics['low_risk_count'] += 1

                # Aggregate risk factors
                risk_factors = record.get('attachment_risk_factors', [])
                if isinstance(risk_factors, list):
                    for factor in risk_factors:
                        if factor and isinstance(factor, str):
                            risk_analytics['top_risk_factors'][factor] = risk_analytics['top_risk_factors'].get(factor, 0) + 1

                # Collect malicious indicators
                malicious_indicators = record.get('attachment_malicious_indicators', [])
                if isinstance(malicious_indicators, list):
                    risk_analytics['malicious_indicators'].extend([ind for ind in malicious_indicators if ind])

                # Track risk scores
                risk_score = record.get('attachment_risk_score', 0)
                if isinstance(risk_score, (int, float)) and risk_score > 0:
                    total_risk_score += risk_score
                    risk_score_count += 1

                    # Track high-risk emails (score >= 50)
                    if risk_score >= 50:
                        subject = record.get('subject', '')
                        if len(subject) > 50:
                            subject = subject[:47] + '...'
                            
                        risk_analytics['high_risk_emails'].append({
                            'record_id': record.get('record_id', ''),
                            'sender': record.get('sender', ''),
                            'subject': subject,
                            'risk_score': risk_score,
                            'risk_level': risk_level,
                            'exfiltration_risk': record.get('attachment_exfiltration_risk', 'Unknown')
                        })

                # Track exfiltration threats
                exfiltration_risk = record.get('attachment_exfiltration_risk', 'None')
                if exfiltration_risk in ['High', 'Medium']:
                    risk_analytics['exfiltration_threats'].append({
                        'sender': record.get('sender', ''),
                        'risk_level': exfiltration_risk,
                        'risk_score': risk_score,
                        'attachments': record.get('attachments', '')
                    })

        # Calculate average risk score
        if risk_score_count > 0:
            risk_analytics['average_risk_score'] = round(total_risk_score / risk_score_count, 2)

        # Sort and limit results
        risk_analytics['top_risk_factors'] = dict(sorted(
            risk_analytics['top_risk_factors'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])

        risk_analytics['high_risk_emails'] = sorted(
            risk_analytics['high_risk_emails'],
            key=lambda x: x.get('risk_score', 0),
            reverse=True
        )[:20]

        risk_analytics['exfiltration_threats'] = sorted(
            risk_analytics['exfiltration_threats'],
            key=lambda x: x.get('risk_score', 0),
            reverse=True
        )[:10]

        # Remove duplicates from malicious indicators
        risk_analytics['malicious_indicators'] = list(set(risk_analytics['malicious_indicators']))[:15]

        app.logger.info(f"Attachment risk analytics completed: {risk_analytics['total_emails_with_attachments']} emails with attachments, {risk_analytics['critical_risk_count']} critical, {risk_analytics['high_risk_count']} high risk")

        return jsonify(risk_analytics)

    except Exception as e:
        app.logger.error(f"Error getting attachment risk analytics for session {session_id}: {str(e)}")
        import traceback
        app.logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return empty structure on error
        return jsonify({
            'total_attachments': 0,
            'total_emails_with_attachments': 0,
            'risk_distribution': {},
            'top_risk_factors': {},
            'malicious_indicators': [],
            'exfiltration_threats': [],
            'high_risk_emails': [],
            'average_risk_score': 0.0,
            'critical_risk_count': 0,
            'high_risk_count': 0,
            'medium_risk_count': 0,
            'low_risk_count': 0,
            'error': str(e)
        })

@app.route('/api/processing_status/<session_id>')
def get_processing_status(session_id):
    """API endpoint to check processing status for large files"""
    try:
        session_manager = SessionManager()
        session_data = session_manager.get_session(session_id)

        if not session_data:
            return jsonify({'error': 'Session not found'}), 404

        processed_data = session_manager.get_processed_data(session_id)

        status = {
            'session_id': session_id,
            'total_records': session_data.get('total_records', 0),
            'processed_records': len(processed_data) if processed_data else 0,
            'processing_complete': len(processed_data) > 0 if processed_data else False,
            'whitelist_filtered': session_data.get('whitelist_filtered', 0),
            'escalated_records': session_data.get('escalated_records', 0),
            'case_management_records': session_data.get('case_management_records', 0),
            'processing_steps': session_data.get('processing_stats', {}).get('processing_steps', []),
            'created_at': session_data.get('created_at'),
            'updated_at': session_data.get('updated_at')
        }

        return jsonify(status)

    except Exception as e:
        app.logger.error(f"Error getting processing status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/whitelist_bau/<session_id>', methods=['POST'])
def whitelist_bau_domains(session_id):
    """API endpoint to whitelist BAU domain pairs"""
    try:
        domain_pairs = request.json.get('domain_pairs', [])

        if not domain_pairs:
            return jsonify({'error': 'No domain pairs provided'}), 400

        # Extract recipient domains from domain pairs
        recipient_domains = []
        for pair in domain_pairs:
            # Format: "sender_domain -> recipient_domain"
            if ' -> ' in pair:
                recipient_domain = pair.split(' -> ')[1].strip()
                recipient_domains.append(recipient_domain)

        # Update whitelist
        session_manager = SessionManager()
        current_whitelist = session_manager.get_whitelists()
        existing_domains = set(current_whitelist.get('domains', []))

        # Add new domains
        new_domains = set(recipient_domains)
        updated_domains = list(existing_domains.union(new_domains))

        result = session_manager.update_whitelist(updated_domains)

        if result['success']:
            return jsonify({
                'success': True,
                'message': f'Added {len(new_domains)} new domains to whitelist',
                'new_domains': list(new_domains),
                'total_domains': len(updated_domains)
            })
        else:
            return jsonify({'error': result['error']}), 500

    except Exception as e:
        app.logger.error(f"Error whitelisting BAU domains: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reprocess_rules/<session_id>', methods=['POST'])
def reprocess_rules(session_id):
    """Reprocess existing session data with proper escalation logic"""
    try:
        # Use the new reprocessing method that handles escalation correctly
        data_processor = DataProcessor()
        result = data_processor.reprocess_existing_session(session_id)

        if result['success']:
            return jsonify({
                'success': True,
                'message': f'Successfully reprocessed session with {result["escalated_count"]} escalations and {result["case_management_count"]} case management records',
                'escalated_count': result['escalated_count'],
                'case_management_count': result['case_management_count'],
                'total_processed': result['total_processed']
            })
        else:
            return jsonify({'error': result['error']}), 500

    except Exception as e:
        app.logger.error(f"Error reprocessing session {session_id}: {str(e)}")
        return jsonify({'error': f'Reprocessing failed: {str(e)}'}), 500

@app.route('/api/whitelist_status')
def get_whitelist_status():
    """API endpoint to check when whitelist was last updated"""
    try:
        session_manager = SessionManager()
        whitelists = session_manager.get_whitelists()
        return jsonify({
            'updated_at': whitelists.get('updated_at', ''),
            'domain_count': len(whitelists.get('domains', []))
        })
    except Exception as e:
        app.logger.error(f"Error getting whitelist status: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Domain Management API Routes
@app.route('/admin/domains/add', methods=['POST'])
def add_domain():
    """API endpoint to add a domain to a classification category"""
    try:
        data = request.json
        category = data.get('category')
        domain = data.get('domain')

        if not category or not domain:
            return jsonify({'success': False, 'error': 'Category and domain are required'}), 400

        domain_manager = DomainManager()
        result = domain_manager.add_domain(category, domain)

        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400

    except Exception as e:
        app.logger.error(f"Error adding domain: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/domains/remove', methods=['POST'])
def remove_domain():
    """API endpoint to remove a domain from a classification category"""
    try:
        data = request.json
        category = data.get('category')
        domain = data.get('domain')

        if not category or not domain:
            return jsonify({'success': False, 'error': 'Category and domain are required'}), 400

        domain_manager = DomainManager()
        result = domain_manager.remove_domain(category, domain)

        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400

    except Exception as e:
        app.logger.error(f"Error removing domain: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/domains/export')
def export_domains():
    """Export domain classifications as JSON file"""
    try:
        domain_manager = DomainManager()
        config = domain_manager.export_config()

        from flask import make_response
        response = make_response(json.dumps(config, indent=2))
        response.headers['Content-Type'] = 'application/json'
        response.headers['Content-Disposition'] = 'attachment; filename=domain_classifications.json'

        return response

    except Exception as e:
        app.logger.error(f"Error exporting domain config: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/domains/reprocess', methods=['POST'])
def reprocess_sessions_with_domains():
    """Reprocess all sessions with current domain classifications"""
    try:
        session_manager = SessionManager()
        data_processor = DataProcessor()

        sessions = session_manager.get_all_sessions()
        sessions_updated = 0
        failed_sessions = []

        for session in sessions:
            session_id = session.get('session_id')
            if not session_id:
                continue

            try:
                # Reprocess the session with new domain classifications
                result = data_processor.reprocess_existing_session(session_id)
                if result['success']:
                    sessions_updated += 1
                else:
                    failed_sessions.append(session_id)

            except Exception as e:
                app.logger.error(f"Error reprocessing session {session_id}: {str(e)}")
                failed_sessions.append(session_id)

        if failed_sessions:
            return jsonify({
                'success': False,
                'error': f'Failed to reprocess sessions: {", ".join(failed_sessions)}',
                'sessions_updated': sessions_updated
            }), 500
        else:
            return jsonify({
                'success': True,
                'sessions_updated': sessions_updated
            })

    except Exception as e:
        app.logger.error(f"Error reprocessing sessions: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/domains/reset', methods=['POST'])
def reset_domains():
    """Reset domain classifications to default values"""
    try:
        domain_manager = DomainManager()
        result = domain_manager.reset_to_defaults()

        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500

    except Exception as e:
        app.logger.error(f"Error resetting domains: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Exclusion Rules API Routes
@app.route('/api/exclusion-rules', methods=['GET'])
def get_exclusion_rules():
    """Get all exclusion rules"""
    try:
        rule_engine = RuleEngine()
        exclusion_rules = rule_engine.get_all_exclusion_rules()
        field_names = RuleEngine.get_field_names()

        return jsonify({
            'success': True,
            'exclusion_rules': exclusion_rules,
            'field_names': field_names
        })

    except Exception as e:
        app.logger.error(f"Error getting exclusion rules: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/exclusion-rules', methods=['POST'])
def add_exclusion_rule():
    """Add new exclusion rule"""
    try:
        data = request.json
        rule_engine = RuleEngine()

        # Handle both single condition (backward compatibility) and multiple conditions
        conditions = data.get('conditions', [])
        if not conditions and 'field' in data:
            # Single condition format for backward compatibility
            conditions = [{
                'field': data.get('field'),
                'operator': data.get('operator'),
                'value': data.get('value'),
                'logic': 'AND'
            }]

        result = rule_engine.add_exclusion_rule(
            name=data.get('name'),
            description=data.get('description'),
            conditions=conditions,
            case_sensitive=data.get('case_sensitive', False)
        )

        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400

    except Exception as e:
        app.logger.error(f"Error adding exclusion rule: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/exclusion-rules/<int:rule_id>', methods=['DELETE'])
def delete_exclusion_rule(rule_id):
    """Delete exclusion rule"""
    try:
        rule_engine = RuleEngine()
        result = rule_engine.delete_exclusion_rule(rule_id)

        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400

    except Exception as e:
        app.logger.error(f"Error deleting exclusion rule: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/exclusion-rules/<int:rule_id>/toggle', methods=['POST'])
def toggle_exclusion_rule(rule_id):
    """Toggle exclusion rule active status"""
    try:
        rule_engine = RuleEngine()
        result = rule_engine.toggle_exclusion_rule(rule_id)

        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400

    except Exception as e:
        app.logger.error(f"Error toggling exclusion rule: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/exclusion-rules/<int:rule_id>', methods=['GET'])
def get_exclusion_rule(rule_id):
    """Get a specific exclusion rule by ID"""
    try:
        rule_engine = RuleEngine()
        result = rule_engine.get_exclusion_rule(rule_id)

        if result['success']:
            field_names = RuleEngine.get_field_names()
            result['field_names'] = field_names
            return jsonify(result)
        else:
            return jsonify(result), 404

    except Exception as e:
        app.logger.error(f"Error getting exclusion rule {rule_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/exclusion-rules/<int:rule_id>', methods=['PUT'])
def update_exclusion_rule_api(rule_id):
    """Update an existing exclusion rule"""
    try:
        data = request.json
        name = data.get('name', '')
        description = data.get('description', '')
        conditions = data.get('conditions', [])
        case_sensitive = data.get('case_sensitive', False)

        if not name.strip():
            return jsonify({'success': False, 'error': 'Rule name is required'}), 400

        if not conditions:
            return jsonify({'success': False, 'error': 'At least one condition is required'}), 400

        rule_engine = RuleEngine()
        result = rule_engine.update_exclusion_rule(rule_id, name, description, conditions, case_sensitive)

        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400

    except Exception as e:
        app.logger.error(f"Error updating exclusion rule {rule_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500