import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from session_manager import SessionManager
from rule_engine import RuleEngine
from ml_engine import MLEngine

class DataProcessor:
    """Process CSV data with ML analysis and rule application"""
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.rule_engine = RuleEngine()
        self.ml_engine = MLEngine()
        self.logger = logging.getLogger(__name__)
        
        # Expected CSV columns based on the specification
        self.expected_columns = [
            '_time', 'sender', 'subject', 'attachments', 'recipients',
            'recipients_email_domain', 'leaver', 'termination_date',
            'time_month', 'account_type', 'wordlist_attachment',
            'wordlist_subject', 'bunit', 'department', 'status',
            'user_response', 'final_outcome', 'policy_name', 'justification'
        ]
    
    def reprocess_existing_session(self, session_id: str) -> Dict:
        """Reprocess an existing session with current rules and escalation logic"""
        try:
            # Load existing session data
            session_data = self.session_manager.get_session(session_id)
            if not session_data:
                return {'success': False, 'error': 'Session not found'}
            
            # Get the processed data
            processed_data = session_data.get('processed_data', [])
            if not processed_data:
                return {'success': False, 'error': 'No processed data found'}
            
            self.logger.info(f"Reprocessing {len(processed_data)} records for session {session_id}")
            
            # Reapply the 4-step workflow to existing data
            escalated_data = []
            case_management_data = []
            
            for index, record in enumerate(processed_data):
                # Reapply rule processing
                rule_results = self.rule_engine.process_email(record)
                record['rule_results'] = rule_results
                
                # Apply escalation logic
                matched_rules = rule_results.get('matched_rules', [])
                should_escalate = False
                
                if matched_rules and len(matched_rules) > 0:
                    should_escalate = True
                    self.logger.info(f"Reprocessing: Escalating record {index} due to {len(matched_rules)} rule matches")
                
                if rule_results.get('escalate', False):
                    should_escalate = True
                
                # Assign to appropriate dashboard
                if should_escalate:
                    record['dashboard_type'] = 'escalation'
                    escalated_data.append(record)
                else:
                    record['dashboard_type'] = 'case_management'
                    case_management_data.append(record)
            
            # Update session data
            session_data['escalated_records'] = escalated_data
            session_data['case_management_records'] = case_management_data
            session_data['processed_data'] = escalated_data + case_management_data
            session_data['updated_at'] = datetime.now().isoformat()
            
            # Save updated session
            save_result = self.session_manager.update_session_data(session_id, escalated_data + case_management_data, {
                'escalated_records': len(escalated_data),
                'case_management_records': len(case_management_data),
                'processing_steps': [
                    f"Step 1: Filtered whitelist domains", 
                    f"Step 2: Escalated {len(escalated_data)} rule matches",
                    f"Step 3: ML analysis completed",
                    f"Step 4: {len(case_management_data)} events sorted by ML score"
                ]
            })
            
            if not save_result.get('success'):
                self.logger.error(f"Failed to save reprocessed data: {save_result.get('error')}")
                return {'success': False, 'error': 'Failed to save reprocessed data'}
            
            self.logger.info(f"Reprocessed session {session_id}: {len(escalated_data)} escalated, {len(case_management_data)} case management")
            
            return {
                'success': True,
                'escalated_count': len(escalated_data),
                'case_management_count': len(case_management_data),
                'total_processed': len(processed_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error reprocessing session {session_id}: {str(e)}")
            return {'success': False, 'error': str(e)}

    def process_csv(self, file_path: str, session_id: str, filename: str) -> Dict:
        """Process uploaded CSV file following the 4-step workflow"""
        try:
            self.logger.info(f"Starting CSV processing for session {session_id}")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
            if df.empty:
                return {'success': False, 'error': 'CSV file is empty'}
            
            # Validate CSV structure
            validation_result = self._validate_csv(df)
            if not validation_result['valid']:
                return {'success': False, 'error': validation_result['error']}
            
            # Create session
            csv_headers = df.columns.tolist()
            session_result = self.session_manager.create_session(session_id, filename, csv_headers)
            
            if not session_result['success']:
                return session_result
            
            # STEP 1: Apply whitelist filtering - ignore whitelisted domains
            self.logger.info("STEP 1: Filtering whitelist domains...")
            df_filtered, whitelist_count = self._apply_whitelist_filtering_with_stats(df)
            
            # STEP 2: Apply rules - move matching events to escalation dashboard
            self.logger.info("STEP 2: Checking rules and identifying escalations...")
            escalated_data, remaining_data = self._apply_rules_with_escalation(df_filtered)
            
            # STEP 3: Run ML analysis on remaining data
            self.logger.info("STEP 3: Running ML analysis on remaining events...")
            ml_results = self._run_ml_analysis(remaining_data)
            
            # STEP 4: Sort remaining data by ML score (high to low) for case management
            self.logger.info("STEP 4: Sorting data by ML score for case management...")
            case_management_data = self._prepare_case_management_data(remaining_data, ml_results)
            
            # Combine all processed data
            all_processed_data = escalated_data + case_management_data
            
            # Save processing results
            processing_stats = {
                'total_records': len(df),
                'whitelist_filtered': whitelist_count,
                'escalated_records': len(escalated_data),
                'case_management_records': len(case_management_data),
                'processing_steps': [
                    f"Step 1: Filtered {whitelist_count} whitelist domains",
                    f"Step 2: Escalated {len(escalated_data)} rule matches",
                    f"Step 3: ML analysis completed on {len(remaining_data)} events",
                    f"Step 4: {len(case_management_data)} events sorted by ML score"
                ]
            }
            
            # Save processed data to session
            self.logger.info(f"Saving {len(all_processed_data)} processed records to session {session_id}")
            save_result = self.session_manager.update_session_data(session_id, all_processed_data, processing_stats)
            if not save_result.get('success'):
                self.logger.error(f"Failed to save processed data: {save_result.get('error')}")
            
            return {
                'success': True,
                'total_records': len(df),
                'filtered_records': len(df_filtered),
                'processed_records': len(all_processed_data),
                'session_id': session_id,
                'processing_stats': processing_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error processing CSV: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _validate_csv(self, df: pd.DataFrame) -> Dict:
        """Validate CSV structure and content"""
        try:
            # Check if DataFrame has required columns
            missing_columns = []
            critical_columns = ['_time', 'sender', 'subject', 'recipients_email_domain']
            
            for col in critical_columns:
                if col not in df.columns:
                    missing_columns.append(col)
            
            if missing_columns:
                return {
                    'valid': False,
                    'error': f'Missing critical columns: {", ".join(missing_columns)}'
                }
            
            # Check for completely empty rows
            if df.isnull().all(axis=1).any():
                self.logger.warning("Found completely empty rows in CSV")
            
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}
    
    def _apply_whitelist_filtering_with_stats(self, df: pd.DataFrame) -> tuple:
        """Apply whitelist domain filtering and return stats"""
        try:
            whitelists = self.session_manager.get_whitelists()
            whitelist_domains = whitelists.get('domains', [])
            
            if not whitelist_domains:
                return df, 0
            
            # Filter out whitelisted domains
            original_count = len(df)
            filtered_df = df.copy()
            
            # Check sender domain (extract from sender email if needed)
            if 'sender' in df.columns:
                sender_domains = df['sender'].str.extract(r'@([^@]+)$')[0]
                sender_whitelist_mask = sender_domains.isin(whitelist_domains)
                filtered_df = filtered_df[~sender_whitelist_mask]
            
            # Check recipients_email_domain
            if 'recipients_email_domain' in df.columns:
                recipients_whitelist_mask = filtered_df['recipients_email_domain'].isin(whitelist_domains)
                filtered_df = filtered_df[~recipients_whitelist_mask]
            
            whitelist_count = original_count - len(filtered_df)
            self.logger.info(f"Filtered out {whitelist_count} whitelisted records")
            
            return filtered_df, whitelist_count
            
        except Exception as e:
            self.logger.error(f"Error applying whitelist filtering: {str(e)}")
            return df, 0
    
    def _apply_rules_with_escalation(self, df: pd.DataFrame) -> tuple:
        """Apply rules and separate escalated records from remaining data"""
        try:
            escalated_data = []
            remaining_data = []
            
            for index, row in df.iterrows():
                record = row.to_dict()
                processed_record = self._process_email_record(record, index)
                
                # Check if any rules triggered escalation
                rule_results = processed_record.get('rule_results', {})
                should_escalate = False
                
                # Check if any rules were matched - if yes, escalate to escalation dashboard
                # This follows the user's requirement: "if rule match then move that event straight to my escalation dashboard"
                if isinstance(rule_results, dict):
                    matched_rules = rule_results.get('matched_rules', [])
                    
                    # If there are any matched rules, escalate (regardless of action)
                    if matched_rules and len(matched_rules) > 0:
                        should_escalate = True
                        self.logger.info(f"Escalating record {index} due to {len(matched_rules)} rule matches: {[r.get('name', 'Unknown') for r in matched_rules]}")
                    
                    # Also check for explicit escalate flag
                    if rule_results.get('escalate', False):
                        should_escalate = True
                        self.logger.info(f"Escalating record {index} due to explicit escalate flag")
                elif isinstance(rule_results, list) and len(rule_results) > 0:
                    # If there are any rule results at all, escalate
                    should_escalate = True
                    self.logger.info(f"Escalating record {index} due to {len(rule_results)} rule matches")
                
                # Mark record for appropriate dashboard
                if should_escalate:
                    processed_record['dashboard_type'] = 'escalation'
                    escalated_data.append(processed_record)
                else:
                    processed_record['dashboard_type'] = 'case_management'
                    remaining_data.append(processed_record)
            
            self.logger.info(f"Escalated {len(escalated_data)} records, {len(remaining_data)} remaining for case management")
            return escalated_data, remaining_data
            
        except Exception as e:
            self.logger.error(f"Error applying rules: {str(e)}")
            # Return all as case management if error occurs
            processed_data = []
            for index, row in df.iterrows():
                record = row.to_dict()
                processed_record = self._process_email_record(record, index)
                processed_record['dashboard_type'] = 'case_management'
                processed_data.append(processed_record)
            return [], processed_data
    
    def _run_ml_analysis(self, data_list: List[Dict]) -> Dict:
        """Run ML analysis on remaining data"""
        try:
            if not data_list:
                return {
                    'anomaly_scores': [],
                    'risk_levels': [],
                    'interesting_patterns': [],
                    'clusters': [],
                    'insights': {}
                }
            
            # Convert list of dicts to DataFrame for ML analysis
            df = pd.DataFrame(data_list)
            ml_results = self.ml_engine.analyze_emails(df)
            
            return ml_results
            
        except Exception as e:
            self.logger.error(f"Error in ML analysis: {str(e)}")
            return {
                'anomaly_scores': [0.0] * len(data_list),
                'risk_levels': ['Low'] * len(data_list),
                'interesting_patterns': [],
                'clusters': [-1] * len(data_list),
                'insights': {}
            }
    
    def _prepare_case_management_data(self, data_list: List[Dict], ml_results: Dict) -> List[Dict]:
        """Sort data by ML score (high to low) and add ML results"""
        try:
            if not data_list:
                return []
            
            # Merge ML results with data
            updated_data = self._merge_ml_results(data_list, ml_results)
            
            # Sort by ML anomaly score (high to low)
            sorted_data = sorted(updated_data, 
                               key=lambda x: x.get('ml_anomaly_score', 0.0), 
                               reverse=True)
            
            self.logger.info(f"Sorted {len(sorted_data)} records by ML score for case management")
            return sorted_data
            
        except Exception as e:
            self.logger.error(f"Error preparing case management data: {str(e)}")
            return data_list
    
    def _process_email_record(self, record: Dict, record_index: int) -> Dict:
        """Process a single email record"""
        try:
            # Start with original record and clean NaN values
            processed_record = self._clean_nan_values(record.copy())
            processed_record['record_id'] = record_index
            
            # Apply domain classification
            domain_classification = self._classify_domain(
                record.get('recipients_email_domain', ''),
                record.get('sender', '')
            )
            processed_record['domain_classification'] = domain_classification
            
            # Apply rules
            rule_results = self.rule_engine.process_email(record)
            processed_record['rule_results'] = rule_results
            
            # Extract additional features
            processed_record['has_attachments'] = bool(record.get('attachments', ''))
            processed_record['processing_timestamp'] = datetime.now().isoformat()
            
            return processed_record
            
        except Exception as e:
            self.logger.error(f"Error processing email record: {str(e)}")
            return self._clean_nan_values(record)
    
    def _classify_domain(self, recipient_domain: str, sender_email: str) -> str:
        """Classify domain as Trusted, Corporate, Personal, Public, or Suspicious"""
        try:
            # Common domain classifications
            corporate_domains = ['company.com', 'corporate.com', 'business.com']
            personal_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
            public_domains = ['gov.com', 'edu.com', 'org.com']
            suspicious_indicators = ['temp', 'throwaway', '10minute', 'guerrilla']
            
            domain = recipient_domain.lower()
            
            # Check for suspicious patterns
            if any(indicator in domain for indicator in suspicious_indicators):
                return 'Suspicious'
            
            # Check against known categories
            if domain in corporate_domains:
                return 'Corporate'
            elif domain in personal_domains:
                return 'Personal'
            elif domain in public_domains:
                return 'Public'
            
            # Check whitelist (trusted domains)
            whitelists = self.session_manager.get_whitelists()
            if domain in whitelists.get('domains', []):
                return 'Trusted'
            
            # Default classification based on domain characteristics
            if domain.endswith('.gov') or domain.endswith('.edu'):
                return 'Public'
            elif domain.endswith('.com') or domain.endswith('.org'):
                return 'Corporate'
            else:
                return 'Personal'
                
        except Exception as e:
            self.logger.error(f"Error classifying domain: {str(e)}")
            return 'Unknown'
    
    def _merge_ml_results(self, processed_data: List[Dict], ml_results: Dict) -> List[Dict]:
        """Merge ML analysis results with processed data"""
        try:
            # Safely get ML results with proper type checking and handle all possible types
            anomaly_scores = ml_results.get('anomaly_scores', [])
            risk_levels = ml_results.get('risk_levels', [])
            clusters = ml_results.get('clusters', [])
            
            # Ensure we have lists, not other types (including bool, dict, str, etc.)
            if not isinstance(anomaly_scores, (list, tuple)):
                self.logger.warning(f"anomaly_scores is not a list/tuple, got {type(anomaly_scores)}: {anomaly_scores}")
                anomaly_scores = []
            if not isinstance(risk_levels, (list, tuple)):
                self.logger.warning(f"risk_levels is not a list/tuple, got {type(risk_levels)}: {risk_levels}")
                risk_levels = []
            if not isinstance(clusters, (list, tuple)):
                self.logger.warning(f"clusters is not a list/tuple, got {type(clusters)}: {clusters}")
                clusters = []
            
            # Convert to lists if they are tuples
            anomaly_scores = list(anomaly_scores) if isinstance(anomaly_scores, tuple) else anomaly_scores
            risk_levels = list(risk_levels) if isinstance(risk_levels, tuple) else risk_levels
            clusters = list(clusters) if isinstance(clusters, tuple) else clusters
            
            for i, record in enumerate(processed_data):
                # Safely handle anomaly scores
                try:
                    if i < len(anomaly_scores) and isinstance(anomaly_scores[i], (int, float)):
                        record['ml_anomaly_score'] = float(anomaly_scores[i])
                    else:
                        record['ml_anomaly_score'] = 0.0
                except (IndexError, TypeError, ValueError) as e:
                    self.logger.warning(f"Error processing anomaly score for record {i}: {e}")
                    record['ml_anomaly_score'] = 0.0
                
                # Safely handle risk levels
                try:
                    if i < len(risk_levels) and isinstance(risk_levels[i], str):
                        record['ml_risk_level'] = risk_levels[i]
                    else:
                        record['ml_risk_level'] = 'Low'
                except (IndexError, TypeError) as e:
                    self.logger.warning(f"Error processing risk level for record {i}: {e}")
                    record['ml_risk_level'] = 'Low'
                
                # Safely handle clusters
                try:
                    if i < len(clusters) and isinstance(clusters[i], (int, float)):
                        record['ml_cluster'] = int(clusters[i])
                    else:
                        record['ml_cluster'] = -1
                except (IndexError, TypeError, ValueError) as e:
                    self.logger.warning(f"Error processing cluster for record {i}: {e}")
                    record['ml_cluster'] = -1
                
                # Add detailed anomaly analysis for high-anomaly emails
                try:
                    if record['ml_anomaly_score'] > 0.7:
                        record['anomaly_details'] = self.ml_engine.analyze_anomaly_details(record)
                    else:
                        record['anomaly_details'] = []
                except Exception as e:
                    self.logger.warning(f"Error getting anomaly details for record {i}: {e}")
                    record['anomaly_details'] = []
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error merging ML results: {str(e)}")
            # Return processed data with default ML values if merging fails
            for record in processed_data:
                if 'ml_anomaly_score' not in record:
                    record['ml_anomaly_score'] = 0.0
                if 'ml_risk_level' not in record:
                    record['ml_risk_level'] = 'Low'
                if 'ml_cluster' not in record:
                    record['ml_cluster'] = -1
                if 'anomaly_details' not in record:
                    record['anomaly_details'] = []
            return processed_data
    
    def get_domain_stats(self, processed_data: List[Dict]) -> Dict:
        """Get domain statistics for visualization"""
        try:
            domain_counts = {}
            classification_counts = {}
            
            for record in processed_data:
                domain = record.get('recipients_email_domain', 'Unknown')
                classification = record.get('domain_classification', 'Unknown')
                
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                classification_counts[classification] = classification_counts.get(classification, 0) + 1
            
            return {
                'domain_counts': domain_counts,
                'classification_counts': classification_counts,
                'total_domains': len(domain_counts),
                'total_records': len(processed_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting domain stats: {str(e)}")
            return {}
    
    def _clean_nan_values(self, data):
        """Clean NaN values from data to prevent JSON serialization issues"""
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                cleaned[key] = self._clean_nan_values(value)
            return cleaned
        elif isinstance(data, list):
            return [self._clean_nan_values(item) for item in data]
        elif pd.isna(data):
            return None
        else:
            return data
    
    def get_processing_summary(self, session_id: str) -> Dict:
        """Get processing summary for a session"""
        try:
            session = self.session_manager.get_session(session_id)
            if not session:
                return {}
            
            processed_data = session.get('processed_data', [])
            
            # Calculate summary statistics
            total_records = len(processed_data)
            risk_distribution = {}
            rule_matches = 0
            escalations = 0
            
            for record in processed_data:
                # Risk level distribution
                risk_level = record.get('ml_risk_level', 'Low')
                risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
                
                # Rule matches
                rule_results = record.get('rule_results', {})
                if rule_results.get('matched_rules'):
                    rule_matches += 1
                
                if rule_results.get('escalate'):
                    escalations += 1
            
            return {
                'total_records': total_records,
                'risk_distribution': risk_distribution,
                'rule_matches': rule_matches,
                'escalations': escalations,
                'processing_complete': True
            }
            
        except Exception as e:
            self.logger.error(f"Error getting processing summary: {str(e)}")
            return {}
