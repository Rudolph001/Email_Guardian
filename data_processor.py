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

    def _has_valid_value(self, value):
        """Check if a value represents actual data or is a null indicator like '-'"""
        if pd.isna(value):
            return False
        str_value = str(value).strip()
        return str_value != '' and str_value != '-' and str_value.lower() != 'nan'

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

            # Check file size and read accordingly
            import os
            file_size = os.path.getsize(file_path)
            self.logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
            
            # For large files, use chunked reading
            if file_size > 50 * 1024 * 1024:  # 50MB threshold
                self.logger.info("Large file detected, using chunked processing")
                return self._process_large_csv(file_path, session_id, filename)

            # Read CSV file normally for smaller files
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
        """Apply rules - rule matches go to case management with Critical risk, only manual escalations go to escalation dashboard"""
        try:
            escalated_data = []  # This will be empty as we're not auto-escalating rule matches anymore
            remaining_data = []

            for index, row in df.iterrows():
                record = row.to_dict()
                processed_record = self._process_email_record(record, index)

                # Check if any rules were matched
                rule_results = processed_record.get('rule_results', {})
                has_rule_matches = False

                if isinstance(rule_results, dict):
                    matched_rules = rule_results.get('matched_rules', [])
                    if matched_rules and len(matched_rules) > 0:
                        has_rule_matches = True
                        self.logger.info(f"Record {index} has {len(matched_rules)} rule matches: {[r.get('name', 'Unknown') for r in matched_rules]}")
                elif isinstance(rule_results, list) and len(rule_results) > 0:
                    has_rule_matches = True
                    self.logger.info(f"Record {index} has {len(rule_results)} rule matches")

                # All records go to case management initially
                # Rule matches get Critical risk level to highlight them
                if has_rule_matches:
                    processed_record['ml_risk_level'] = 'Critical'  # Override risk level for rule matches
                    processed_record['rule_priority'] = True  # Flag for priority handling
                    self.logger.info(f"Setting Critical risk level for record {index} due to rule matches")

                processed_record['dashboard_type'] = 'case_management'
                remaining_data.append(processed_record)

            self.logger.info(f"All {len(remaining_data)} records sent to case management ({len([r for r in remaining_data if r.get('rule_priority')])} with rule matches)")
            return escalated_data, remaining_data  # escalated_data will be empty

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
            processed_record['has_attachments'] = self._has_valid_value(record.get('attachments', ''))
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
            attachment_classifications = ml_results.get('attachment_classifications', [])

            # Debug logging for attachment data
            self.logger.info(f"Debug: Processing {len(processed_data)} records for ML merge")
            for i, record in enumerate(processed_data[:3]):  # Show first 3 records
                attachments = record.get('attachments', '')
                self.logger.info(f"Debug: Record {i} - attachments field: '{attachments}' (type: {type(attachments)})")

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
            if not isinstance(attachment_classifications, (list, tuple)):
                self.logger.warning(f"attachment_classifications is not a list/tuple, got {type(attachment_classifications)}: {attachment_classifications}")
                attachment_classifications = []

            # Convert to lists if they are tuples
            anomaly_scores = list(anomaly_scores) if isinstance(anomaly_scores, tuple) else anomaly_scores
            risk_levels = list(risk_levels) if isinstance(risk_levels, tuple) else risk_levels
            clusters = list(clusters) if isinstance(clusters, tuple) else clusters
            attachment_classifications = list(attachment_classifications) if isinstance(attachment_classifications, tuple) else attachment_classifications

            # Debug logging for attachment classifications
            self.logger.info(f"Debug: ML attachment classifications: {attachment_classifications}")

            # Merge ML results with processed data
            for i, record in enumerate(processed_data):
                if i < len(anomaly_scores):
                    record['ml_anomaly_score'] = anomaly_scores[i]
                
                # Preserve Critical risk level for rule matches, otherwise use ML risk level
                if i < len(risk_levels):
                    # If record has rule_priority flag (meaning it matched rules), keep Critical risk level
                    if record.get('rule_priority') and record.get('ml_risk_level') == 'Critical':
                        # Keep the Critical level set during rule processing
                        self.logger.info(f"Preserving Critical risk level for record {i} due to rule matches")
                    else:
                        # Use ML-determined risk level for non-rule matches
                        record['ml_risk_level'] = risk_levels[i]
                
                if i < len(clusters):
                    record['ml_cluster'] = clusters[i]
                if i < len(attachment_classifications):
                    record['attachment_classification'] = attachment_classifications[i]
                    self.logger.info(f"Debug: Record {i} got attachment classification: {attachment_classifications[i]}")

                # Ensure we have a domain classification
                if 'domain_classification' not in record:
                    record['domain_classification'] = self._classify_domain(record.get('recipients_email_domain', ''))

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
                if 'attachment_classification' not in record:
                    record['attachment_classification'] = 'Unknown'
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

    def _process_large_csv(self, file_path: str, session_id: str, filename: str) -> Dict:
        """Process large CSV files in chunks"""
        try:
            chunk_size = 1000
            all_processed_data = []
            total_records = 0
            whitelist_count = 0
            
            # Read CSV headers first
            df_sample = pd.read_csv(file_path, nrows=1)
            csv_headers = df_sample.columns.tolist()
            
            # Validate CSV structure
            validation_result = self._validate_csv(df_sample)
            if not validation_result['valid']:
                return {'success': False, 'error': validation_result['error']}
            
            # Create session
            session_result = self.session_manager.create_session(session_id, filename, csv_headers)
            if not session_result['success']:
                return session_result
            
            self.logger.info(f"Processing large CSV in chunks of {chunk_size}")
            
            # Process file in chunks
            chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
            
            for chunk_num, chunk_df in enumerate(chunk_iter):
                self.logger.info(f"Processing chunk {chunk_num + 1}")
                
                total_records += len(chunk_df)
                
                # STEP 1: Apply whitelist filtering
                df_filtered, chunk_whitelist_count = self._apply_whitelist_filtering_with_stats(chunk_df)
                whitelist_count += chunk_whitelist_count
                
                # STEP 2: Apply rules
                escalated_data, remaining_data = self._apply_rules_with_escalation(df_filtered)
                
                # STEP 3 & 4: ML analysis and sorting (in smaller batches)
                if remaining_data:
                    ml_results = self._run_ml_analysis(remaining_data)
                    case_management_data = self._prepare_case_management_data(remaining_data, ml_results)
                    all_processed_data.extend(escalated_data + case_management_data)
            
            # Save processing results
            processing_stats = {
                'total_records': total_records,
                'whitelist_filtered': whitelist_count,
                'escalated_records': len([d for d in all_processed_data if d.get('dashboard_type') == 'escalation']),
                'case_management_records': len([d for d in all_processed_data if d.get('dashboard_type') == 'case_management']),
                'processing_steps': [
                    f"Step 1: Filtered {whitelist_count} whitelist domains",
                    f"Step 2-4: Processed {len(all_processed_data)} records in chunks"
                ]
            }
            
            # Save processed data to session
            self.logger.info(f"Saving {len(all_processed_data)} processed records to session {session_id}")
            save_result = self.session_manager.update_session_data(session_id, all_processed_data, processing_stats)
            
            return {
                'success': True,
                'total_records': total_records,
                'processed_records': len(all_processed_data),
                'session_id': session_id,
                'processing_stats': processing_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error processing large CSV: {str(e)}")
            return {'success': False, 'error': str(e)}

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