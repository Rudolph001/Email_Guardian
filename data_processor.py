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

    def _convert_dataframe_to_lowercase(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert all text data in DataFrame to lowercase, preserving column headers"""
        df_copy = df.copy()

        # Convert all string columns to lowercase
        for column in df_copy.columns:
            if df_copy[column].dtype == 'object':  # Object dtype usually indicates string data
                # Apply lowercase conversion only to non-null string values
                df_copy[column] = df_copy[column].apply(
                    lambda x: x.lower().strip() if isinstance(x, str) and pd.notna(x) else x
                )

        self.logger.info(f"Converted text data to lowercase for {len(df_copy.columns)} columns")
        return df_copy

    def _clean_record_data(self, record):
        """Clean record data by converting None values and '-' to proper defaults, and convert all text values to lowercase"""
        cleaned_record = {}
        for key, value in record.items():
            if value is None or (isinstance(value, str) and value.strip().lower() in ['', '-', 'nan', 'null', 'none']):
                # Provide appropriate defaults for common fields
                if key in ['subject', 'sender', 'recipients', 'department']:
                    cleaned_record[key] = 'n/a'  # Lowercase default
                elif key in ['has_attachments']:
                    cleaned_record[key] = False
                elif key in ['ml_anomaly_score', 'ml_risk_score']:
                    cleaned_record[key] = 0.0
                else:
                    cleaned_record[key] = 'n/a'  # Lowercase default
            else:
                # Convert ALL text values to lowercase, preserve numbers and booleans
                if isinstance(value, str):
                    cleaned_record[key] = value.lower().strip()
                elif isinstance(value, (int, float, bool)):
                    cleaned_record[key] = value
                else:
                    # Convert any other type to string and make lowercase
                    cleaned_record[key] = str(value).lower().strip()
        return cleaned_record

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

            # For large files, use chunked reading - reduced threshold to handle 10K+ records better
            if file_size > 10 * 1024 * 1024:  # 10MB threshold (approximately 5000+ records)
                self.logger.info(f"Large file detected ({file_size / (1024*1024):.2f} MB), using chunked processing")
                return self._process_large_csv(file_path, session_id, filename)

            # Read CSV file with error handling
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError as e:
                return {
                    'success': False, 
                    'error': f'File encoding error: Unable to read CSV file. Please ensure it\'s saved as UTF-8. Error: {str(e)}',
                    'error_type': 'encoding_error'
                }
            except pd.errors.EmptyDataError:
                return {
                    'success': False, 
                    'error': 'CSV file is empty or contains no data',
                    'error_type': 'empty_file'
                }
            except pd.errors.ParserError as e:
                return {
                    'success': False, 
                    'error': f'CSV parsing error: {str(e)}. Please check file format and ensure proper CSV structure.',
                    'error_type': 'parser_error'
                }

            if len(df) == 0:
                return {
                    'success': False, 
                    'error': 'CSV file contains no data rows',
                    'error_type': 'no_data'
                }

            # Convert all text data to lowercase (preserving column headers)
            self.logger.info("Converting all imported data values to lowercase...")
            df = self._convert_dataframe_to_lowercase(df)

            # Validate CSV structure with detailed error reporting
            validation_result = self._validate_csv(df)
            if not validation_result['valid']:
                return {
                    'success': False, 
                    'error': validation_result['error'],
                    'error_type': validation_result.get('error_type', 'validation_error'),
                    'validation_details': validation_result
                }

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
                return {'success': False, 'error': f'Failed to save processed data: {save_result.get("error", "Unknown error")}'}

            # Extract sample data for preview (first 10 records)
            sample_data = []
            if all_processed_data:
                sample_data = all_processed_data[:10]
                # Clean sample data for JSON serialization
                for record in sample_data:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None

            return {
                'success': True,
                'total_records': len(df),
                'filtered_records': len(df_filtered),
                'processed_records': len(all_processed_data),
                'session_id': session_id,
                'processing_stats': processing_stats,
                'sample_data': sample_data
            }

        except Exception as e:
            self.logger.error(f"Error processing CSV: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _validate_csv(self, df: pd.DataFrame) -> Dict:
        """Validate CSV structure and content with detailed error reporting"""
        try:
            validation_errors = []

            # Check if DataFrame has required columns
            missing_columns = []
            critical_columns = ['_time', 'sender', 'subject', 'recipients_email_domain']

            for col in critical_columns:
                if col not in df.columns:
                    missing_columns.append(col)

            if len(missing_columns) > 0:
                return {
                    'valid': False,
                    'error': f'Missing critical columns: {", ".join(missing_columns)}',
                    'error_type': 'missing_columns',
                    'missing_columns': missing_columns,
                    'available_columns': list(df.columns)
                }

            # Check for data type issues and invalid values
            for row_idx, row in df.head(100).iterrows():  # Check first 100 rows for validation
                try:
                    # Check _time field format
                    time_value = row.get('_time')
                    if pd.notna(time_value) and time_value != '-':
                        try:
                            pd.to_datetime(time_value)
                        except:
                            validation_errors.append({
                                'row': row_idx + 2,  # +2 because pandas is 0-indexed and we have header
                                'field': '_time',
                                'value': str(time_value)[:100],  # Limit value length
                                'error': 'Invalid date/time format'
                            })

                    # Check sender field format (should contain email)
                    sender_value = row.get('sender')
                    if pd.notna(sender_value) and sender_value != '-':
                        if '@' not in str(sender_value) and str(sender_value).strip() != '':
                            validation_errors.append({
                                'row': row_idx + 2,
                                'field': 'sender',
                                'value': str(sender_value)[:100],
                                'error': 'Sender field should contain email address'
                            })

                    # Check recipients_email_domain format
                    domain_value = row.get('recipients_email_domain')
                    if pd.notna(domain_value) and domain_value != '-':
                        domain_str = str(domain_value).strip()
                        if domain_str and ('.' not in domain_str or '@' in domain_str):
                            validation_errors.append({
                                'row': row_idx + 2,
                                'field': 'recipients_email_domain',
                                'value': str(domain_value)[:100],
                                'error': 'Should be domain only (e.g., "company.com"), not full email'
                            })

                    # Check for extremely long values that might cause issues
                    for col in df.columns:
                        cell_value = row.get(col)
                        if pd.notna(cell_value):
                            cell_str = str(cell_value)
                            if len(cell_str) > 10000:  # Increased limit to 10,000 characters
                                validation_errors.append({
                                    'row': row_idx + 2,
                                    'field': col,
                                    'value': cell_str[:100] + '...',
                                    'error': f'Value is too long ({len(cell_str)} characters) - consider splitting large content'
                                })

                except Exception as field_error:
                    validation_errors.append({
                        'row': row_idx + 2,
                        'field': 'unknown',
                        'value': 'N/A',
                        'error': f'Error processing row: {str(field_error)}'
                    })

            # Check for completely empty rows
            empty_rows = df.isnull().all(axis=1)
            if len(empty_rows) > 0 and empty_rows.any():
                empty_row_numbers = [idx + 2 for idx in empty_rows[empty_rows].index]
                self.logger.warning(f"Found completely empty rows at: {empty_row_numbers}")
                validation_errors.append({
                    'row': empty_row_numbers[0] if empty_row_numbers else 'unknown',
                    'field': 'all_fields',
                    'value': 'empty',
                    'error': f'Completely empty rows found at rows: {empty_row_numbers[:5]}'
                })

            # Return detailed errors if found
            if validation_errors:
                # Limit to first 10 errors to avoid overwhelming the user
                limited_errors = validation_errors[:10]
                error_summary = []
                for err in limited_errors:
                    error_summary.append(f"Row {err['row']}, Field '{err['field']}': {err['error']} (Value: '{err['value']}')")

                more_errors = len(validation_errors) - len(limited_errors)
                if more_errors > 0:
                    error_summary.append(f"... and {more_errors} more errors")

                return {
                    'valid': False,
                    'error': 'Data validation failed:\n' + '\n'.join(error_summary),
                    'error_type': 'data_validation',
                    'validation_errors': limited_errors,
                    'total_errors': len(validation_errors),
                    'detailed_errors': validation_errors  # Include all errors for detailed view
                }

            return {'valid': True}

        except Exception as e:
            return {
                'valid': False, 
                'error': f'Validation error: {str(e)}',
                'error_type': 'validation_exception'
            }

    def get_processing_errors(self, session_id: str) -> Dict:
        """Get detailed processing errors for a session"""
        try:
            session_data = self.session_manager.get_session(session_id)
            if not session_data:
                return {'errors': [], 'processing_failed': True, 'error': 'Session not found'}

            processing_errors = []
            processed_data = session_data.get('processed_data', [])

            # Check for records with processing errors
            for idx, record in enumerate(processed_data):
                if record and isinstance(record, dict):
                    # Check for various error fields
                    if 'processing_error' in record:
                        processing_errors.append({
                            'record_index': idx,
                            'error_type': 'processing_error',
                            'error': record['processing_error'],
                            'field': 'general',
                            'value': 'N/A'
                        })

                    if 'domain_error' in record:
                        processing_errors.append({
                            'record_index': idx,
                            'error_type': 'domain_classification_error',
                            'error': record['domain_error'],
                            'field': 'recipients_email_domain',
                            'value': record.get('recipients_email_domain', 'N/A')
                        })

                    if 'rule_error' in record:
                        processing_errors.append({
                            'record_index': idx,
                            'error_type': 'rule_processing_error',
                            'error': record['rule_error'],
                            'field': 'rule_processing',
                            'value': 'N/A'
                        })

                    if 'attachment_error' in record:
                        processing_errors.append({
                            'record_index': idx,
                            'error_type': 'attachment_processing_error',
                            'error': record['attachment_error'],
                            'field': 'attachments',
                            'value': record.get('attachments', 'N/A')
                        })

            return {
                'errors': processing_errors,
                'processing_failed': len(processing_errors) > 0,
                'total_errors': len(processing_errors)
            }

        except Exception as e:
            self.logger.error(f"Error getting processing errors: {str(e)}")
            return {
                'errors': [{'error_type': 'system_error', 'error': str(e), 'field': 'system', 'value': 'N/A'}],
                'processing_failed': True,
                'total_errors': 1
            }

    def _apply_whitelist_filtering_with_stats(self, df: pd.DataFrame) -> tuple:
        """Apply exclusion rules and whitelist domain filtering, return stats"""
        try:
            # Step 0: Apply exclusion rules first
            initial_count = len(df)
            filtered_by_exclusion = []
            exclusion_filtered = 0

            for index, row in df.iterrows():
                record = row.to_dict()

                # Check exclusion rules
                if self.rule_engine.should_exclude_record(record):
                    exclusion_filtered += 1
                    self.logger.debug(f"Excluded record {index} based on exclusion rules")
                    continue

                filtered_by_exclusion.append(record)

            self.logger.info(f"Excluded {exclusion_filtered} records based on exclusion rules")

            # Convert back to DataFrame for whitelist processing
            if not filtered_by_exclusion:
                return pd.DataFrame(), exclusion_filtered

            df_after_exclusion = pd.DataFrame(filtered_by_exclusion)

            # Step 1: Apply whitelist domain filtering
            whitelists = self.session_manager.get_whitelists()
            whitelist_domains = whitelists.get('domains', [])

            if not whitelist_domains:
                return df_after_exclusion, exclusion_filtered

            # Filter out whitelisted domains
            original_count = len(df_after_exclusion)
            filtered_df = df_after_exclusion.copy()

            # Check sender domain (extract from sender email if needed)
            if 'sender' in df_after_exclusion.columns:
                sender_domains = df_after_exclusion['sender'].str.extract(r'@([^@]+)$')[0]
                sender_whitelist_mask = sender_domains.isin(whitelist_domains)
                if sender_whitelist_mask.any():
                    filtered_df = filtered_df[~sender_whitelist_mask]

            # Check recipients_email_domain
            if 'recipients_email_domain' in df_after_exclusion.columns:
                recipients_whitelist_mask = filtered_df['recipients_email_domain'].isin(whitelist_domains)
                if recipients_whitelist_mask.any():
                    filtered_df = filtered_df[~recipients_whitelist_mask]

            whitelist_count = original_count - len(filtered_df)
            total_filtered = exclusion_filtered + whitelist_count

            self.logger.info(f"Filtered out {whitelist_count} whitelisted records, {total_filtered} total filtered")

            return filtered_df, total_filtered

        except Exception as e:
            self.logger.error(f"Error applying filtering: {str(e)}")
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
        """Process a single email record with detailed error handling"""
        try:
            # Start with original record and clean NaN values and None values
            processed_record = self._clean_nan_values(record.copy())
            processed_record = self._clean_record_data(processed_record)
            processed_record['record_id'] = record_index

            # Apply domain classification with error handling
            try:
                domain_classification = self._classify_domain(
                    record.get('recipients_email_domain', ''),
                    record.get('sender', '')
                )
                processed_record['domain_classification'] = domain_classification
            except Exception as domain_error:
                self.logger.error(f"Error classifying domain for record {record_index}: {str(domain_error)}")
                self.logger.error(f"Problematic values - recipients_email_domain: '{record.get('recipients_email_domain')}', sender: '{record.get('sender')}'")
                processed_record['domain_classification'] = 'Unknown'
                processed_record['domain_error'] = str(domain_error)

            # Apply rules with error handling
            try:
                rule_results = self.rule_engine.process_email(record)
                processed_record['rule_results'] = rule_results
            except Exception as rule_error:
                self.logger.error(f"Error applying rules for record {record_index}: {str(rule_error)}")
                self.logger.error(f"Problematic record data: {dict(list(record.items())[:5])}")  # Log first 5 fields
                processed_record['rule_results'] = {'matched_rules': [], 'escalate': False}
                processed_record['rule_error'] = str(rule_error)

            # Extract additional features with error handling
            try:
                processed_record['has_attachments'] = self._has_valid_value(record.get('attachments', ''))
            except Exception as attachment_error:
                self.logger.error(f"Error processing attachments for record {record_index}: {str(attachment_error)}")
                self.logger.error(f"Attachment value: '{record.get('attachments')}'")
                processed_record['has_attachments'] = False
                processed_record['attachment_error'] = str(attachment_error)

            processed_record['processing_timestamp'] = datetime.now().isoformat()

            return processed_record

        except Exception as e:
            self.logger.error(f"Critical error processing email record {record_index}: {str(e)}")
            self.logger.error(f"Record keys: {list(record.keys()) if isinstance(record, dict) else 'Not a dict'}")

            # Return a minimal safe record
            safe_record = self._clean_nan_values(record.copy()) if isinstance(record, dict) else {}
            safe_record = self._clean_record_data(safe_record)
            safe_record.update({
                'record_id': record_index,
                'processing_error': str(e),
                'processing_timestamp': datetime.now().isoformat(),
                'domain_classification': 'Unknown',
                'rule_results': {'matched_rules': [], 'escalate': False},
                'has_attachments': False
            })
            return safe_record

    def _classify_domain(self, recipient_domain: str, sender_email: str) -> str:
        """Classify domain using the new domain manager system"""
        try:
            from domain_manager import DomainManager
            domain_manager = DomainManager()

            # Use the domain manager for classification
            classification = domain_manager.classify_domain(recipient_domain)

            # Fall back to whitelist check if unknown
            if classification == 'Unknown':
                whitelists = self.session_manager.get_whitelists()
                if recipient_domain.lower() in whitelists.get('domains', []):
                    return 'Trusted'

            return classification

        except Exception as e:
            self.logger.error(f"Error classifying domain: {str(e)}")
            return 'Unknown'

    def _merge_ml_results(self, processed_data: List[Dict], ml_results: Dict) -> List[Dict]:
        """Merge ML analysis results with processed data"""
        try:
            # Ensure processed_data is not None and filter None entries
            if processed_data is None:
                processed_data = []
            processed_data = [record for record in processed_data if record is not None]

            # Safely get ML results with proper type checking and handle all possible types
            anomaly_scores = ml_results.get('anomaly_scores', []) if ml_results else []
            risk_levels = ml_results.get('risk_levels', []) if ml_results else []
            clusters = ml_results.get('clusters', []) if ml_results else []
            attachment_classifications = ml_results.get('attachment_classifications', []) if ml_results else []

            # Debug logging for attachment data
            self.logger.info(f"Debug: Processing {len(processed_data)} records for ML merge")
            for i, record in enumerate(processed_data[:3] if len(processed_data) >= 3 else processed_data):  # Show first 3 records safely
                if record is not None:
                    attachments = record.get('attachments', '')self.logger.info(f"Debug: Record {i} - attachments field: '{attachments}' (type: {type(attachments)})")

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

                # Add detailed attachment risk scoring
                if 'attachment_risk_scores' in ml_results and i < len(ml_results['attachment_risk_scores']):
                    risk_data = ml_results['attachment_risk_scores'][i]
                    record['attachment_risk_score'] = risk_data.get('risk_score', 0.0)
                    record['attachment_risk_level'] = risk_data.get('risk_level', 'Unknown')
                    record['attachment_risk_factors'] = risk_data.get('risk_factors', [])
                    record['attachment_malicious_indicators'] = risk_data.get('malicious_indicators', [])
                    record['attachment_exfiltration_risk'] = risk_data.get('exfiltration_risk', 'None')
                    record['attachment_count'] = risk_data.get('attachment_count', 0)

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
            chunk_size = 2500  # Increased chunk size for better performance with large files
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

                # Convert chunk data to lowercase
                chunk_df = self._convert_dataframe_to_lowercase(chunk_df)

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

            # Save processed data to session with error handling
            self.logger.info(f"Saving {len(all_processed_data)} processed records to session {session_id}")
            save_result = self.session_manager.update_session_data(session_id, all_processed_data, processing_stats)
            if not save_result.get('success'):
                self.logger.error(f"Failed to save large CSV processed data: {save_result.get('error')}")
                return {'success': False, 'error': f'Failed to save processed data: {save_result.get("error", "Unknown error")}'}

            # Extract sample data for preview (first 10 records)
            sample_data = []
            if all_processed_data:
                sample_data = all_processed_data[:10]
                # Clean sample data for JSON serialization
                for record in sample_data:
                    for key, value in record.items():
                        if pd.isna(value):
                            record[key] = None

            return {
                'success': True,
                'total_records': total_records,
                'processed_records': len(all_processed_data),
                'session_id': session_id,
                'processing_stats': processing_stats,
                'sample_data': sample_data
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
```

Applying changes to fix array/DataFrame boolean context issues and DataFrame empty checks.