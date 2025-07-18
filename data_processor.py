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
    
    def process_csv(self, file_path: str, session_id: str, filename: str) -> Dict:
        """Process uploaded CSV file"""
        try:
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
            
            # Apply whitelist filtering
            df_filtered = self._apply_whitelist_filtering(df)
            
            # Process each email record
            processed_data = []
            for index, row in df_filtered.iterrows():
                processed_record = self._process_email_record(row.to_dict(), index)
                processed_data.append(processed_record)
            
            # Apply ML analysis only if we have data
            if len(df_filtered) > 0:
                ml_results = self.ml_engine.analyze_emails(df_filtered)
            else:
                ml_results = {
                    'anomaly_scores': [],
                    'risk_levels': [],
                    'interesting_patterns': [],
                    'clusters': [],
                    'insights': {}
                }
            
            # Merge ML results with processed data
            processed_data = self._merge_ml_results(processed_data, ml_results)
            
            # Save processed data to session
            self.logger.info(f"Saving {len(processed_data)} processed records to session {session_id}")
            if processed_data:
                self.logger.info(f"Sample processed record keys: {list(processed_data[0].keys())}")
            
            save_result = self.session_manager.update_session_data(session_id, processed_data)
            if not save_result['success']:
                self.logger.error(f"Failed to save processed data: {save_result.get('error')}")
            
            return {
                'success': True,
                'total_records': len(df),
                'filtered_records': len(df_filtered),
                'processed_records': len(processed_data),
                'session_id': session_id
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
    
    def _apply_whitelist_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply whitelist domain filtering"""
        try:
            whitelists = self.session_manager.get_whitelists()
            whitelist_domains = whitelists.get('domains', [])
            
            if not whitelist_domains:
                return df
            
            # Filter out whitelisted domains
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
            
            self.logger.info(f"Filtered out {len(df) - len(filtered_df)} whitelisted records")
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"Error applying whitelist filtering: {str(e)}")
            return df
    
    def _process_email_record(self, record: Dict, record_index: int) -> Dict:
        """Process a single email record"""
        try:
            # Start with original record
            processed_record = record.copy()
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
            return record
    
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
            # Safely get ML results with proper type checking
            anomaly_scores = ml_results.get('anomaly_scores', [])
            risk_levels = ml_results.get('risk_levels', [])
            clusters = ml_results.get('clusters', [])
            
            # Ensure we have lists, not other types
            if not isinstance(anomaly_scores, list):
                anomaly_scores = []
            if not isinstance(risk_levels, list):
                risk_levels = []
            if not isinstance(clusters, list):
                clusters = []
            
            for i, record in enumerate(processed_data):
                if i < len(anomaly_scores) and isinstance(anomaly_scores[i], (int, float)):
                    record['ml_anomaly_score'] = float(anomaly_scores[i])
                else:
                    record['ml_anomaly_score'] = 0.0
                
                if i < len(risk_levels) and isinstance(risk_levels[i], str):
                    record['ml_risk_level'] = risk_levels[i]
                else:
                    record['ml_risk_level'] = 'Low'
                
                if i < len(clusters) and isinstance(clusters[i], (int, float)):
                    record['ml_cluster'] = int(clusters[i])
                else:
                    record['ml_cluster'] = -1
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error merging ML results: {str(e)}")
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
