import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import json
import logging
import re
from typing import Dict, List

class MLEngine:
    """Advanced ML engine for email anomaly detection and analysis"""

    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.label_encoder = LabelEncoder()
        self.logger = logging.getLogger(__name__)

        # Attachment classification keywords
        self.business_keywords = [
            'contract', 'agreement', 'proposal', 'invoice', 'report', 'presentation', 
            'budget', 'financial', 'quarterly', 'annual', 'meeting', 'agenda', 
            'minutes', 'policy', 'procedure', 'manual', 'guidelines', 'specification',
            'requirements', 'project', 'timeline', 'roadmap', 'strategy', 'analysis',
            'forecast', 'revenue', 'expense', 'tax', 'audit', 'compliance',
            'memo', 'briefing', 'summary', 'overview', 'review', 'assessment',
            'evaluation', 'performance', 'metrics', 'kpi', 'dashboard', 'chart'
        ]

        self.personal_keywords = [
            'personal', 'private', 'family', 'photo', 'picture', 'vacation', 'holiday',
            'birthday', 'anniversary', 'wedding', 'graduation', 'resume', 'cv',
            'medical', 'health', 'insurance', 'benefits', 'payroll', 'salary',
            'leave', 'time-off', 'sick', 'doctor', 'appointment', 'receipt',
            'bank', 'statement', 'loan', 'mortgage', 'credit', 'tax-return',
            'social', 'security', 'passport', 'driver', 'license', 'birth',
            'certificate', 'marriage', 'divorce', 'will', 'testament'
        ]

        self.suspicious_keywords = [
            'urgent', 'password', 'confidential', 'login', 'credentials', 'verify',
            'security alert', 'important notice', 'account update', 'restricted',
            'malware', 'virus', 'ransomware', 'trojan', 'spyware', 'phishing',
            'click here', 'immediate action', 'limited time', 'free offer',
            'prize', 'winner', 'lottery', 'inheritance', 'secret', 'anonymous',
            'unusual activity', 'compromised', 'breach', 'unauthorized', 'fraudulent'
        ]

    def analyze_emails(self, df):
        """Perform comprehensive ML analysis on email data"""
        try:
            results = {
                'anomaly_scores': [],
                'risk_levels': [],
                'interesting_patterns': [],
                'clusters': [],
                'insights': {}
            }

            if df.empty or len(df) == 0:
                self.logger.info("No data to analyze - returning empty results")
                return results

            # Prepare features for ML analysis
            features = self._prepare_features(df)

            # Anomaly detection
            anomaly_scores = self._detect_anomalies(features)
            results['anomaly_scores'] = anomaly_scores.tolist()

            # Risk level classification
            risk_levels = self._classify_risk_levels(df, anomaly_scores)
            results['risk_levels'] = risk_levels

            # Pattern discovery
            patterns = self._discover_patterns(df)
            results['interesting_patterns'] = patterns

            # Clustering analysis
            clusters = self._perform_clustering(features)
            results['clusters'] = clusters.tolist()

            # Classify attachments
            attachment_classifications = self._classify_attachments(df)
            results['attachment_classifications'] = attachment_classifications

            # Generate insights
            insights = self._generate_insights(df, anomaly_scores, risk_levels, patterns)
            results['insights'] = insights

            return results

        except Exception as e:
            self.logger.error(f"Error in ML analysis: {str(e)}")
            return {
                'anomaly_scores': [],
                'risk_levels': [],
                'interesting_patterns': [],
                'clusters': [],
                'insights': {}
            }

    def _prepare_features(self, df):
        """Prepare numerical features for ML algorithms"""
        features = []

        # Time-based features
        if '_time' in df.columns:
            df['_time'] = pd.to_datetime(df['_time'], errors='coerce')
            features.append(df['_time'].dt.hour.fillna(12))  # Hour of day
            features.append(df['_time'].dt.weekday.fillna(2))  # Day of week

        # Domain-based features
        if 'recipients_email_domain' in df.columns:
            domain_counts = df['recipients_email_domain'].value_counts()
            features.append(df['recipients_email_domain'].map(domain_counts).fillna(1))

        # Attachment features
        if 'attachments' in df.columns:
            attachment_counts = df['attachments'].str.count(',').fillna(0) + 1
            features.append(attachment_counts)

        # Subject length
        if 'subject' in df.columns:
            subject_lengths = df['subject'].str.len().fillna(0)
            features.append(subject_lengths)

        # Wordlist matches
        if 'wordlist_attachment' in df.columns:
            features.append(df['wordlist_attachment'].map({'Yes': 1, 'No': 0}).fillna(0))

        if 'wordlist_subject' in df.columns:
            features.append(df['wordlist_subject'].map({'Yes': 1, 'No': 0}).fillna(0))

        # Convert to numpy array
        if features:
            return np.column_stack(features)
        else:
            return np.array([]).reshape(len(df), 0)

    def _detect_anomalies(self, features):
        """Detect anomalies using Isolation Forest"""
        if features.shape[1] == 0:
            return np.zeros(features.shape[0])

        try:
            # Fit and predict
            anomaly_scores = self.isolation_forest.fit_predict(features)
            # Convert to probability-like scores (0-1)
            decision_scores = self.isolation_forest.decision_function(features)
            normalized_scores = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min())

            return normalized_scores
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {str(e)}")
            return np.zeros(features.shape[0])

    def analyze_anomaly_details(self, email_data):
        """Analyze specific anomaly details for a single email"""
        anomaly_details = []

        try:
            # Check for unusual time patterns
            if '_time' in email_data:
                email_time = pd.to_datetime(email_data['_time'], errors='coerce')
                if not pd.isna(email_time):
                    hour = email_time.hour
                    if hour < 6 or hour > 22:  # Outside business hours
                        anomaly_details.append({
                            'type': 'Temporal Anomaly',
                            'description': f'Email sent at unusual hour: {hour}:00',
                            'severity': 'Medium'
                        })

                    weekday = email_time.weekday()
                    if weekday >= 5:  # Weekend
                        anomaly_details.append({
                            'type': 'Temporal Anomaly',
                            'description': 'Email sent during weekend',
                            'severity': 'Low'
                        })

            # Check for content-based anomalies
            if email_data.get('wordlist_attachment') == 'Yes':
                anomaly_details.append({
                    'type': 'Content Anomaly',
                    'description': 'Attachment contains suspicious keywords',
                    'severity': 'High'
                })

            if email_data.get('wordlist_subject') == 'Yes':
                anomaly_details.append({
                    'type': 'Content Anomaly',
                    'description': 'Subject line contains suspicious keywords',
                    'severity': 'High'
                })

            # Check for behavioral anomalies
            if email_data.get('leaver') == 'Yes':
                anomaly_details.append({
                    'type': 'Behavioral Anomaly',
                    'description': 'Email from user marked as leaver',
                    'severity': 'Critical'
                })

            # Check for domain-based anomalies
            domain = email_data.get('recipients_email_domain', '')
            if domain and not any(trusted in domain.lower() for trusted in ['gmail.com', 'outlook.com', 'company.com']):
                anomaly_details.append({
                    'type': 'Domain Anomaly',
                    'description': f'Email sent to potentially suspicious domain: {domain}',
                    'severity': 'Medium'
                })

            # Check for attachment anomalies
            if email_data.get('has_attachments') and email_data.get('attachments'):
                attachments = email_data.get('attachments', '')
                if len(attachments.split(',')) > 3:
                    anomaly_details.append({
                        'type': 'Attachment Anomaly',
                        'description': f'Unusually high number of attachments: {len(attachments.split(","))}',
                        'severity': 'Medium'
                    })

            # Check for subject length anomalies
            subject = email_data.get('subject', '')
            if len(subject) > 100:
                anomaly_details.append({
                    'type': 'Content Anomaly',
                    'description': f'Unusually long subject line: {len(subject)} characters',
                    'severity': 'Low'
                })
            elif len(subject) < 5:
                anomaly_details.append({
                    'type': 'Content Anomaly',
                    'description': 'Unusually short subject line',
                    'severity': 'Low'
                })

        except Exception as e:
            self.logger.error(f"Error analyzing anomaly details: {str(e)}")

        return anomaly_details

    def _classify_risk_levels(self, df, anomaly_scores):
        """Classify emails into risk levels based on multiple factors"""
        risk_levels = []

        for i, row in df.iterrows():
            score = anomaly_scores[i] if i < len(anomaly_scores) else 0
            risk_factors = 0

            # Check various risk factors
            if row.get('wordlist_attachment') == 'Yes':
                risk_factors += 2
            if row.get('wordlist_subject') == 'Yes':
                risk_factors += 2
            if row.get('leaver') == 'Yes':
                risk_factors += 3
            if score > 0.8:
                risk_factors += 2
            if row.get('status') == 'Blocked':
                risk_factors += 1

            # Classify based on total risk factors
            if risk_factors >= 5:
                risk_levels.append('Critical')
            elif risk_factors >= 3:
                risk_levels.append('High')
            elif risk_factors >= 1:
                risk_levels.append('Medium')
            else:
                risk_levels.append('Low')

        return risk_levels

    def _discover_patterns(self, df):
        """Discover interesting patterns in the data"""
        patterns = []

        try:
            # Pattern 1: Unusual sending times
            if '_time' in df.columns:
                df['_time'] = pd.to_datetime(df['_time'], errors='coerce')
                hour_counts = df['_time'].dt.hour.value_counts()
                unusual_hours = hour_counts[hour_counts < hour_counts.mean() * 0.3].index
                if len(unusual_hours) > 0:
                    patterns.append({
                        'type': 'temporal',
                        'description': f'Emails sent during unusual hours: {list(unusual_hours)}',
                        'severity': 'Medium'
                    })

            # Pattern 2: High-risk domain patterns
            if 'recipients_email_domain' in df.columns:
                domain_counts = df['recipients_email_domain'].value_counts()
                suspicious_domains = domain_counts[domain_counts == 1].index  # Single-use domains
                if len(suspicious_domains) > 5:
                    patterns.append({
                        'type': 'domain',
                        'description': f'Multiple single-use domains detected: {len(suspicious_domains)} domains',
                        'severity': 'High'
                    })

            # Pattern 3: Wordlist pattern analysis
            if 'wordlist_attachment' in df.columns and 'wordlist_subject' in df.columns:
                both_matches = df[(df['wordlist_attachment'] == 'Yes') & (df['wordlist_subject'] == 'Yes')]
                if len(both_matches) > 0:
                    patterns.append({
                        'type': 'content',
                        'description': f'{len(both_matches)} emails with both subject and attachment wordlist matches',
                        'severity': 'Critical'
                    })

            # Pattern 4: Leaver activity patterns
            if 'leaver' in df.columns:
                leaver_emails = df[df['leaver'] == 'Yes']
                if len(leaver_emails) > 0:
                    patterns.append({
                        'type': 'behavioral',
                        'description': f'{len(leaver_emails)} emails from users marked as leavers',
                        'severity': 'High'
                    })

        except Exception as e:
            self.logger.error(f"Error discovering patterns: {str(e)}")

        return patterns

    def analyze_anomaly_details(self, record: Dict) -> List[Dict]:
        """Generate detailed anomaly analysis for high-risk emails"""
        try:
            anomaly_details = []

            # Check for unusual attachment patterns
            if record.get('has_attachments'):
                attachments = record.get('attachments', '')
                if any(ext in attachments.lower() for ext in ['.exe', '.bat', '.scr', '.zip']):
                    anomaly_details.append({
                        'type': 'Suspicious Attachment Type',
                        'description': 'Email contains potentially dangerous file types',
                        'field': 'attachments',
                        'severity': 'High'
                    })

                # Check for multiple attachments
                if len(attachments.split(',')) > 3:
                    anomaly_details.append({
                        'type': 'Multiple Attachments',
                        'description': 'Email contains an unusually high number of attachments',
                        'field': 'attachments',
                        'severity': 'Medium'
                    })

            # Check for unusual recipient patterns
            recipients = record.get('recipients', '')
            if recipients and len(recipients.split(',')) > 5:
                anomaly_details.append({
                    'type': 'Mass Distribution',
                    'description': 'Email sent to an unusually large number of recipients',
                    'field': 'recipients',
                    'severity': 'High'
                })

            # Check for external domain communication
            sender_domain = record.get('sender', '').split('@')[-1] if '@' in record.get('sender', '') else ''
            recipient_domain = record.get('recipients_email_domain', '')

            if sender_domain and recipient_domain and sender_domain != recipient_domain:
                public_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
                if recipient_domain.lower() in public_domains:
                    anomaly_details.append({
                        'type': 'External Personal Email',
                        'description': 'Communication with external personal email services',
                        'field': 'recipients_email_domain',
                        'severity': 'Medium'
                    })

            # Check for timing anomalies (if timestamp available)
            timestamp = record.get('_time', '')
            if timestamp:
                try:
                    from datetime import datetime
                    email_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    hour = email_time.hour

                    # Check for off-hours activity (before 7 AM or after 7 PM)
                    if hour < 7 or hour > 19:
                        anomaly_details.append({
                            'type': 'Off-Hours Activity',
                            'description': 'Email sent during unusual business hours',
                            'field': '_time',
                            'severity': 'Medium'
                        })
                except:
                    pass

            # If no specific anomalies found but score is high, add generic anomaly
            if not anomaly_details:
                anomaly_details.append({
                    'type': 'Statistical Anomaly',
                    'description': 'Email exhibits patterns that deviate significantly from normal behavior',
                    'field': 'ml_analysis',
                    'severity': 'High'
                })

            return anomaly_details

        except Exception as e:
            self.logger.error(f"Error analyzing anomaly details: {str(e)}")
            return [{
                'type': 'Analysis Error',
                'description': f'Unable to analyze anomaly details: {str(e)}',
                'field': 'system',
                'severity': 'Low'
            }]

    def _perform_clustering(self, features):
        """Perform clustering analysis to identify groups of similar emails"""
        if features.shape[1] == 0:
            return np.zeros(features.shape[0])

        try:
            clusters = self.dbscan.fit_predict(features)
            return clusters
        except Exception as e:
            self.logger.error(f"Error in clustering: {str(e)}")
            return np.zeros(features.shape[0])

    def _generate_insights(self, df, anomaly_scores, risk_levels, patterns):
        """Generate comprehensive insights from the analysis"""
        insights = {
            'total_emails': len(df),
            'risk_distribution': {},
            'anomaly_summary': {},
            'pattern_summary': {},
            'recommendations': []
        }

        # Risk distribution
        risk_counts = pd.Series(risk_levels).value_counts()
        insights['risk_distribution'] = risk_counts.to_dict()

        # Anomaly summary
        high_anomaly_count = sum(1 for score in anomaly_scores if score > 0.7)
        insights['anomaly_summary'] = {
            'high_anomaly_count': high_anomaly_count,
            'anomaly_percentage': (high_anomaly_count / len(df)) * 100 if len(df) > 0 else 0
        }

        # Pattern summary
        insights['pattern_summary'] = {
            'total_patterns': len(patterns),
            'critical_patterns': len([p for p in patterns if p['severity'] == 'Critical']),
            'high_patterns': len([p for p in patterns if p['severity'] == 'High'])
        }

        # Generate recommendations
        recommendations = []

        if high_anomaly_count > len(df) * 0.1:
            recommendations.append("High number of anomalous emails detected. Consider reviewing filtering rules.")

        if risk_counts.get('Critical', 0) > 0:
            recommendations.append("Critical risk emails identified. Immediate review recommended.")

        if len([p for p in patterns if p['severity'] == 'Critical']) > 0:
            recommendations.append("Critical patterns detected. Consider implementing additional security measures.")

        insights['recommendations'] = recommendations

        return insights

    def detect_bau_emails(self, df):
        """Detect BAU (Business As Usual) emails - high volume business communications"""
        try:
            bau_analysis = {
                'sender_domain_patterns': {},
                'recipient_domain_patterns': {},
                'high_volume_pairs': [],
                'business_communication_clusters': [],
                'bau_candidates': [],
                'bau_stats': {}
            }

            # Analyze sender to recipient domain patterns
            domain_pairs = {}
            sender_volumes = {}
            recipient_volumes = {}

            for index, row in df.iterrows():
                sender = row.get('sender', '')
                recipient_domain = row.get('recipients_email_domain', '')
                
                # Extract sender domain
                sender_domain = sender.split('@')[-1] if '@' in sender else sender
                
                # Count volumes
                sender_volumes[sender] = sender_volumes.get(sender, 0) + 1
                recipient_volumes[recipient_domain] = recipient_volumes.get(recipient_domain, 0) + 1
                
                # Track domain pairs
                pair_key = f"{sender_domain} -> {recipient_domain}"
                if pair_key not in domain_pairs:
                    domain_pairs[pair_key] = {
                        'count': 0,
                        'senders': set(),
                        'subjects': set(),
                        'has_attachments': 0,
                        'business_keywords': 0
                    }
                
                domain_pairs[pair_key]['count'] += 1
                domain_pairs[pair_key]['senders'].add(sender)
                domain_pairs[pair_key]['subjects'].add(row.get('subject', ''))
                
                if row.get('has_attachments'):
                    domain_pairs[pair_key]['has_attachments'] += 1
                
                # Check for business keywords in subject
                subject = str(row.get('subject', '')).lower()
                if any(keyword in subject for keyword in ['report', 'invoice', 'contract', 'meeting', 'proposal']):
                    domain_pairs[pair_key]['business_keywords'] += 1

            # Identify high-volume BAU patterns (threshold: >5 emails)
            bau_threshold = 5
            for pair_key, data in domain_pairs.items():
                if data['count'] >= bau_threshold:
                    bau_score = self._calculate_bau_score(data)
                    
                    bau_analysis['high_volume_pairs'].append({
                        'pair': pair_key,
                        'volume': data['count'],
                        'unique_senders': len(data['senders']),
                        'unique_subjects': len(data['subjects']),
                        'attachment_ratio': data['has_attachments'] / data['count'],
                        'business_keyword_ratio': data['business_keywords'] / data['count'],
                        'bau_score': bau_score,
                        'bau_likelihood': self._get_bau_likelihood(bau_score)
                    })

            # Sort by BAU score (highest first)
            bau_analysis['high_volume_pairs'].sort(key=lambda x: x['bau_score'], reverse=True)

            # Generate BAU candidates for whitelisting
            for pair_data in bau_analysis['high_volume_pairs']:
                if pair_data['bau_likelihood'] in ['High', 'Very High']:
                    bau_analysis['bau_candidates'].append({
                        'domain_pair': pair_data['pair'],
                        'volume': pair_data['volume'],
                        'confidence': pair_data['bau_likelihood'],
                        'recommendation': self._get_bau_recommendation(pair_data)
                    })

            # Calculate BAU statistics
            total_emails = len(df)
            high_volume_emails = sum(pair['volume'] for pair in bau_analysis['high_volume_pairs'])
            
            bau_analysis['bau_stats'] = {
                'total_emails': total_emails,
                'high_volume_emails': high_volume_emails,
                'bau_percentage': (high_volume_emails / total_emails * 100) if total_emails > 0 else 0,
                'unique_domain_pairs': len(domain_pairs),
                'bau_candidates_count': len(bau_analysis['bau_candidates']),
                'top_sender_domains': dict(sorted(
                    {sender.split('@')[-1]: count for sender, count in sender_volumes.items()}.items(),
                    key=lambda x: x[1], reverse=True
                )[:10]),
                'top_recipient_domains': dict(sorted(recipient_volumes.items(), key=lambda x: x[1], reverse=True)[:10])
            }

            return bau_analysis

        except Exception as e:
            self.logger.error(f"Error in BAU detection: {str(e)}")
            return {
                'sender_domain_patterns': {},
                'recipient_domain_patterns': {},
                'high_volume_pairs': [],
                'business_communication_clusters': [],
                'bau_candidates': [],
                'bau_stats': {}
            }

    def _calculate_bau_score(self, data):
        """Calculate BAU likelihood score (0-100)"""
        score = 0
        
        # Volume factor (0-30 points)
        volume_score = min(30, data['count'] * 2)
        score += volume_score
        
        # Sender diversity (0-20 points) - multiple senders indicate business process
        sender_diversity = min(20, len(data['senders']) * 4)
        score += sender_diversity
        
        # Subject diversity (0-20 points) - varied subjects indicate legitimate business
        subject_diversity = min(20, len(data['subjects']) * 2)
        score += subject_diversity
        
        # Attachment ratio (0-15 points) - business emails often have attachments
        attachment_score = data['has_attachments'] / data['count'] * 15
        score += attachment_score
        
        # Business keyword ratio (0-15 points)
        business_score = data['business_keywords'] / data['count'] * 15
        score += business_score
        
        return min(100, score)

    def _get_bau_likelihood(self, score):
        """Convert BAU score to likelihood category"""
        if score >= 80:
            return 'Very High'
        elif score >= 60:
            return 'High'
        elif score >= 40:
            return 'Medium'
        elif score >= 20:
            return 'Low'
        else:
            return 'Very Low'

    def _get_bau_recommendation(self, pair_data):
        """Generate recommendation for BAU pair"""
        if pair_data['bau_likelihood'] == 'Very High':
            return 'Strongly recommend whitelisting - Clear business communication pattern'
        elif pair_data['bau_likelihood'] == 'High':
            return 'Recommend whitelisting - Likely legitimate business emails'
        elif pair_data['bau_likelihood'] == 'Medium':
            return 'Review manually - Mixed signals in communication pattern'
        else:
            return 'Monitor - Insufficient data for confident classification'

    def get_insights(self, processed_data):
        """Get ML insights for dashboard display"""
        if not processed_data:
            return {
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
                'bau_analysis': {
                    'bau_candidates_count': 0,
                    'bau_percentage': 0,
                    'high_volume_pairs': []
                },
                'recommendations': []
            }

        # Convert to DataFrame for analysis
        df = pd.DataFrame(processed_data)

        # Perform ML analysis
        ml_results = self.analyze_emails(df)

        # Perform BAU analysis
        bau_results = self.detect_bau_emails(df)

        # Return the insights with proper structure
        insights = ml_results['insights']

        # Ensure anomaly_summary has the required fields
        if 'anomaly_summary' not in insights:
            insights['anomaly_summary'] = {}

        if 'anomaly_percentage' not in insights['anomaly_summary']:
            insights['anomaly_summary']['anomaly_percentage'] = 0

        if 'high_anomaly_count' not in insights['anomaly_summary']:
            insights['anomaly_summary']['high_anomaly_count'] = 0

        # Ensure pattern_summary has the required fields
        if 'pattern_summary' not in insights:
            insights['pattern_summary'] = {}

        for field in ['total_patterns', 'critical_patterns', 'high_patterns']:
            if field not in insights['pattern_summary']:
                insights['pattern_summary'][field] = 0

        # Add BAU analysis
        insights['bau_analysis'] = bau_results['bau_stats']
        insights['bau_analysis']['high_volume_pairs'] = bau_results['high_volume_pairs'][:10]  # Top 10
        insights['bau_analysis']['bau_candidates'] = bau_results['bau_candidates']

        return insights

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract relevant features from the email data"""
        try:
            # Initialize an empty list to hold feature columns
            features = []

            # Numerical Features
            if 'size' in df.columns:
                features.append(df['size'].fillna(0))  # Fill missing values with 0

            # Categorical Features (one-hot encode)
            if 'leaver' in df.columns:
                features.append(df['leaver'].map({'Yes': 1, 'No': 0}).fillna(0))

            if 'has_attachments' in df.columns:
                features.append(df['has_attachments'].map({True: 1, False: 0}).fillna(0))

            # Datetime Features
            if '_time' in df.columns:
                df['_time'] = pd.to_datetime(df['_time'], errors='coerce')
                features.append(df['_time'].dt.hour.fillna(-1))  # Hour of day
                features.append(df['_time'].dt.dayofweek.fillna(-1))  # Day of week

            # Text Features (TF-IDF)
            if 'subject' in df.columns:
                tfidf = self.tfidf_vectorizer.fit_transform(df['subject'].fillna(''))
                tfidf_df = pd.DataFrame(tfidf.toarray(), columns=self.tfidf_vectorizer.get_feature_names_out())
                features.extend([tfidf_df[col] for col in tfidf_df.columns])

            # Combine features into a DataFrame
            if features:
                feature_df = pd.DataFrame(features).T  # Transpose to align columns properly
                return feature_df.fillna(0)  # Fill any remaining missing values
            else:
                return pd.DataFrame()  # Return an empty DataFrame if no features extracted

        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return pd.DataFrame()  # Return an empty DataFrame in case of an error

    def _detect_patterns(self, df: pd.DataFrame, features: pd.DataFrame) -> List:
        """Detect interesting patterns in the data"""
        try:
            patterns = []

            # Pattern 1: High anomaly scores in specific time frames
            if '_time' in df.columns:
                df['_time'] = pd.to_datetime(df['_time'], errors='coerce')
                hour_counts = df['_time'].dt.hour.value_counts()
                unusual_hours = hour_counts[hour_counts < hour_counts.mean() * 0.5].index
                if len(unusual_hours) > 0:
                    patterns.append(
                        {'type': 'temporal', 'description': f'Emails sent during unusual hours: {list(unusual_hours)}',
                         'severity': 'Medium'})

            # Pattern 2: Suspicious Keywords in Subject
            if 'subject' in df.columns:
                suspicious_keywords = ['urgent', 'password', 'confidential']
                keyword_matches = df['subject'].str.contains('|'.join(suspicious_keywords), case=False).sum()
                if keyword_matches > 5:
                    patterns.append({
                        'type': 'content',
                        'description': f'Multiple emails with suspicious keywords in subject: {keyword_matches} matches',
                        'severity': 'High'
                    })

            # Pattern 3: High volume from Leavers
            if 'leaver' in df.columns:
                leaver_emails = df[df['leaver'] == 'Yes']
                if len(leaver_emails) > 0:
                    patterns.append(
                        {'type': 'behavioral', 'description': f'{len(leaver_emails)} emails from users marked as leavers',
                         'severity': 'High'})

            return patterns

        except Exception as e:
            self.logger.error(f"Error detecting patterns: {str(e)}")
            return []

    def analyze_emails(self, df: pd.DataFrame) -> Dict:
        """Analyze emails and return ML insights"""
        try:
            if df.empty:
                return {
                    'anomaly_scores': [],
                    'risk_levels': [],
                    'interesting_patterns': [],
                    'clusters': [],
                    'attachment_classifications': [],
                    'insights': {}
                }

            # Convert to numeric features for ML analysis
            features = self._extract_features(df)

            if features.empty:
                return {
                    'anomaly_scores': [0.0] * len(df),
                    'risk_levels': ['Low'] * len(df),
                    'interesting_patterns': [],
                    'clusters': [-1] * len(df),
                    'attachment_classifications': ['Unknown'] * len(df),
                    'insights': {}
                }

            # Anomaly detection
            anomaly_scores = self._detect_anomalies(features)

            # Risk level classification
            risk_levels = self._classify_risk_levels(features, anomaly_scores)

            # Clustering
            clusters = self._perform_clustering(features)

            # Pattern detection
            patterns = self._detect_patterns(df, features)

            # Attachment classification
            attachment_classifications = self._classify_attachments(df)

            # Ensure all results are proper lists
            anomaly_scores_list = []
            if hasattr(anomaly_scores, 'tolist'):
                anomaly_scores_list = anomaly_scores.tolist()
            elif isinstance(anomaly_scores, (list, tuple)):
                anomaly_scores_list = list(anomaly_scores)
            elif isinstance(anomaly_scores, (int, float)):
                anomaly_scores_list = [float(anomaly_scores)] * len(df)
            else:
                anomaly_scores_list = [0.0] * len(df)

            risk_levels_list = []
            if isinstance(risk_levels, (list, tuple)):
                risk_levels_list = list(risk_levels)
            elif isinstance(risk_levels, str):
                risk_levels_list = [risk_levels] * len(df)
            else:
                risk_levels_list = ['Low'] * len(df)

            clusters_list = []
            if hasattr(clusters, 'tolist'):
                clusters_list = clusters.tolist()
            elif isinstance(clusters, (list, tuple)):
                clusters_list = list(clusters)
            elif isinstance(clusters, (int, float)):
                clusters_list = [int(clusters)] * len(df)
            else:
                clusters_list = [-1] * len(df)

            # Ensure all lists have the same length as the DataFrame
            target_length = len(df)
            if len(anomaly_scores_list) != target_length:
                anomaly_scores_list = (anomaly_scores_list * target_length)[:target_length] if anomaly_scores_list else [0.0] * target_length
            if len(risk_levels_list) != target_length:
                risk_levels_list = (risk_levels_list * target_length)[:target_length] if risk_levels_list else ['Low'] * target_length
            if len(clusters_list) != target_length:
                clusters_list = (clusters_list * target_length)[:target_length] if clusters_list else [-1] * target_length

            return {
                'anomaly_scores': anomaly_scores_list,
                'risk_levels': risk_levels_list,
                'interesting_patterns': patterns if isinstance(patterns, list) else [],
                'clusters': clusters_list,
                'attachment_classifications': attachment_classifications,
                'insights': self._generate_insights(df, anomaly_scores_list, risk_levels_list, patterns)
            }

        except Exception as e:
            self.logger.error(f"Error in ML analysis: {str(e)}")
            # Return safe defaults
            target_length = len(df) if not df.empty else 0
            return {
                'anomaly_scores': [0.0] * target_length,
                'risk_levels': ['Low'] * target_length,
                'interesting_patterns': [],
                'clusters': [-1] * target_length,
                'attachment_classifications': ['Unknown'] * target_length,
                'insights': {}
            }

    def _classify_attachments(self, df):
        """Classify attachments as business, personal, or unknown based on filename analysis"""
        try:
            classifications = []

            if 'attachments' not in df.columns:
                self.logger.info("No 'attachments' column found in DataFrame")
                return ['No Attachments'] * len(df)

            self.logger.info(f"Processing {len(df)} records for attachment classification")

            for idx, row in df.iterrows():
                attachments = row.get('attachments', '')
                self.logger.info(f"Row {idx}: attachments = '{attachments}' (type: {type(attachments)})")

                if pd.isna(attachments) or str(attachments).strip() == '' or str(attachments).lower() == 'nan' or str(attachments).strip() == '-':
                    classifications.append('No Attachments')
                    continue

                # Parse multiple attachments (comma-separated)
                attachment_str = str(attachments).strip()
                attachment_list = [att.strip() for att in attachment_str.split(',') if att.strip()]
                
                self.logger.info(f"Processing attachment list: {attachment_list}")

                business_score = 0
                personal_score = 0
                suspicious_score = 0

                for attachment in attachment_list:
                    attachment_lower = attachment.lower()
                    self.logger.info(f"Analyzing attachment: '{attachment}' -> '{attachment_lower}'")

                    # Check for business keywords
                    for keyword in self.business_keywords:
                        if keyword in attachment_lower:
                            business_score += 1
                            self.logger.info(f"Business keyword '{keyword}' found in '{attachment}'")
                            break

                    # Check for personal keywords
                    for keyword in self.personal_keywords:
                        if keyword in attachment_lower:
                            personal_score += 1
                            self.logger.info(f"Personal keyword '{keyword}' found in '{attachment}'")
                            break

                    # Check for suspicious keywords
                    for keyword in self.suspicious_keywords:
                        if keyword in attachment_lower:
                            suspicious_score += 1
                            self.logger.info(f"Suspicious keyword '{keyword}' found in '{attachment}'")
                            break

                    # Check file extensions for business patterns
                    if any(ext in attachment_lower for ext in ['.xlsx', '.pptx', '.docx', '.pdf']):
                        if any(word in attachment_lower for word in ['report', 'proposal', 'contract', 'invoice']):
                            business_score += 0.5
                            self.logger.info(f"Business file pattern detected in '{attachment}'")

                    # Check file extensions for personal patterns
                    if any(ext in attachment_lower for ext in ['.jpg', '.jpeg', '.png', '.mp4', '.mp3']):
                        if any(word in attachment_lower for word in ['photo', 'pic', 'image', 'video']):
                            personal_score += 0.5
                            self.logger.info(f"Personal file pattern detected in '{attachment}'")

                # Determine classification
                classification_result = 'Unknown'
                if suspicious_score > 0:
                    classification_result = 'Suspicious'
                elif business_score > personal_score:
                    classification_result = 'Business'
                elif personal_score > business_score:
                    classification_result = 'Personal'
                elif business_score == personal_score and business_score > 0:
                    classification_result = 'Mixed'
                else:
                    # For report.pdf, let's check if it matches business patterns
                    if any(word in attachment_str.lower() for word in ['report', 'document', 'file']):
                        classification_result = 'Business'
                    else:
                        classification_result = 'Unknown'

                self.logger.info(f"Final classification for '{attachment_str}': {classification_result} (business: {business_score}, personal: {personal_score}, suspicious: {suspicious_score})")
                classifications.append(classification_result)

            self.logger.info(f"Classified {len(classifications)} attachment records: {classifications}")
            return classifications

        except Exception as e:
            self.logger.error(f"Error classifying attachments: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return ['Unknown'] * len(df)

    def _classify_single_attachment_list(self, attachment_list):
        """Classify attachments as Business, Personal, or Mixed"""
        try:
            if not attachment_list:
                return 'Unknown'

            # Debug logging
            self.logger.info(f"Debug: Processing attachments: {attachment_list}")
            self.logger.info(f"Debug: Business keywords: {self.business_keywords[:10]}")  # Show first 10
            self.logger.info(f"Debug: Personal keywords: {self.personal_keywords[:10]}")    # Show first 10

            business_score = 0
            personal_score = 0
            suspicious_score = 0

            for attachment in attachment_list:
                original_attachment = attachment
                attachment = attachment.strip().lower()
                self.logger.info(f"Debug: Processing attachment: '{original_attachment}' -> '{attachment}'")

                # Check for business keywords
                for keyword in self.business_keywords:
                    if keyword in attachment:
                        business_score += 1
                        self.logger.info(f"Debug: Business keyword match: '{keyword}' in '{attachment}'")
                        break

                # Check for personal keywords
                for keyword in self.personal_keywords:
                    if keyword in attachment:
                        personal_score += 1
                        self.logger.info(f"Debug: Personal keyword match: '{keyword}' in '{attachment}'")
                        break

                # Check for suspicious keywords
                for keyword in self.suspicious_keywords:
                    if keyword in attachment:
                        suspicious_score += 1
                        self.logger.info(f"Debug: Suspicious keyword match: '{keyword}' in '{attachment}'")
                        break

                # Check file extensions for business patterns
                if any(ext in attachment for ext in ['.xlsx', '.pptx', '.docx', '.pdf']):
                    if any(word in attachment for word in ['report', 'proposal', 'contract', 'invoice']):
                        business_score += 0.5
                        self.logger.info(f"Debug: File extension + business word match for '{attachment}'")

            self.logger.info(f"Debug: Final scores - Business: {business_score}, Personal: {personal_score}, Suspicious: {suspicious_score}")

            # Determine classification
            if suspicious_score > 0:
                classification = 'Suspicious'
            elif business_score > personal_score:
                classification = 'Business'
            elif personal_score > business_score:
                classification = 'Personal'
            elif business_score > 0 and personal_score > 0:
                classification = 'Mixed'
            else:
                classification = 'Unknown'

            self.logger.info(f"Debug: Final classification: {classification}")
            return classification

        except Exception as e:
            self.logger.error(f"Error classifying attachments: {str(e)}")
            return 'Unknown'