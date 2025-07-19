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

        # Attachment classification keywords - all in lowercase since data is lowercase
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
        """Perform comprehensive ML analysis on email data with batch processing"""
        try:
            results = {
                'anomaly_scores': [],
                'risk_levels': [],
                'interesting_patterns': [],
                'clusters': [],
                'insights': {}
            }

            if len(df) == 0:
                self.logger.info("No data to analyze - returning empty results")
                return results

            # For large datasets, process in batches
            batch_size = 1000
            total_rows = len(df)

            if total_rows > batch_size:
                self.logger.info(f"Processing large dataset ({total_rows} records) in batches of {batch_size}")

                all_anomaly_scores = []
                all_risk_levels = []
                all_clusters = []
                all_attachment_classifications = []

                for i in range(0, total_rows, batch_size):
                    batch_df = df.iloc[i:i+batch_size].copy()
                    self.logger.info(f"Processing batch {i//batch_size + 1}/{(total_rows + batch_size - 1)//batch_size}")

                    # Process batch
                    batch_results = self._process_batch(batch_df)

                    all_anomaly_scores.extend(batch_results['anomaly_scores'])
                    all_risk_levels.extend(batch_results['risk_levels'])
                    all_clusters.extend(batch_results['clusters'])
                    all_attachment_classifications.extend(batch_results['attachment_classifications'])

                results['anomaly_scores'] = all_anomaly_scores
                results['risk_levels'] = all_risk_levels
                results['clusters'] = all_clusters
                results['attachment_classifications'] = all_attachment_classifications

                # Process patterns and insights on full dataset but with sampling
                sampled_df = df.sample(n=min(2000, len(df)), random_state=42)
                results['interesting_patterns'] = self._discover_patterns(sampled_df)
                results['insights'] = self._generate_insights(sampled_df, all_anomaly_scores[:len(sampled_df)], all_risk_levels[:len(sampled_df)], results['interesting_patterns'])

                return results

            # Prepare features for ML analysis (small dataset)
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
        if len(features) > 0:
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
                single_use_mask = domain_counts == 1
                if single_use_mask.any():
                    suspicious_domains = domain_counts[single_use_mask].index
                    if len(suspicious_domains) > 5:
                        patterns.append({
                            'type': 'domain',
                            'description': f'Multiple single-use domains detected: {len(suspicious_domains)} domains',
                            'severity': 'High'
                        })

            # Pattern 3: Wordlist pattern analysis
            if 'wordlist_attachment' in df.columns and 'wordlist_subject' in df.columns:
                attachment_matches = df['wordlist_attachment'] == 'Yes'
                subject_matches = df['wordlist_subject'] == 'Yes'
                both_matches_mask = attachment_matches & subject_matches
                if both_matches_mask.any():
                    both_matches_count = both_matches_mask.sum()
                    patterns.append({
                        'type': 'content',
                        'description': f'{both_matches_count} emails with both subject and attachment wordlist matches',
                        'severity': 'Critical'
                    })

            # Pattern 4: Leaver activity patterns
            if 'leaver' in df.columns:
                leaver_mask = df['leaver'] == 'Yes'
                if leaver_mask.any():
                    leaver_count = leaver_mask.sum()
                    patterns.append({
                        'type': 'behavioral',
                        'description': f'{leaver_count} emails from users marked as leavers',
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
        """
        Detect Business As Usual (BAU) emails based on communication patterns
        Returns analysis of regular communication patterns that should be whitelisted
        """
        try:
            if len(df) == 0:
                return {
                    'bau_candidates': [],
                    'bau_percentage': 0,
                    'high_volume_pairs': [],
                    'unique_domains': 0,
                    'recommendations': ['No data available for BAU analysis']
                }

            # Extract sender and recipient domains
            def extract_domain(email):
                if pd.isna(email) or not isinstance(email, str):
                    return None
                email = str(email).strip()
                if '@' in email:
                    return email.split('@')[-1].lower().strip()
                return None

            # Create domain pairs for analysis
            domain_pairs = []
            sender_recipient_pairs = []

            for _, row in df.iterrows():
                sender = str(row.get('sender', '')).strip()
                sender_domain = extract_domain(sender)
                recipients = str(row.get('recipients', ''))

                if sender_domain and recipients and recipients != 'nan':
                    # Handle multiple recipients
                    recipient_list = []
                    if ',' in recipients:
                        recipient_list = [r.strip() for r in recipients.split(',') if r.strip()]
                    elif ';' in recipients:
                        recipient_list = [r.strip() for r in recipients.split(';') if r.strip()]
                    else:
                        recipient_list = [recipients.strip()]

                    for recipient in recipient_list:
                        recipient_domain = extract_domain(recipient)
                        if recipient_domain and sender_domain != recipient_domain:
                            pair = f"{sender_domain} -> {recipient_domain}"
                            domain_pairs.append(pair)
                            sender_recipient_pairs.append({
                                'sender': sender,
                                'recipient': recipient,
                                'sender_domain': sender_domain,
                                'recipient_domain': recipient_domain,
                                'pair': pair
                            })

            if not domain_pairs:
                return {
                    'bau_candidates': [],
                    'bau_percentage': 0,
                    'high_volume_pairs': [],
                    'unique_domains': 0,
                    'recommendations': ['No valid domain pairs found for BAU analysis']
                }

            # Count frequency of domain pairs
            pair_counts = {}
            for pair in domain_pairs:
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            # Calculate statistics
            total_pairs = len(domain_pairs)
            total_unique_pairs = len(pair_counts)

            # Adjust threshold based on data size
            if total_pairs < 10:
                high_volume_threshold = 2  # At least 2 emails for small datasets
            elif total_pairs < 50:
                high_volume_threshold = max(2, int(total_pairs * 0.15))  # 15% for medium datasets
            else:
                high_volume_threshold = max(3, int(total_pairs * 0.1))   # 10% for large datasets

            high_volume_pairs = []
            bau_candidates = []

            for pair, count in pair_counts.items():
                percentage = (count / total_pairs) * 100
                high_volume_pairs.append({
                    'pair': pair,
                    'count': count,
                    'percentage': round(percentage, 1)
                })

                # More flexible BAU criteria
                if count >= high_volume_threshold or percentage >= 5.0:
                    bau_candidates.append(pair)

            # Sort by count descending
            high_volume_pairs.sort(key=lambda x: x['count'], reverse=True)

            # Calculate BAU percentage more accurately
            bau_email_count = sum(pair_counts[pair] for pair in bau_candidates)
            bau_percentage = (bau_email_count / total_pairs) * 100 if total_pairs > 0 else 0

            # Get unique domains involved in BAU
            unique_domains = len(set([pair.split(' -> ')[1] for pair in bau_candidates]))

            # Generate detailed recommendations
            recommendations = []
            if len(bau_candidates) > 0:
                recommendations.append(f"Found {len(bau_candidates)} potential BAU communication patterns")
                recommendations.append(f"These patterns represent {bau_percentage:.1f}% of all email communications")

                if bau_percentage > 70:
                    recommendations.append("High BAU percentage - mostly routine business communications")
                    recommendations.append("Consider whitelisting top patterns to reduce false positives")
                elif bau_percentage > 30:
                    recommendations.append("Moderate BAU percentage - mixed communication patterns")
                    recommendations.append("Review patterns manually before implementing whitelist rules")
                else:
                    recommendations.append("Low BAU percentage - communications are diverse")
                    recommendations.append("Focus on highest volume patterns for potential whitelisting")

                if unique_domains > 5:
                    recommendations.append(f"Communications span {unique_domains} domains - indicates diverse business relationships")
            else:
                recommendations.append("No clear BAU patterns detected")
                recommendations.append("All communications appear to be unique or low-volume")
                recommendations.append("Manual review recommended for establishing whitelist rules")

            return {
                'bau_candidates': bau_candidates,
                'bau_percentage': round(bau_percentage, 1),
                'high_volume_pairs': high_volume_pairs[:15],  # Top 15 for better visibility
                'unique_domains': unique_domains,
                'recommendations': recommendations,
                'total_communications': total_pairs,
                'unique_patterns': total_unique_pairs,
                'bau_stats': {
                    'bau_candidates_count': len(bau_candidates),
                    'bau_percentage': round(bau_percentage, 1),
                    'high_volume_pairs': high_volume_pairs[:10]
                }
            }

        except Exception as e:
            self.logger.error(f"Error in BAU detection: {str(e)}")
            import traceback
            self.logger.error(f"BAU detection traceback: {traceback.format_exc()}")
            return {
                'bau_candidates': [],
                'bau_percentage': 0,
                'high_volume_pairs': [],
                'unique_domains': 0,
                'recommendations': [f"Error in BAU analysis: {str(e)}"],
                'total_communications': 0,
                'unique_patterns': 0
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
        # Check if we have data to analyze
        if processed_data is None or len(processed_data) == 0:
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
            if len(features) > 0:
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
            if len(df) == 0:
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

            if features.shape[0] == 0 or features.shape[1] == 0:
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

            # Calculate detailed attachment risk scores
            attachment_risk_scores = self._calculate_attachment_risk_scores(df)

            return {
                'anomaly_scores': anomaly_scores_list,
                'risk_levels': risk_levels_list,
                'interesting_patterns': patterns if isinstance(patterns, list) else [],
                'clusters': clusters_list,
                'attachment_classifications': attachment_classifications,
                'attachment_risk_scores': attachment_risk_scores,
                'insights': self._generate_insights(df, anomaly_scores_list, risk_levels_list, patterns)
            }

        except Exception as e:
            self.logger.error(f"Error in ML analysis: {str(e)}")
            # Return safe defaults
            target_length = len(df) if len(df) > 0 else 0
            return {
                'anomaly_scores': [0.0] * target_length,
                'risk_levels': ['Low'] * target_length,
                'interesting_patterns': [],
                'clusters': [-1] * target_length,
                'attachment_classifications': ['Unknown'] * target_length,
                'insights': {}
            }

    def _classify_attachments(self, df):
        """Classify attachments with advanced risk scoring for malicious intent and data exfiltration"""
        try:
            classifications = []
            risk_scores = []

            if 'attachments' not in df.columns:
                self.logger.info("No 'attachments' column found in DataFrame")
                return ['No Attachments'] * len(df)

            self.logger.info(f"Processing {len(df)} records for advanced attachment risk scoring")

            for idx, row in df.iterrows():
                attachments = row.get('attachments', '')

                if pd.isna(attachments) or str(attachments).strip() == '' or str(attachments).lower() == 'nan' or str(attachments).strip() == '-':
                    classifications.append('No Attachments')
                    continue

                # Parse multiple attachments (comma-separated)
                attachment_str = str(attachments).strip()
                attachment_list = [att.strip() for att in attachment_str.split(',') if att.strip()]

                # Calculate comprehensive attachment risk score
                risk_analysis = self._calculate_attachment_risk_score(attachment_list, row)

                classifications.append(risk_analysis['classification'])

            self.logger.info(f"Completed advanced attachment risk scoring for {len(classifications)} records")
            return classifications

        except Exception as e:
            self.logger.error(f"Error in advanced attachment classification: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return ['Unknown'] * len(df)

    def _calculate_attachment_risk_score(self, attachment_list, email_record):
        """Calculate comprehensive attachment risk score for malicious intent and data exfiltration"""
        try:
            if not attachment_list:
                return {
                    'classification': 'No Attachments',
                    'risk_score': 0.0,
                    'risk_factors': [],
                    'malicious_indicators': [],
                    'exfiltration_risk': 'None'
                }

            total_risk_score = 0.0
            risk_factors = []
            malicious_indicators = []

            # Define high-risk file extensions and patterns
            executable_extensions = ['.exe', '.bat', '.cmd', '.com', '.scr', '.vbs', '.js', '.jar', '.app', '.dmg']
            archive_extensions = ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2']
            office_macro_extensions = ['.docm', '.xlsm', '.pptm', '.xls', '.doc', '.ppt']
            sensitive_extensions = ['.sql', '.db', '.mdb', '.accdb', '.csv', '.xlsx', '.pdf']

            # Advanced malicious patterns
            malicious_patterns = [
                'invoice', 'receipt', 'payment', 'urgent', 'confidential', 'secure',
                'backup', 'dump', 'export', 'database', 'credentials', 'password',
                'login', 'account', 'financial', 'tax', 'salary', 'payroll',
                'personal', 'private', 'copy', 'duplicate', 'temp', 'tmp'
            ]

            # Data exfiltration indicators
            exfiltration_patterns = [
                'customer', 'client', 'contact', 'lead', 'prospect', 'employee',
                'staff', 'directory', 'list', 'database', 'backup', 'export',
                'dump', 'extract', 'report', 'summary', 'analysis', 'complete'
            ]

            for attachment in attachment_list:
                attachment_lower = attachment.lower()
                file_risk = 0.0

                # 1. File Extension Risk Analysis
                if any(ext in attachment_lower for ext in executable_extensions):
                    file_risk += 8.0
                    malicious_indicators.append(f"Executable file type: {attachment}")
                    risk_factors.append("High-risk executable attachment")

                if any(ext in attachment_lower for ext in office_macro_extensions):
                    file_risk += 5.0
                    malicious_indicators.append(f"Macro-enabled document: {attachment}")
                    risk_factors.append("Macro-enabled office document")

                if any(ext in attachment_lower for ext in archive_extensions):
                    file_risk += 3.0
                    risk_factors.append("Compressed archive file")

                # 2. Filename Pattern Analysis for Malicious Intent
                for pattern in malicious_patterns:
                    if pattern in attachment_lower:
                        file_risk += 2.0
                        malicious_indicators.append(f"Suspicious filename pattern '{pattern}': {attachment}")
                        risk_factors.append(f"Suspicious naming pattern: {pattern}")

                # 3. Data Exfiltration Risk Assessment
                exfiltration_score = 0.0
                for pattern in exfiltration_patterns:
                    if pattern in attachment_lower:
                        exfiltration_score += 1.5
                        risk_factors.append(f"Data exfiltration indicator: {pattern}")

                # 4. File Size and Volume Risk (simulated based on naming patterns)
                if any(word in attachment_lower for word in ['complete', 'full', 'all', 'entire', 'bulk']):
                    file_risk += 3.0
                    risk_factors.append("Large dataset indicator in filename")

                # 5. Obfuscation and Evasion Techniques
                if any(char in attachment for char in ['_', '.', '-']) and len([c for c in attachment if c.isalnum()]) < len(attachment) * 0.7:
                    file_risk += 2.0
                    malicious_indicators.append(f"Obfuscated filename: {attachment}")
                    risk_factors.append("Potentially obfuscated filename")

                # 6. Double Extension Check
                if attachment_lower.count('.') > 1:
                    extensions = attachment_lower.split('.')
                    if len(extensions) > 2 and extensions[-2] in ['pdf', 'doc', 'jpg', 'png']:
                        file_risk += 4.0
                        malicious_indicators.append(f"Double extension detected: {attachment}")
                        risk_factors.append("Double file extension (evasion technique)")

                # 7. Temporal Risk Assessment (based on email context)
                sender = email_record.get('sender', '').lower()
                if email_record.get('leaver') == 'yes':
                    file_risk += 4.0
                    risk_factors.append("Attachment from departing employee")

                # 8. Domain and Context Risk
                if 'wordlist_attachment' in email_record and email_record.get('wordlist_attachment') == 'yes':
                    file_risk += 3.0
                    risk_factors.append("Attachment matches security wordlist")

                total_risk_score += file_risk

            # Normalize risk score (0-100 scale)
            normalized_risk = min(100.0, (total_risk_score / len(attachment_list)) * 2)

            # Determine classification based on comprehensive risk assessment
            if normalized_risk >= 70:
                classification = 'Critical Risk'
                exfiltration_risk = 'High'
            elif normalized_risk >= 50:
                classification = 'High Risk'
                exfiltration_risk = 'Medium'
            elif normalized_risk >= 30:
                classification = 'Medium Risk'
                exfiltration_risk = 'Low'
            elif normalized_risk >= 15:
                classification = 'Low Risk'
                exfiltration_risk = 'Very Low'
            else:
                # Apply business/personal classification for low-risk files
                business_score = sum(1 for att in attachment_list for keyword in self.business_keywords if keyword in att.lower())
                personal_score = sum(1 for att in attachment_list for keyword in self.personal_keywords if keyword in att.lower())

                if business_score > personal_score:
                    classification = 'Business'
                elif personal_score > business_score:
                    classification = 'Personal'
                else:
                    classification = 'Normal'
                exfiltration_risk = 'None'

            return {
                'classification': classification,
                'risk_score': round(normalized_risk, 2),
                'risk_factors': list(set(risk_factors)),
                'malicious_indicators': malicious_indicators,
                'exfiltration_risk': exfiltration_risk,
                'attachment_count': len(attachment_list),
                'attachment_details': attachment_list
            }

        except Exception as e:
            self.logger.error(f"Error calculating attachment risk score: {str(e)}")
            return {
                'classification': 'Unknown',
                'risk_score': 0.0,
                'risk_factors': [],
                'malicious_indicators': [],
                'exfiltration_risk': 'Unknown'
            }

    def _process_batch(self, batch_df):
        """Process a batch of emails for large datasets"""
        try:
            # Prepare features for this batch
            features = self._prepare_features(batch_df)

            # Anomaly detection
            anomaly_scores = self._detect_anomalies(features)

            # Risk level classification
            risk_levels = self._classify_risk_levels(batch_df, anomaly_scores)

            # Clustering
            clusters = self._perform_clustering(features)

            # Attachment classification
            attachment_classifications = self._classify_attachments(batch_df)

            return {
                'anomaly_scores': anomaly_scores.tolist() if hasattr(anomaly_scores, 'tolist') else list(anomaly_scores),
                'risk_levels': risk_levels,
                'clusters': clusters.tolist() if hasattr(clusters, 'tolist') else list(clusters),
                'attachment_classifications': attachment_classifications
            }

        except Exception as e:
            self.logger.error(f"Error processing batch: {str(e)}")
            batch_size = len(batch_df)
            return {
                'anomaly_scores': [0.0] * batch_size,
                'risk_levels': ['Low'] * batch_size,
                'clusters': [-1] * batch_size,
                'attachment_classifications': ['Unknown'] * batch_size
            }

    def _calculate_attachment_risk_scores(self, df):
        """Calculate detailed attachment risk scores for all records"""
        try:
            risk_scores = []

            if 'attachments' not in df.columns:
                return [{'risk_score': 0.0, 'risk_level': 'No Attachments'}] * len(df)

            for idx, row in df.iterrows():
                attachments = row.get('attachments', '')

                if pd.isna(attachments) or str(attachments).strip() == '' or str(attachments).lower() == 'nan' or str(attachments).strip() == '-':
                    risk_scores.append({
                        'risk_score': 0.0,
                        'risk_level': 'No Attachments',
                        'risk_factors': [],
                        'malicious_indicators': [],
                        'exfiltration_risk': 'None'
                    })
                    continue

                attachment_list = [att.strip() for att in str(attachments).split(',') if att.strip()]
                risk_analysis = self._calculate_attachment_risk_score(attachment_list, row)

                risk_scores.append({
                    'risk_score': risk_analysis['risk_score'],
                    'risk_level': risk_analysis['classification'],
                    'risk_factors': risk_analysis['risk_factors'],
                    'malicious_indicators': risk_analysis['malicious_indicators'],
                    'exfiltration_risk': risk_analysis['exfiltration_risk'],
                    'attachment_count': risk_analysis['attachment_count']
                })

            self.logger.info(f"Calculated detailed risk scores for {len(risk_scores)} attachment records")
            return risk_scores

        except Exception as e:
            self.logger.error(f"Error calculating attachment risk scores: {str(e)}")
            return [{'risk_score': 0.0, 'risk_level': 'Unknown'}] * len(df)

    def _classify_single_attachment_list(self, attachment_list):
        """Classify attachments as Business, Personal, or Mixed"""
        try:
            if not attachment_list:
                return 'Unknown'

            business_score = 0
            personal_score = 0
            suspicious_score = 0

            for attachment in attachment_list:
                attachment = attachment.strip().lower()

                # Check for business keywords
                for keyword in self.business_keywords:
                    if keyword in attachment:
                        business_score += 1
                        break

                # Check for personal keywords
                for keyword in self.personal_keywords:
                    if keyword in attachment:
                        personal_score += 1
                        break

                # Check for suspicious keywords
                for keyword in self.suspicious_keywords:
                    if keyword in attachment:
                        suspicious_score += 1
                        break

                # Check file extensions for business patterns
                if any(ext in attachment for ext in ['.xlsx', '.pptx', '.docx', '.pdf']):
                    if any(word in attachment for word in ['report', 'proposal', 'contract', 'invoice']):
                        business_score += 0.5

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

            return classification

        except Exception as e:
            self.logger.error(f"Error classifying attachments: {str(e)}")
            return 'Unknown'