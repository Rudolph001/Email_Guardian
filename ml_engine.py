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

class MLEngine:
    """Advanced ML engine for email anomaly detection and analysis"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.label_encoder = LabelEncoder()
        self.logger = logging.getLogger(__name__)
    
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
            
            if df.empty:
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
    
    def get_insights(self, processed_data):
        """Get ML insights for dashboard display"""
        if not processed_data:
            return {
                'total_emails': 0,
                'risk_distribution': {},
                'anomaly_summary': {},
                'pattern_summary': {},
                'recommendations': []
            }
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(processed_data)
        
        # Perform ML analysis
        ml_results = self.analyze_emails(df)
        
        return ml_results['insights']
