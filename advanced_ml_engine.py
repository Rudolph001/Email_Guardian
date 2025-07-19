import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import datetime, timedelta
import json
import logging
import re
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
import networkx as nx

class AdvancedMLEngine:
    """Advanced ML engine for comprehensive email analysis and risk assessment"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # ML models
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Text analysis
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        self.count_vectorizer = CountVectorizer(max_features=500, stop_words='english')
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        
        # Risk indicators for justification analysis
        self.high_risk_justification_patterns = [
            r'urgent.*deadline', r'time.*sensitive', r'immediate.*action',
            r'confidential.*urgent', r'emergency.*transfer', r'bypas.*approval',
            r'special.*arrangement', r'off.*record', r'personal.*favor',
            r'one.*time.*exception', r'temporary.*access', r'quick.*fix',
            r'management.*request', r'director.*order', r'ceo.*instruction'
        ]
        
        self.suspicious_behavior_patterns = [
            r'working.*late', r'weekend.*work', r'holiday.*access',
            r'remote.*location', r'unusual.*time', r'after.*hours',
            r'before.*leaving', r'last.*day', r'resignation.*notice'
        ]

    def analyze_comprehensive_email_data(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive ML analysis of email data including:
        - Justification sentiment and risk analysis
        - Recipient domain behavior analysis  
        - Communication pattern analysis
        - Temporal anomaly detection
        - Network analysis of email flows
        """
        try:
            results = {
                'justification_analysis': self._analyze_justifications(df),
                'recipient_domain_analysis': self._analyze_recipient_domains(df),
                'communication_patterns': self._analyze_communication_patterns(df),
                'temporal_anomalies': self._analyze_temporal_patterns(df),
                'network_analysis': self._analyze_email_networks(df),
                'behavioral_clustering': self._cluster_user_behavior(df),
                'risk_indicators': self._detect_risk_indicators(df),
                'departmental_analysis': self._analyze_departmental_patterns(df)
            }
            
            # Enhanced anomaly scoring with multiple factors
            results['enhanced_anomaly_scores'] = self._calculate_enhanced_anomaly_scores(df, results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {str(e)}")
            return {}

    def _analyze_justifications(self, df: pd.DataFrame) -> Dict:
        """Advanced analysis of justification field using NLP and pattern matching"""
        if 'justification' not in df.columns or df['justification'].isna().all():
            return {'status': 'no_justification_data'}
        
        justifications = df['justification'].fillna('').astype(str)
        
        analysis = {
            'risk_scores': [],
            'sentiment_analysis': [],
            'pattern_matches': [],
            'urgency_indicators': [],
            'deception_indicators': [],
            'topic_modeling': {},
            'length_analysis': {},
            'language_complexity': []
        }
        
        for just in justifications:
            if not just or just == '-':
                analysis['risk_scores'].append(0)
                analysis['sentiment_analysis'].append('neutral')
                analysis['pattern_matches'].append([])
                analysis['urgency_indicators'].append(False)
                analysis['deception_indicators'].append(False)
                analysis['language_complexity'].append(0)
                continue
                
            # Risk pattern matching
            risk_score = 0
            matched_patterns = []
            
            for pattern in self.high_risk_justification_patterns:
                if re.search(pattern, just.lower()):
                    risk_score += 2
                    matched_patterns.append(pattern)
            
            for pattern in self.suspicious_behavior_patterns:
                if re.search(pattern, just.lower()):
                    risk_score += 1
                    matched_patterns.append(pattern)
            
            # Urgency detection
            urgency_words = ['urgent', 'asap', 'immediate', 'emergency', 'critical', 'deadline']
            has_urgency = any(word in just.lower() for word in urgency_words)
            
            # Deception indicators
            deception_words = ['mistake', 'error', 'forgot', 'accident', 'misunderstood', 'confusion']
            has_deception = any(word in just.lower() for word in deception_words)
            
            # Language complexity (readability proxy)
            complexity = len(just.split()) / max(len(just.split('.')), 1)  # Words per sentence
            
            analysis['risk_scores'].append(min(risk_score, 10))
            analysis['pattern_matches'].append(matched_patterns)
            analysis['urgency_indicators'].append(has_urgency)
            analysis['deception_indicators'].append(has_deception)
            analysis['language_complexity'].append(complexity)
        
        # Length analysis
        lengths = [len(just) for just in justifications if just and just != '-']
        if lengths:
            analysis['length_analysis'] = {
                'avg_length': np.mean(lengths),
                'std_length': np.std(lengths),
                'long_justifications': len([l for l in lengths if l > np.mean(lengths) + 2*np.std(lengths)]),
                'short_justifications': len([l for l in lengths if l < 50])
            }
        
        return analysis

    def _analyze_recipient_domains(self, df: pd.DataFrame) -> Dict:
        """Comprehensive analysis of recipient domain patterns and behaviors"""
        if 'recipients_email_domain' not in df.columns:
            return {'status': 'no_domain_data'}
            
        domains = df['recipients_email_domain'].fillna('').astype(str)
        
        analysis = {
            'domain_frequency': Counter(),
            'external_domains': set(),
            'suspicious_domains': [],
            'new_domains': [],
            'domain_risk_scores': {},
            'communication_patterns': defaultdict(list),
            'temporal_domain_usage': defaultdict(list)
        }
        
        # Domain categorization
        internal_indicators = ['company.com', 'corp.com', 'internal']
        suspicious_indicators = ['.tk', '.ml', '.ga', 'temp', 'disposable', '10minute']
        
        for idx, domain_str in enumerate(domains):
            if not domain_str or domain_str == '-':
                continue
                
            # Handle multiple domains
            domain_list = domain_str.split(',') if ',' in domain_str else [domain_str]
            
            for domain in domain_list:
                domain = domain.strip().lower()
                if not domain:
                    continue
                    
                analysis['domain_frequency'][domain] += 1
                
                # External domain detection
                if not any(indicator in domain for indicator in internal_indicators):
                    analysis['external_domains'].add(domain)
                
                # Suspicious domain detection
                if any(indicator in domain for indicator in suspicious_indicators):
                    analysis['suspicious_domains'].append(domain)
                
                # Risk scoring
                risk_score = 0
                if domain in analysis['suspicious_domains']:
                    risk_score += 5
                if analysis['domain_frequency'][domain] == 1:  # New domain
                    risk_score += 2
                if len(domain.split('.')) > 3:  # Subdomain complexity
                    risk_score += 1
                    
                analysis['domain_risk_scores'][domain] = risk_score
                
                # Temporal patterns
                if '_time' in df.columns:
                    timestamp = df.iloc[idx]['_time']
                    analysis['temporal_domain_usage'][domain].append(timestamp)
        
        # Communication pattern analysis
        analysis['top_external_domains'] = dict(Counter({k: v for k, v in analysis['domain_frequency'].items() 
                                                       if k in analysis['external_domains']}).most_common(10))
        
        analysis['domain_diversity'] = len(analysis['external_domains'])
        analysis['external_ratio'] = len(analysis['external_domains']) / max(len(analysis['domain_frequency']), 1)
        
        return analysis

    def _analyze_communication_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze email communication patterns and behaviors"""
        patterns = {
            'sender_patterns': defaultdict(list),
            'recipient_patterns': defaultdict(list),
            'volume_patterns': {},
            'frequency_analysis': {},
            'unusual_communications': []
        }
        
        if 'sender' in df.columns and 'recipients' in df.columns:
            for idx, row in df.iterrows():
                sender = str(row.get('sender', '')).strip().lower()
                recipients = str(row.get('recipients', '')).strip().lower()
                
                if sender and sender != '-':
                    patterns['sender_patterns'][sender].append(idx)
                
                if recipients and recipients != '-':
                    recipient_list = recipients.split(',') if ',' in recipients else [recipients]
                    for recipient in recipient_list:
                        recipient = recipient.strip()
                        if recipient:
                            patterns['recipient_patterns'][recipient].append(idx)
            
            # Volume analysis
            patterns['volume_patterns'] = {
                'top_senders': dict(Counter({k: len(v) for k, v in patterns['sender_patterns'].items()}).most_common(10)),
                'top_recipients': dict(Counter({k: len(v) for k, v in patterns['recipient_patterns'].items()}).most_common(10)),
                'total_unique_senders': len(patterns['sender_patterns']),
                'total_unique_recipients': len(patterns['recipient_patterns'])
            }
        
        return patterns

    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect temporal anomalies in email patterns"""
        if '_time' not in df.columns:
            return {'status': 'no_temporal_data'}
        
        temporal_analysis = {
            'hourly_patterns': defaultdict(int),
            'daily_patterns': defaultdict(int),
            'weekend_activity': 0,
            'after_hours_activity': 0,
            'anomalous_times': [],
            'burst_patterns': []
        }
        
        try:
            for timestamp_str in df['_time']:
                if not timestamp_str or timestamp_str == '-':
                    continue
                    
                # Parse timestamp (handle various formats)
                try:
                    if isinstance(timestamp_str, str):
                        # Try common timestamp formats
                        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%m/%d/%Y %H:%M:%S']:
                            try:
                                dt = datetime.strptime(timestamp_str, fmt)
                                break
                            except ValueError:
                                continue
                        else:
                            continue  # Skip if no format matches
                    else:
                        dt = pd.to_datetime(timestamp_str)
                    
                    hour = dt.hour
                    day = dt.weekday()  # 0=Monday, 6=Sunday
                    
                    temporal_analysis['hourly_patterns'][hour] += 1
                    temporal_analysis['daily_patterns'][day] += 1
                    
                    # Weekend activity (Saturday=5, Sunday=6)
                    if day >= 5:
                        temporal_analysis['weekend_activity'] += 1
                    
                    # After hours (before 8 AM or after 6 PM)
                    if hour < 8 or hour > 18:
                        temporal_analysis['after_hours_activity'] += 1
                        temporal_analysis['anomalous_times'].append(timestamp_str)
                        
                except Exception as e:
                    self.logger.debug(f"Could not parse timestamp {timestamp_str}: {e}")
                    continue
            
            # Calculate percentages
            total_emails = len(df)
            if total_emails > 0:
                temporal_analysis['weekend_percentage'] = (temporal_analysis['weekend_activity'] / total_emails) * 100
                temporal_analysis['after_hours_percentage'] = (temporal_analysis['after_hours_activity'] / total_emails) * 100
            
        except Exception as e:
            self.logger.error(f"Error in temporal analysis: {e}")
        
        return temporal_analysis

    def _analyze_email_networks(self, df: pd.DataFrame) -> Dict:
        """Create network analysis of email communications"""
        if 'sender' not in df.columns or 'recipients' not in df.columns:
            return {'status': 'insufficient_data'}
        
        # Create network graph
        G = nx.DiGraph()
        
        for _, row in df.iterrows():
            sender = str(row.get('sender', '')).strip().lower()
            recipients = str(row.get('recipients', '')).strip().lower()
            
            if sender and sender != '-' and recipients and recipients != '-':
                recipient_list = recipients.split(',') if ',' in recipients else [recipients]
                for recipient in recipient_list:
                    recipient = recipient.strip()
                    if recipient:
                        if G.has_edge(sender, recipient):
                            G[sender][recipient]['weight'] += 1
                        else:
                            G.add_edge(sender, recipient, weight=1)
        
        analysis = {
            'network_size': G.number_of_nodes(),
            'edge_count': G.number_of_edges(),
            'density': nx.density(G) if G.number_of_nodes() > 0 else 0,
            'central_nodes': {},
            'communities': [],
            'isolated_communications': []
        }
        
        if G.number_of_nodes() > 0:
            # Centrality measures
            try:
                centrality = nx.degree_centrality(G)
                analysis['central_nodes'] = dict(sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10])
            except:
                analysis['central_nodes'] = {}
            
            # Find isolated or unusual communication patterns
            for node in G.nodes():
                if G.degree(node) == 1:  # Only one connection
                    analysis['isolated_communications'].append(node)
        
        return analysis

    def _cluster_user_behavior(self, df: pd.DataFrame) -> Dict:
        """Cluster users based on their email behavior patterns"""
        try:
            # Feature engineering for clustering
            features = []
            user_features = defaultdict(lambda: {
                'email_count': 0,
                'avg_subject_length': 0,
                'has_attachments_ratio': 0,
                'external_ratio': 0,
                'weekend_ratio': 0,
                'after_hours_ratio': 0,
                'risk_score_avg': 0
            })
            
            # Aggregate features by sender
            for _, row in df.iterrows():
                sender = str(row.get('sender', '')).strip().lower()
                if not sender or sender == '-':
                    continue
                
                user_features[sender]['email_count'] += 1
                
                # Subject length
                subject = str(row.get('subject', ''))
                if subject and subject != '-':
                    user_features[sender]['avg_subject_length'] += len(subject)
                
                # Attachments
                attachments = str(row.get('attachments', ''))
                if attachments and attachments != '-':
                    user_features[sender]['has_attachments_ratio'] += 1
            
            # Convert to feature matrix
            if len(user_features) > 3:  # Need minimum samples for clustering
                feature_matrix = []
                user_list = []
                
                for user, features_dict in user_features.items():
                    if features_dict['email_count'] > 0:
                        # Normalize features
                        features_dict['avg_subject_length'] /= features_dict['email_count']
                        features_dict['has_attachments_ratio'] /= features_dict['email_count']
                        
                        feature_vector = [
                            features_dict['email_count'],
                            features_dict['avg_subject_length'],
                            features_dict['has_attachments_ratio']
                        ]
                        
                        feature_matrix.append(feature_vector)
                        user_list.append(user)
                
                if len(feature_matrix) > 3:
                    # Perform clustering
                    feature_matrix = np.array(feature_matrix)
                    scaled_features = self.scaler.fit_transform(feature_matrix)
                    
                    clusters = self.kmeans.fit_predict(scaled_features)
                    
                    # Analyze clusters
                    cluster_analysis = defaultdict(list)
                    for user, cluster in zip(user_list, clusters):
                        cluster_analysis[cluster].append(user)
                    
                    return {
                        'clusters': dict(cluster_analysis),
                        'cluster_centers': self.kmeans.cluster_centers_.tolist(),
                        'total_users_clustered': len(user_list)
                    }
            
            return {'status': 'insufficient_data_for_clustering'}
            
        except Exception as e:
            self.logger.error(f"Error in behavioral clustering: {e}")
            return {'error': str(e)}

    def _detect_risk_indicators(self, df: pd.DataFrame) -> Dict:
        """Detect various risk indicators across all data fields"""
        risk_indicators = {
            'high_risk_emails': [],
            'suspicious_patterns': [],
            'data_exfiltration_indicators': [],
            'insider_threat_markers': [],
            'policy_violations': []
        }
        
        for idx, row in df.iterrows():
            risk_score = 0
            risk_factors = []
            
            # Check for leaving employee indicators
            if row.get('leaver', '').lower() == 'yes':
                risk_score += 5
                risk_factors.append('departing_employee')
            
            # Check policy violations
            policy = str(row.get('policy_name', '')).lower()
            if 'violation' in policy or 'blocked' in policy:
                risk_score += 3
                risk_factors.append('policy_violation')
            
            # Check for suspicious attachments
            attachments = str(row.get('attachments', '')).lower()
            suspicious_extensions = ['.exe', '.bat', '.scr', '.zip', '.rar']
            if any(ext in attachments for ext in suspicious_extensions):
                risk_score += 2
                risk_factors.append('suspicious_attachment')
            
            # Check subject for suspicious content
            subject = str(row.get('subject', '')).lower()
            suspicious_subjects = ['urgent', 'confidential', 'personal', 'resignation']
            if any(word in subject for word in suspicious_subjects):
                risk_score += 1
                risk_factors.append('suspicious_subject')
            
            if risk_score >= 5:
                risk_indicators['high_risk_emails'].append({
                    'index': idx,
                    'risk_score': risk_score,
                    'risk_factors': risk_factors,
                    'sender': row.get('sender', ''),
                    'subject': row.get('subject', '')
                })
        
        return risk_indicators

    def _analyze_departmental_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze patterns by department and business unit"""
        if 'department' not in df.columns and 'bunit' not in df.columns:
            return {'status': 'no_departmental_data'}
        
        dept_analysis = {
            'department_email_volumes': defaultdict(int),
            'department_risk_scores': defaultdict(list),
            'cross_department_communications': defaultdict(set),
            'high_risk_departments': []
        }
        
        for _, row in df.iterrows():
            dept = str(row.get('department', '')).strip()
            bunit = str(row.get('bunit', '')).strip()
            
            # Use department or business unit
            org_unit = dept if dept and dept != '-' else bunit
            
            if org_unit and org_unit != '-':
                dept_analysis['department_email_volumes'][org_unit] += 1
                
                # Calculate risk score for this email
                risk_score = 0
                if row.get('leaver', '').lower() == 'yes':
                    risk_score += 3
                if 'urgent' in str(row.get('subject', '')).lower():
                    risk_score += 1
                
                dept_analysis['department_risk_scores'][org_unit].append(risk_score)
        
        # Calculate average risk scores
        for dept, scores in dept_analysis['department_risk_scores'].items():
            avg_risk = np.mean(scores) if scores else 0
            if avg_risk > 2:
                dept_analysis['high_risk_departments'].append((dept, avg_risk))
        
        return dept_analysis

    def _calculate_enhanced_anomaly_scores(self, df: pd.DataFrame, analysis_results: Dict) -> List[float]:
        """Calculate comprehensive anomaly scores combining multiple ML analyses"""
        enhanced_scores = []
        
        for idx, row in df.iterrows():
            score = 0.0
            
            # Justification risk score
            if 'justification_analysis' in analysis_results and len(analysis_results['justification_analysis'].get('risk_scores', [])) > idx:
                score += analysis_results['justification_analysis']['risk_scores'][idx] * 0.3
            
            # Domain risk score
            domain = str(row.get('recipients_email_domain', '')).lower()
            if 'recipient_domain_analysis' in analysis_results:
                domain_scores = analysis_results['recipient_domain_analysis'].get('domain_risk_scores', {})
                if domain in domain_scores:
                    score += domain_scores[domain] * 0.2
            
            # Temporal anomaly
            if 'temporal_anomalies' in analysis_results:
                timestamp = str(row.get('_time', ''))
                if timestamp in analysis_results['temporal_anomalies'].get('anomalous_times', []):
                    score += 2.0
            
            # Risk indicators
            if row.get('leaver', '').lower() == 'yes':
                score += 3.0
            
            if 'urgent' in str(row.get('subject', '')).lower():
                score += 1.0
            
            # Normalize to 0-10 scale
            enhanced_scores.append(min(score, 10.0))
        
        return enhanced_scores

    def generate_ml_insights_report(self, analysis_results: Dict) -> Dict:
        """Generate comprehensive insights report from ML analysis"""
        insights = {
            'executive_summary': {},
            'key_findings': [],
            'risk_recommendations': [],
            'behavioral_insights': {},
            'technical_metrics': {}
        }
        
        # Executive summary
        total_analyses = len([k for k, v in analysis_results.items() if isinstance(v, dict) and v.get('status') != 'no_data'])
        insights['executive_summary'] = {
            'total_ml_analyses_performed': total_analyses,
            'high_risk_areas_identified': 0,
            'anomalies_detected': len(analysis_results.get('enhanced_anomaly_scores', [])),
            'recommendation_priority': 'medium'
        }
        
        # Key findings from each analysis
        if 'justification_analysis' in analysis_results:
            just_analysis = analysis_results['justification_analysis']
            if 'risk_scores' in just_analysis:
                high_risk_justifications = len([s for s in just_analysis['risk_scores'] if s > 5])
                insights['key_findings'].append(f"Found {high_risk_justifications} high-risk justifications requiring review")
        
        if 'recipient_domain_analysis' in analysis_results:
            domain_analysis = analysis_results['recipient_domain_analysis']
            if 'suspicious_domains' in domain_analysis:
                suspicious_count = len(domain_analysis['suspicious_domains'])
                insights['key_findings'].append(f"Identified {suspicious_count} suspicious recipient domains")
        
        return insights