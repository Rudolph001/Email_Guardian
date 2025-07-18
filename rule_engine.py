import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any

class RuleEngine:
    """Rule processing engine for email filtering and classification"""

    def __init__(self):
        self.rules_file = 'data/rules.json'
        self.exclusion_rules_file = 'data/exclusion_rules.json'
        self.logger = logging.getLogger(__name__)
        self.operators = {
            'equals': self._equals,
            'contains': self._contains,
            'not_equals': self._not_equals,
            'in_list': self._in_list,
            'greater_than': self._greater_than,
            'less_than': self._less_than,
            'matches_pattern': self._matches_pattern
        }
        self.initialize_exclusion_rules_file()

    def initialize_exclusion_rules_file(self):
        """Initialize exclusion rules file if it doesn't exist"""
        if not os.path.exists(self.exclusion_rules_file):
            os.makedirs('data', exist_ok=True)
            # The exclusion_rules.json file was already created, so no need to create defaults

    @staticmethod
    def get_field_names():
        """Get all available field names from CSV imports for rule creation"""
        # Common email export field names from Tessian and similar systems
        return [
            '_time', 'sent_timestamp', 'sender', 'recipient', 'recipients', 'subject',
            'attachments', 'attachment_classification', 'wordlist_subject', 'wordlist_attachment',
            'leaver', 'status', 'blocked', 'direction', 'message_id', 'thread_id',
            'cc', 'bcc', 'reply_to', 'importance', 'sensitivity', 'size', 'has_attachments',
            'domain_classification', 'sender_domain', 'recipient_domain', 'external_recipients',
            'internal_recipients', 'attachment_count', 'attachment_types', 'attachment_names',
            'content_type', 'encryption_status', 'dlp_policy', 'compliance_status',
            'risk_score', 'threat_classification', 'quarantine_status', 'delivery_status',
            'spam_score', 'phishing_score', 'malware_detected', 'safe_links_status',
            'department', 'job_title', 'manager', 'location', 'employee_id', 'cost_center'
        ]

    @staticmethod
    def initialize_rules_file():
        """Initialize rules file with default rules"""
        rules_file = 'data/rules.json'
        if not os.path.exists(rules_file):
            default_rules = [
                {
                    'id': 1,
                    'name': 'High Risk Leavers',
                    'description': 'Flag emails from users marked as leavers',
                    'conditions': [
                        {'field': 'leaver', 'operator': 'equals', 'value': 'Yes'}
                    ],
                    'actions': [
                        {'type': 'mark_priority', 'value': 'High'},
                        {'type': 'escalate', 'value': True}
                    ],
                    'priority': 1,
                    'active': True,
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                },
                {
                    'id': 2,
                    'name': 'Wordlist Matches',
                    'description': 'Flag emails with wordlist matches in subject or attachments',
                    'conditions': [
                        {'field': 'wordlist_subject', 'operator': 'equals', 'value': 'Yes'},
                        {'field': 'wordlist_attachment', 'operator': 'equals', 'value': 'Yes', 'logic': 'OR'}
                    ],
                    'actions': [
                        {'type': 'mark_priority', 'value': 'Medium'},
                        {'type': 'notify', 'value': 'Content flagged by wordlist'}
                    ],
                    'priority': 2,
                    'active': True,
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                },
                {
                    'id': 3,
                    'name': 'Blocked Emails',
                    'description': 'Handle blocked emails',
                    'conditions': [
                        {'field': 'status', 'operator': 'equals', 'value': 'Blocked'}
                    ],
                    'actions': [
                        {'type': 'mark_priority', 'value': 'High'},
                        {'type': 'escalate', 'value': True}
                    ],
                    'priority': 1,
                    'active': True,
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                }
            ]

            with open(rules_file, 'w') as f:
                json.dump(default_rules, f, indent=2)

    def load_rules(self) -> List[Dict]:
        """Load all rules from file"""
        try:
            if os.path.exists(self.rules_file):
                with open(self.rules_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            self.logger.error(f"Error loading rules: {str(e)}")
            return []

    def save_rules(self, rules: List[Dict]) -> bool:
        """Save rules to file"""
        try:
            with open(self.rules_file, 'w') as f:
                json.dump(rules, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Error saving rules: {str(e)}")
            return False

    def create_rule(self, rule_data: Dict) -> Dict:
        """Create a new rule"""
        try:
            rules = self.load_rules()

            # Generate new ID
            max_id = max([rule.get('id', 0) for rule in rules]) if rules else 0
            new_id = max_id + 1

            new_rule = {
                'id': new_id,
                'name': rule_data['name'],
                'description': rule_data.get('description', ''),
                'conditions': rule_data.get('conditions', []),
                'actions': rule_data.get('actions', []),
                'priority': rule_data.get('priority', 1),
                'active': rule_data.get('active', True),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }

            rules.append(new_rule)

            if self.save_rules(rules):
                return {'success': True, 'rule_id': new_id}
            else:
                return {'success': False, 'error': 'Failed to save rule'}

        except Exception as e:
            self.logger.error(f"Error creating rule: {str(e)}")
            return {'success': False, 'error': str(e)}

    def update_rule(self, rule_id: int, rule_data: Dict) -> Dict:
        """Update an existing rule"""
        try:
            rules = self.load_rules()

            for i, rule in enumerate(rules):
                if rule['id'] == rule_id:
                    rules[i].update(rule_data)
                    rules[i]['updated_at'] = datetime.now().isoformat()

                    if self.save_rules(rules):
                        return {'success': True}
                    else:
                        return {'success': False, 'error': 'Failed to save rule'}

            return {'success': False, 'error': 'Rule not found'}

        except Exception as e:
            self.logger.error(f"Error updating rule: {str(e)}")
            return {'success': False, 'error': str(e)}

    def delete_rule(self, rule_id: int) -> Dict:
        """Delete a rule"""
        try:
            rules = self.load_rules()
            rules = [rule for rule in rules if rule['id'] != rule_id]

            if self.save_rules(rules):
                return {'success': True}
            else:
                return {'success': False, 'error': 'Failed to save rules'}

        except Exception as e:
            self.logger.error(f"Error deleting rule: {str(e)}")
            return {'success': False, 'error': str(e)}

    def process_email(self, email_data: Dict) -> Dict:
        """Process a single email against all active rules"""
        rules = self.load_rules()
        active_rules = [rule for rule in rules if rule.get('active', True)]

        # Sort by priority (lower number = higher priority)
        active_rules.sort(key=lambda x: x.get('priority', 999))

        results = {
            'matched_rules': [],
            'actions': [],
            'priority': 'Low',
            'escalate': False,
            'notifications': []
        }

        for rule in active_rules:
            if self._evaluate_conditions(email_data, rule['conditions']):
                results['matched_rules'].append({
                    'id': rule['id'],
                    'name': rule['name'],
                    'description': rule['description']
                })

                # Process actions
                for action in rule.get('actions', []):
                    self._process_action(action, results)

        return results

    def _evaluate_conditions(self, email_data: Dict, conditions: List[Dict]) -> bool:
        """Evaluate rule conditions against email data"""
        if not conditions:
            return True

        if len(conditions) == 1:
            # Single condition
            condition = conditions[0]
            field = condition.get('field', '')
            operator = condition.get('operator', 'equals')
            value = condition.get('value', '')
            field_value = email_data.get(field, '')
            
            if operator in self.operators:
                return self.operators[operator](field_value, value)
            return False

        # Multiple conditions - build expression
        result = None
        
        for i, condition in enumerate(conditions):
            field = condition.get('field', '')
            operator = condition.get('operator', 'equals')
            value = condition.get('value', '')
            logic = condition.get('logic', 'AND')
            
            # Get field value from email data
            field_value = email_data.get(field, '')
            
            # Evaluate current condition
            if operator in self.operators:
                condition_result = self.operators[operator](field_value, value)
            else:
                condition_result = False
            
            if i == 0:
                # First condition
                result = condition_result
            else:
                # Apply logic from current condition to combine with previous result
                if logic == 'AND':
                    result = result and condition_result
                elif logic == 'OR':
                    result = result or condition_result
        
        return result if result is not None else False

    def _process_action(self, action: Dict, results: Dict):
        """Process a single action"""
        action_type = action.get('type', '')
        action_value = action.get('value', '')

        if action_type == 'mark_priority':
            # Update priority if higher than current
            priority_levels = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
            current_priority = priority_levels.get(results['priority'], 1)
            new_priority = priority_levels.get(action_value, 1)

            if new_priority > current_priority:
                results['priority'] = action_value

        elif action_type == 'escalate':
            if action_value:
                results['escalate'] = True

        elif action_type == 'notify':
            results['notifications'].append(action_value)

        results['actions'].append(action)

    # Operator implementations
    def _equals(self, field_value: Any, condition_value: Any) -> bool:
        # If condition_value contains commas, treat it as a list for backwards compatibility
        if ',' in str(condition_value):
            values = [v.strip().lower() for v in str(condition_value).split(',') if v.strip()]
            return str(field_value).lower() in values
        return str(field_value).lower() == str(condition_value).lower()

    def _contains(self, field_value: Any, condition_value: Any) -> bool:
        return str(condition_value).lower() in str(field_value).lower()

    def _not_equals(self, field_value: Any, condition_value: Any) -> bool:
        return str(field_value).lower() != str(condition_value).lower()

    def _in_list(self, field_value: Any, condition_value: Any) -> bool:
        if isinstance(condition_value, list):
            return str(field_value).lower() in [str(v).lower() for v in condition_value]
        else:
            # Handle comma-separated string values
            values = [v.strip().lower() for v in str(condition_value).split(',') if v.strip()]
            return str(field_value).lower() in values

    def _greater_than(self, field_value: Any, condition_value: Any) -> bool:
        try:
            return float(field_value) > float(condition_value)
        except (ValueError, TypeError):
            return False

    def _less_than(self, field_value: Any, condition_value: Any) -> bool:
        try:
            return float(field_value) < float(condition_value)
        except (ValueError, TypeError):
            return False

    def _matches_pattern(self, field_value: Any, condition_value: Any) -> bool:
        import re
        try:
            pattern = re.compile(str(condition_value), re.IGNORECASE)
            return bool(pattern.search(str(field_value)))
        except re.error:
            return False

    @staticmethod
    def get_all_rules() -> List[Dict]:
        """Get all rules for display"""
        engine = RuleEngine()
        return engine.load_rules()

    def get_rule_results(self, session_id: str) -> Dict:
        """Get rule processing results for a session"""
        try:
            # This would typically load from a processed results file
            # For now, return basic structure
            return {
                'total_rules_applied': len(self.load_rules()),
                'emails_processed': 0,
                'rules_matched': 0,
                'escalations_triggered': 0
            }
        except Exception as e:
            self.logger.error(f"Error getting rule results: {str(e)}")
            return {
                'total_rules_applied': 0,
                'emails_processed': 0,
                'rules_matched': 0,
                'escalations_triggered': 0
            }

    def get_exclusion_rules(self) -> List[Dict]:
        """Get all active exclusion rules"""
        try:
            if not os.path.exists(self.exclusion_rules_file):
                return []
                
            with open(self.exclusion_rules_file, 'r') as f:
                all_rules = json.load(f)
                
            return [rule for rule in all_rules if rule.get('active', True)]
            
        except Exception as e:
            self.logger.error(f"Error loading exclusion rules: {str(e)}")
            return []

    def save_exclusion_rules(self, rules: List[Dict]) -> bool:
        """Save exclusion rules to file"""
        try:
            os.makedirs('data', exist_ok=True)
            with open(self.exclusion_rules_file, 'w') as f:
                json.dump(rules, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Error saving exclusion rules: {str(e)}")
            return False

    def add_exclusion_rule(self, name: str, description: str, field: str, operator: str, value: str, case_sensitive: bool = False) -> Dict:
        """Add a new exclusion rule"""
        try:
            rules = self.get_all_exclusion_rules()
            
            new_id = max([rule.get('id', 0) for rule in rules], default=0) + 1
            
            new_rule = {
                'id': new_id,
                'name': name.strip(),
                'description': description.strip(),
                'field': field,
                'operator': operator,
                'value': value,
                'case_sensitive': case_sensitive,
                'active': True,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            rules.append(new_rule)
            
            if self.save_exclusion_rules(rules):
                self.logger.info(f"Added exclusion rule: {name}")
                return {'success': True, 'rule': new_rule}
            else:
                return {'success': False, 'error': 'Failed to save rule'}
                
        except Exception as e:
            self.logger.error(f"Error adding exclusion rule: {str(e)}")
            return {'success': False, 'error': str(e)}

    def get_all_exclusion_rules(self) -> List[Dict]:
        """Get all exclusion rules (active and inactive)"""
        try:
            if not os.path.exists(self.exclusion_rules_file):
                return []
                
            with open(self.exclusion_rules_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Error loading all exclusion rules: {str(e)}")
            return []

    def delete_exclusion_rule(self, rule_id: int) -> Dict:
        """Delete an exclusion rule"""
        try:
            rules = self.get_all_exclusion_rules()
            rules = [rule for rule in rules if rule.get('id') != rule_id]
            
            if self.save_exclusion_rules(rules):
                self.logger.info(f"Deleted exclusion rule ID: {rule_id}")
                return {'success': True}
            else:
                return {'success': False, 'error': 'Failed to save changes'}
                
        except Exception as e:
            self.logger.error(f"Error deleting exclusion rule: {str(e)}")
            return {'success': False, 'error': str(e)}

    def toggle_exclusion_rule(self, rule_id: int) -> Dict:
        """Toggle active status of an exclusion rule"""
        try:
            rules = self.get_all_exclusion_rules()
            
            for rule in rules:
                if rule.get('id') == rule_id:
                    rule['active'] = not rule.get('active', True)
                    rule['updated_at'] = datetime.now().isoformat()
                    break
            
            if self.save_exclusion_rules(rules):
                self.logger.info(f"Toggled exclusion rule ID: {rule_id}")
                return {'success': True}
            else:
                return {'success': False, 'error': 'Failed to save changes'}
                
        except Exception as e:
            self.logger.error(f"Error toggling exclusion rule: {str(e)}")
            return {'success': False, 'error': str(e)}

    def should_exclude_record(self, record: Dict) -> bool:
        """Check if a record should be excluded based on exclusion rules"""
        try:
            exclusion_rules = self.get_exclusion_rules()
            
            for rule in exclusion_rules:
                field = rule.get('field')
                operator = rule.get('operator')
                value = rule.get('value')
                case_sensitive = rule.get('case_sensitive', False)
                
                if not field or not operator or value is None:
                    continue
                
                record_value = record.get(field)
                if record_value is None:
                    continue
                
                record_value = str(record_value)
                rule_value = str(value)
                
                if not case_sensitive:
                    record_value = record_value.lower()
                    rule_value = rule_value.lower()
                
                if operator == 'equals' and record_value == rule_value:
                    self.logger.info(f"Excluding record: {rule['name']} matched ({field}={record_value})")
                    return True
                elif operator == 'contains' and rule_value in record_value:
                    self.logger.info(f"Excluding record: {rule['name']} matched ({field} contains {rule_value})")
                    return True
                elif operator == 'not_equals' and record_value != rule_value:
                    self.logger.info(f"Excluding record: {rule['name']} matched ({field}!={rule_value})")
                    return True
                elif operator == 'starts_with' and record_value.startswith(rule_value):
                    self.logger.info(f"Excluding record: {rule['name']} matched ({field} starts with {rule_value})")
                    return True
                elif operator == 'ends_with' and record_value.endswith(rule_value):
                    self.logger.info(f"Excluding record: {rule['name']} matched ({field} ends with {rule_value})")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking exclusion rules: {str(e)}")
            return False