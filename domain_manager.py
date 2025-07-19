import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional

class DomainManager:
    """Manage domain classifications for email analysis"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.domains_file = 'data/domain_classifications.json'
        self.initialize_domains_file()

    def initialize_domains_file(self):
        """Initialize domain classifications file if it doesn't exist"""
        os.makedirs('data', exist_ok=True)
        default_domains = {
            "trusted": [
                "company.com",
                "corporate.com",
                "business.com",
                "rdcomputers.co.za"
            ],
            "corporate": [
                "microsoft.com",
                "apple.com",
                "google.com",
                "amazon.com",
                "salesforce.com",
                "linkedin.com",
                "office365.com"
            ],
            "personal": [
                "gmail.com",
                "yahoo.com",
                "hotmail.com",
                "outlook.com",
                "icloud.com",
                "aol.com",
                "protonmail.com"
            ],
            "public": [
                "gov.uk",
                "gov.us",
                "edu.com",
                "edu.uk",
                "ac.uk",
                "org.uk",
                "mil.us",
                "state.gov"
            ],
            "suspicious": [
                "tempmail.com",
                "10minutemail.com",
                "guerrillamail.com",
                "throwaway.email",
                "mailinator.com",
                "yopmail.com"
            ],
            "updated_at": datetime.now().isoformat()
        }
        self.save_domains(default_domains)
        self.logger.info("Initialized domain classifications with default data")

    def get_domains(self) -> Dict:
        """Get all domain classifications"""
        try:
            if not os.path.exists(self.domains_file):
                self.logger.info("Domain classifications file not found, initializing...")
                self.initialize_domains_file()
            
            with open(self.domains_file, 'r') as f:
                data = json.load(f)
                
            # Validate the structure
            if not isinstance(data, dict):
                raise ValueError("Invalid domain classifications format")
                
            # Ensure all required keys exist
            required_keys = ["trusted", "corporate", "personal", "public", "suspicious"]
            for key in required_keys:
                if key not in data:
                    data[key] = []
                    
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading domain classifications: {str(e)}")
            # Re-initialize with defaults
            self.initialize_domains_file()
            return {
                "trusted": [
                    "company.com",
                    "corporate.com", 
                    "business.com",
                    "rdcomputers.co.za"
                ],
                "corporate": [
                    "microsoft.com",
                    "apple.com",
                    "google.com",
                    "amazon.com",
                    "salesforce.com",
                    "linkedin.com",
                    "office365.com"
                ],
                "personal": [
                    "gmail.com",
                    "yahoo.com",
                    "hotmail.com",
                    "outlook.com",
                    "icloud.com",
                    "aol.com",
                    "protonmail.com"
                ],
                "public": [
                    "gov.uk",
                    "gov.us",
                    "edu.com",
                    "edu.uk",
                    "ac.uk",
                    "org.uk",
                    "mil.us",
                    "state.gov"
                ],
                "suspicious": [
                    "tempmail.com",
                    "10minutemail.com",
                    "guerrillamail.com",
                    "throwaway.email",
                    "mailinator.com",
                    "yopmail.com"
                ],
                "updated_at": datetime.now().isoformat()
            }

    def save_domains(self, domains: Dict) -> bool:
        """Save domain classifications"""
        try:
            domains['updated_at'] = datetime.now().isoformat()
            with open(self.domains_file, 'w') as f:
                json.dump(domains, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Error saving domain classifications: {str(e)}")
            return False

    def add_domain(self, category: str, domain: str) -> Dict:
        """Add a domain to a specific category"""
        try:
            if category not in ['trusted', 'corporate', 'personal', 'public', 'suspicious']:
                return {'success': False, 'error': 'Invalid category'}

            domain = domain.lower().strip()
            if not domain or '.' not in domain:
                return {'success': False, 'error': 'Invalid domain format'}

            domains = self.get_domains()
            
            # Check if domain already exists in any category
            for cat, domain_list in domains.items():
                if cat != 'updated_at' and domain in domain_list:
                    if cat == category:
                        return {'success': False, 'error': f'Domain already exists in {category} category'}
                    else:
                        return {'success': False, 'error': f'Domain already exists in {cat} category'}

            domains[category].append(domain)
            if self.save_domains(domains):
                self.logger.info(f"Added domain {domain} to {category} category")
                return {'success': True, 'message': f'Domain added to {category} category'}
            else:
                return {'success': False, 'error': 'Failed to save domain classifications'}

        except Exception as e:
            self.logger.error(f"Error adding domain: {str(e)}")
            return {'success': False, 'error': str(e)}

    def remove_domain(self, category: str, domain: str) -> Dict:
        """Remove a domain from a specific category"""
        try:
            if category not in ['trusted', 'corporate', 'personal', 'public', 'suspicious']:
                return {'success': False, 'error': 'Invalid category'}

            domains = self.get_domains()
            
            if domain not in domains[category]:
                return {'success': False, 'error': f'Domain not found in {category} category'}

            domains[category].remove(domain)
            if self.save_domains(domains):
                self.logger.info(f"Removed domain {domain} from {category} category")
                return {'success': True, 'message': f'Domain removed from {category} category'}
            else:
                return {'success': False, 'error': 'Failed to save domain classifications'}

        except Exception as e:
            self.logger.error(f"Error removing domain: {str(e)}")
            return {'success': False, 'error': str(e)}

    def classify_domain(self, domain: str) -> str:
        """Classify a domain based on current classifications"""
        try:
            domain = domain.lower().strip()
            domains = self.get_domains()

            # Check exact matches first
            for category, domain_list in domains.items():
                if category != 'updated_at' and domain in domain_list:
                    return category.title()

            # Check for patterns
            if domain.endswith('.gov') or domain.endswith('.mil'):
                return 'Public'
            elif domain.endswith('.edu') or domain.endswith('.ac.uk'):
                return 'Public'
            elif any(suspicious in domain for suspicious in ['temp', 'throw', '10min', 'guerr']):
                return 'Suspicious'
            elif domain in ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']:
                return 'Personal'
            
            return 'Unknown'

        except Exception as e:
            self.logger.error(f"Error classifying domain {domain}: {str(e)}")
            return 'Unknown'

    def get_domain_stats(self) -> Dict:
        """Get statistics about domain classifications"""
        try:
            domains = self.get_domains()
            stats = {}
            total = 0
            
            for category in ['trusted', 'corporate', 'personal', 'public', 'suspicious']:
                count = len(domains.get(category, []))
                stats[category] = count
                total += count
            
            stats['total'] = total
            return stats

        except Exception as e:
            self.logger.error(f"Error getting domain stats: {str(e)}")
            return {'trusted': 0, 'corporate': 0, 'personal': 0, 'public': 0, 'suspicious': 0, 'total': 0}

    def reset_to_defaults(self) -> Dict:
        """Reset domain classifications to default values"""
        try:
            self.initialize_domains_file()
            self.logger.info("Reset domain classifications to defaults")
            return {'success': True, 'message': 'Domain classifications reset to defaults'}
        except Exception as e:
            self.logger.error(f"Error resetting domains to defaults: {str(e)}")
            return {'success': False, 'error': str(e)}

    def export_config(self) -> Dict:
        """Export current domain configuration"""
        try:
            return self.get_domains()
        except Exception as e:
            self.logger.error(f"Error exporting domain config: {str(e)}")
            return {}