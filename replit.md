# Email Guardian - Tessian Email Exfiltration Detection System

## Overview

Email Guardian is a full-stack Flask application designed to ingest Tessian email CSV exports, detect potential data exfiltration, and manage security escalations. The system processes email data through machine learning algorithms, applies customizable rules, and provides a comprehensive dashboard for security analysts to review and manage email security incidents.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application follows a modular Flask architecture with the following core components:

### Frontend Architecture
- **Template Engine**: Jinja2 templates with Bootstrap 5 for responsive UI
- **Static Assets**: CSS and JavaScript files for styling and client-side functionality
- **Charts**: Chart.js for data visualization and analytics dashboards

### Backend Architecture
- **Framework**: Flask with SQLAlchemy ORM for database operations
- **Database**: SQLite (configurable to PostgreSQL via environment variable)
- **File Processing**: CSV processing with pandas for data ingestion
- **Machine Learning**: Scikit-learn for anomaly detection and risk classification

### Key Design Patterns
- **Modular Components**: Separate classes for different responsibilities (DataProcessor, RuleEngine, MLEngine, SessionManager)
- **Session Management**: File-based session tracking with JSON persistence
- **Rule Engine**: Configurable rule system for automated email classification
- **ML Pipeline**: Integrated machine learning for anomaly detection and risk scoring

## Key Components

### 1. Data Processing Pipeline
- **CSV Ingestion**: Handles uploaded CSV files with dynamic column detection
- **Whitelist Filtering**: Removes emails from trusted domains before analysis
- **ML Analysis**: Applies anomaly detection and risk classification
- **Rule Application**: Processes emails through configurable business rules

### 2. Machine Learning Engine (`ml_engine.py`)
- **Anomaly Detection**: Uses Isolation Forest for outlier detection
- **Clustering**: DBSCAN for pattern discovery
- **Text Analysis**: TF-IDF vectorization for subject and attachment analysis
- **Risk Classification**: Multi-level risk scoring (Critical, High, Medium, Low)

### 3. Rule Engine (`rule_engine.py`)
- **Configurable Rules**: JSON-based rule definitions
- **Flexible Conditions**: Support for multiple operators (equals, contains, patterns)
- **Actions**: Automated responses (escalation, priority marking)
- **Priority System**: Rule execution based on priority levels

### 4. Session Management (`session_manager.py`)
- **Upload Sessions**: Tracks processing sessions with unique IDs
- **Data Persistence**: JSON file storage for sessions and whitelists
- **Status Tracking**: Monitors processing progress and completion

### 5. Web Interface
- **Dashboard**: Main overview of sessions and statistics
- **Session Details**: Detailed view of processed emails with filtering
- **Rule Management**: UI for creating and managing detection rules
- **Admin Panel**: Whitelist management and system configuration

## Data Flow

1. **CSV Upload**: User uploads Tessian email export CSV file
2. **Session Creation**: System generates unique session ID and stores metadata
3. **Data Validation**: Validates CSV structure and required columns
4. **Whitelist Filtering**: Removes emails from trusted domains
5. **ML Processing**: Applies anomaly detection and risk classification
6. **Rule Application**: Processes emails through business rules
7. **Data Storage**: Saves processed data to database and JSON files
8. **Dashboard Display**: Presents results in interactive web interface
9. **Case Management**: Analysts can clear or escalate individual cases
10. **Outlook Integration**: Generates draft emails for escalated cases

## External Dependencies

### Python Libraries
- **Flask**: Web framework and application structure
- **SQLAlchemy**: Database ORM and migrations
- **Pandas**: Data processing and CSV handling
- **Scikit-learn**: Machine learning algorithms
- **NumPy**: Numerical computing support

### Frontend Libraries
- **Bootstrap 5**: CSS framework for responsive design
- **Font Awesome**: Icon library for UI elements
- **Chart.js**: Data visualization and analytics charts
- **Google Fonts**: Typography (Source Sans Pro, Roboto)

### Database
- **SQLite**: Default database (development)
- **PostgreSQL**: Production database option via DATABASE_URL

## Deployment Strategy

### Development Setup
- **Local Development**: Flask development server with debug mode
- **File Storage**: Local file system for uploads and data persistence
- **Database**: SQLite for simplified development setup

### Production Considerations
- **Database Migration**: Environment-based database URL configuration
- **File Handling**: Configurable upload folder and file size limits
- **Security**: Session secrets and CSRF protection
- **Proxy Support**: ProxyFix middleware for reverse proxy deployments

### Environment Configuration
- **DATABASE_URL**: Database connection string
- **SESSION_SECRET**: Application secret key
- **Upload Settings**: Configurable file size limits and storage paths

The system is designed to be easily deployable on various platforms while maintaining flexibility for different deployment scenarios and scale requirements.