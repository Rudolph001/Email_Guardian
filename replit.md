# Email Guardian - Tessian Email Exfiltration Detection System

## Overview

Email Guardian is a full-stack Flask application designed to ingest Tessian email CSV exports, detect potential data exfiltration, and manage security escalations. The system processes email data through machine learning algorithms, applies customizable rules, and provides a comprehensive dashboard for security analysts to review and manage email security incidents.

## User Preferences

Preferred communication style: Simple, everyday language.
Upload limit preference: High capacity (500MB) for large email data files.

## Recent Changes

- **Comprehensive Case Management Filtering System** (July 18, 2025):
  - Added advanced filtering on all available email fields
  - Primary filters: Risk Level, Rule Matches, Status, Quick Search
  - Advanced filters: Sender, Recipient, Domain, Subject, Attachment Type
  - ML Score range filtering (min/max values)
  - Date range filtering with calendar picker
  - Time range filtering for specific hours
  - Email size filtering (small/medium/large)
  - Link detection filtering (has links/no links)
  - Collapsible advanced filters section to keep UI clean
  - Filter persistence using localStorage
  - Auto-apply filters on dropdown changes
  - Clear all filters functionality
  - Save/load filter presets
  - Real-time filter count display
  - Enhanced search across all fields including attachment classifications
- **Enhanced Session Dashboard with Multiple Analytics Views** (July 18, 2025):
  - Added comprehensive analytics dashboard with 6 new chart types
  - Whitelisted Domains Analysis chart showing filtered vs analyzed email breakdown
  - Attachment Analysis chart categorizing attachment types (Business, Personal, Suspicious, etc.)
  - Time Distribution chart showing email patterns by hour of day
  - Top Senders & Recipients chart highlighting volume leaders
  - Whitelist Impact card showing filtering effectiveness with progress bars
  - Data Quality card displaying record completeness statistics
  - Processing Stats card showing session metadata and ML analysis status
  - Added dedicated analytics views: Whitelist Analysis, Time Analysis, Sender Analysis
  - Created dropdown navigation menu for easy access to specialized analytics views
  - Implemented smooth scrolling to chart sections
  - Added breadcrumb navigation for better user experience
- **Corporate Color Scheme Update** (July 18, 2025):
  - Updated application color palette to professional corporate theme
  - Changed from earthy colors to modern corporate blues, grays, and whites
  - Primary color: Dark gray (#1f2937) for professional and authoritative look
  - Secondary color: Corporate blue (#3b82f6) for trustworthy and modern feel
  - Background: Clean light gray (#f8fafc) for minimal professional appearance
  - Added Inter font family for modern, corporate typography
  - Updated all UI elements to match new corporate branding standards
  - Removed escalation counter badge from case management view for cleaner interface
- **Updated Rule Processing Workflow** (July 18, 2025):
  - Changed rule matches to go to case management dashboard with Critical risk level instead of directly to escalation
  - Only manually escalated cases (via escalate button) now appear in escalation dashboard
  - Rule matches are prioritized with Critical risk level to highlight them for review
  - Cleaned up dashboard logic to properly separate manual escalations from rule matches
- **Fixed Attachment Classification Bug** (July 18, 2025):
  - Resolved issue where ML attachment classification was returning `<class 'str'>` instead of actual classification
  - Fixed duplicate `analyze_emails` functions that were causing attachment_classifications to be missing
  - Added proper handling of "-" values as null/empty data throughout the system
  - Updated `has_attachments` logic to correctly identify when attachments field contains "-" vs actual file names
- **Fixed Step 2 Rule Escalation Logic** (July 18, 2025):
  - Resolved bug where rule matches weren't properly escalating to escalation dashboard
  - Fixed dashboard separation logic to use dashboard_type instead of status field
  - Updated escalation count display to show correct numbers across all dashboards
  - Any event matching a rule now automatically moves to escalation dashboard
- **4-Step Processing Workflow** (July 18, 2025): 
  - Implemented user-requested 4-step data processing workflow
  - Step 1: Whitelist domain filtering - ignore trusted events
  - Step 2: Rule matching - move rule matches to escalation dashboard
  - Step 3: ML analysis - run machine learning on remaining events
  - Step 4: Case management sorting - sort by ML score (high to low)
- **Dashboard Separation** (July 18, 2025): 
  - Created separate escalation and case management dashboards
  - Added visual progress indicators during import process
  - Updated upload modal to show processing workflow steps
- **Fixed ML Attachment Classification Bug** (July 18, 2025): Fixed critical bug where attachment classification was showing `<class 'str'>` instead of actual classification results due to duplicate function definitions in ML engine
- **Large File Processing Optimization** (July 18, 2025): Enhanced processing for 10,000+ record files
  - Reduced chunked processing threshold from 50MB to 10MB for better handling of large datasets
  - Increased chunk size from 1000 to 2500 records for improved performance
  - Added comprehensive error handling for session data saving failures
  - Implemented automatic compression for session data >5MB
  - Added processing status API endpoint for monitoring large file uploads
  - Fixed BAU analysis JavaScript errors when no session data is available
- **Migration to Replit Environment** (July 18, 2025): Successfully migrated project from Replit Agent to standard Replit environment with all functionality preserved
- **Upload Limit Increase** (July 18, 2025): Increased file upload limit from 16MB to 500MB to handle large email datasets
- **Security Configuration** (July 18, 2025): Implemented proper client/server separation with robust security practices

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