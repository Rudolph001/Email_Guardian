
# Email Guardian - Complete Build Prompt for Replit AI

## Project Overview
Build a comprehensive **Email Security Analysis Platform** called "Email Guardian" - a full-stack web application for analyzing Tessian email export data to detect security threats, data exfiltration attempts, and policy violations.

## Core Architecture Requirements

### Technology Stack
- **Backend**: Python Flask web framework
- **Frontend**: Bootstrap 5 + vanilla JavaScript with Chart.js
- **Database**: SQLite with SQLAlchemy ORM
- **ML/Analytics**: pandas, numpy, scikit-learn, NetworkX
- **File Processing**: CSV parsing with chunked processing for large files
- **Deployment**: Gunicorn on port 5000 (0.0.0.0:5000)

### File Structure
```
email-guardian/
├── main.py                    # Flask app entry point
├── app.py                     # Flask app factory and config
├── routes.py                  # All web routes and API endpoints
├── models.py                  # SQLAlchemy database models
├── session_manager.py         # Data persistence and session management
├── data_processor.py          # CSV processing and workflow engine
├── ml_engine.py              # Machine learning analysis engine
├── advanced_ml_engine.py     # Advanced ML analytics
├── rule_engine.py            # Business rules and exclusion engine
├── domain_manager.py         # Domain classification system
├── static/css/style.css      # Custom styling
├── static/js/main.js         # Frontend JavaScript
├── templates/                # Jinja2 HTML templates
├── data/                     # JSON data storage
├── uploads/                  # CSV file uploads
└── instance/                 # SQLite database
```

## Functional Requirements

### 1. CSV Data Processing Engine
- **Input**: Tessian email export CSV files with columns:
  - `_time`, `sender`, `subject`, `attachments`, `recipients`, `recipients_email_domain`
  - `leaver`, `termination_date`, `wordlist_attachment`, `wordlist_subject`
  - `bunit`, `department`, `status`, `user_response`, `final_outcome`, `justification`
- **Features**:
  - Case-insensitive field matching
  - Large file chunked processing (2500 records/chunk)
  - Data validation and error reporting
  - Automatic lowercase conversion for consistency

### 2. Policy Build & Exclusion Rules Engine
- **Pre-Upload Filtering**: Comprehensive exclusion rules applied before processing
- **Multi-Condition Logic**: Complex AND/OR combinations with field-level operators
- **Supported Operators**: 
  - `equals`, `contains`, `not_equals`, `starts_with`, `ends_with`
  - `in_list`, `greater_than`, `less_than`, `matches_pattern` (regex)
- **Advanced Features**:
  - Case-sensitive/insensitive matching per rule
  - Value parsing for quoted phrases and comma-separated lists
  - Dynamic field mapping for any CSV structure
  - Rule testing and impact preview
- **Management Interface**: Visual rule builder with drag-and-drop conditions

### 3. 4-Step Processing Workflow
1. **Exclusion Rules**: Filter records based on configurable exclusion criteria
2. **Whitelist Filtering**: Remove trusted domain communications
3. **Rule Engine**: Apply security rules, mark rule matches as Critical risk
4. **ML Analysis**: Score remaining records with anomaly detection

### 3. Machine Learning Engine
- **Anomaly Detection**: Isolation Forest for unusual patterns
- **Risk Classification**: Critical/High/Medium/Low based on multiple factors
- **Clustering**: DBSCAN for behavioral pattern grouping
- **Advanced Features**:
  - Attachment risk scoring with malware/exfiltration detection
  - BAU (Business As Usual) communication pattern analysis
  - Temporal anomaly detection
  - Network analysis of email flows

### 4. Dashboard System (3 Main Views)

#### A. Main Dashboard (`/dashboard/<session_id>`)
- Processing statistics and ML insights
- Risk distribution charts
- BAU analysis with whitelisting recommendations
- Attachment risk analytics with detailed breakdowns

#### B. Case Management (`/cases/<session_id>`)
- Paginated table of all processed emails
- Advanced filtering (risk level, rule matches, status)
- Detailed case popup with ML explanations
- Case status management (Active/Cleared/Escalated)

#### C. Escalation Dashboard (`/escalations/<session_id>`)
- Critical cases requiring immediate attention
- Draft email generation for escalations
- Resolution tracking and case closure
- Priority-based sorting

### 5. Advanced Analytics Features
- **BAU Analysis**: Identify routine business communications for whitelisting
- **Attachment Intelligence**: Comprehensive malware and exfiltration risk scoring
- **Temporal Patterns**: Detect off-hours and unusual timing activities
- **Domain Classification**: Categorize domains (Corporate/Personal/Public/Suspicious)
- **Communication Networks**: Graph analysis of sender-recipient relationships

### 6. Administration Panel (`/admin`)
- **Whitelist Management**: Domain whitelist with auto-reprocessing
- **Domain Classifications**: Trusted/Corporate/Personal/Public/Suspicious categories
- **Attachment Keywords**: Business/Personal/Suspicious keyword lists
- **Session Management**: View, export, delete processing sessions
- **Rules Engine**: Create/edit/delete security rules with conditions/actions

### 7. Rules Engine (`/rules`)
- **Security Rules**: Complex condition-based rules with AND/OR logic
- **Exclusion Rules**: Pre-filter records before processing
- **Auto-Reprocessing**: Automatically reprocess all sessions when rules change
- **Supported Conditions**: Contains, equals, starts_with, ends_with, regex
- **Actions**: Escalate, flag, score_modifier, tag

## Technical Implementation Details

### Data Flow Architecture
1. **Upload** → CSV validation and session creation
2. **Processing** → 4-step workflow with ML analysis
3. **Storage** → JSON-based session data with compression
4. **Dashboard** → Real-time analytics and case management
5. **Export** → JSON export with all processed data

### Performance Optimizations
- Chunked processing for large files (10MB+ threshold)
- Lazy loading with pagination (50 records/page)
- Compressed data storage for large sessions
- Efficient filtering at data layer
- Background processing indicators

### Security Features
- Input validation and sanitization
- Safe file upload handling
- XSS protection in templates
- CSRF protection on forms
- Secure session management

### API Endpoints (Key Routes)
```python
# Core Application
GET  /                                    # Main index
POST /upload                             # CSV file upload
GET  /dashboard/<session_id>             # Main dashboard
GET  /cases/<session_id>                 # Case management
GET  /escalations/<session_id>           # Escalation dashboard

# Analytics APIs
GET  /api/ml_insights/<session_id>       # ML analysis data
GET  /api/bau_analysis/<session_id>      # BAU recommendations
GET  /api/attachment_risk_analytics/<session_id>  # Attachment intelligence
GET  /api/case/<session_id>/<record_id>  # Individual case details

# Exclusion Rules & Policy Management
GET  /api/exclusion-rules               # Get all exclusion rules
POST /api/exclusion-rules               # Create new exclusion rule
GET  /api/exclusion-rules/<rule_id>     # Get specific exclusion rule
PUT  /api/exclusion-rules/<rule_id>     # Update exclusion rule
DELETE /api/exclusion-rules/<rule_id>   # Delete exclusion rule
POST /api/exclusion-rules/<rule_id>/toggle # Toggle rule active status

# Administration
GET  /admin                              # Admin panel
GET  /rules                              # Rules management
POST /admin/whitelist                    # Update whitelist
POST /rules/create                       # Create new rule
```

### Database Models
```python
# SQLAlchemy models needed:
class EmailRecord(db.Model)      # Individual email records
class ProcessingSession(db.Model) # Session metadata
class Rule(db.Model)             # Security rules configuration
```

### Frontend Components
- **Chart.js Integration**: Risk distribution, trends, BAU analysis charts
- **Bootstrap Modals**: Case details, escalation forms, rule creation, exclusion rule builder
- **DataTables**: Advanced sorting/filtering for case management
- **Progress Indicators**: File upload and processing status
- **Policy Builder Interface**: Visual exclusion rule creation with dynamic condition management
- **Real-time Validation**: Field existence checking and rule testing
- **Responsive Design**: Mobile-friendly interface

### Key JavaScript Functions
```javascript
// Core functionality needed:
- loadMLInsights(sessionId)
- showCaseDetails(sessionId, recordId)
- escalateCase(sessionId, recordId)  
- filterCases(filters)
- loadBAUAnalysis(sessionId)
- updateCaseStatus(sessionId, recordId, status)
```

## Advanced Features to Implement

### 1. Business As Usual (BAU) Analysis
- Identify high-volume sender-recipient domain pairs
- Calculate communication frequency and patterns
- Generate whitelist recommendations
- One-click whitelist addition from BAU results

### 2. Attachment Risk Intelligence
- Multi-layer risk scoring (0-100 scale)
- Malicious indicator detection (double extensions, executables)
- Data exfiltration pattern recognition
- Risk factor categorization and explanation

### 3. Advanced ML Analytics
- Justification sentiment analysis
- Recipient domain behavior profiling
- Communication pattern anomalies
- Temporal analysis with business hours detection
- Network graph analysis of email flows

### 4. Enhanced Case Management
- Bulk case operations
- Advanced search with regex support
- Case assignment and tracking
- Escalation workflow automation
- Email draft generation for escalations

## Styling and UX Requirements
- **Professional Business Theme**: Clean, corporate design
- **Color Scheme**: Bootstrap primary colors with risk-based color coding
- **Icons**: Font Awesome 5 icons throughout
- **Responsive**: Mobile-first design principles  
- **Loading States**: Spinners and progress indicators
- **Error Handling**: User-friendly error messages and validation

## Data Storage Strategy
- **Sessions**: JSON files with metadata in `data/sessions/`
- **Processed Data**: Compressed JSON for large datasets
- **Configuration**: JSON files for whitelists, rules, domain classifications
- **Uploads**: Original CSV files preserved in `uploads/`
- **Database**: SQLite for relational data (rules, sessions metadata)

## Performance Targets
- Handle CSV files up to 50MB (25,000+ records)
- Page load times under 2 seconds
- Processing time under 30 seconds for 10,000 records
- Memory usage under 1GB during processing
- Responsive UI with smooth interactions

## Error Handling Requirements
- Comprehensive CSV validation with detailed error messages
- Graceful handling of malformed data
- Processing error recovery and logging
- User-friendly error displays with technical details
- Session recovery capabilities

Build this as a complete, production-ready application with all features fully implemented. Focus on code quality, performance, and user experience. The application should be immediately usable for email security analysis upon completion.
