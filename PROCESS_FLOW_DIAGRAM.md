
# Email Guardian - Process Flow Diagram

## System Overview

This document describes the complete process flow for Email Guardian's email security analysis system. The diagram shows all backend operations, data processing steps, and performance points to help identify optimization opportunities and understand system behavior.

## Complete Process Flow

```
┌─────────────────┐
│   User Action   │
│  (File Upload)  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ File Upload     │
│ Validation      │
│ (CSV Format)    │
└─────────┬───────┘
          │
          ▼
     ┌─────────┐
     │ Valid?  │◄─── Error: Return to upload
     └────┬────┘
          │ ✓
          ▼
┌─────────────────┐
│ Session         │
│ Creation        │
│ (Generate UUID) │
└─────────┬───────┘
          │
          ▼
┌─────────────────────────────────────────┐
│        4-Step Processing Pipeline       │
├─────────────────────────────────────────┤
│                                         │
│ ┌─────────────────┐                    │
│ │ Step 0:         │                    │
│ │ Exclusion Rules │                    │
│ │ Filter unwanted │                    │
│ │ records         │                    │
│ └─────────┬───────┘                    │
│           │                            │
│           ▼                            │
│ ┌─────────────────┐                    │
│ │ Step 1:         │                    │
│ │ Whitelist       │                    │
│ │ Domain          │                    │
│ │ Filtering       │                    │
│ └─────────┬───────┘                    │
│           │                            │
│           ▼                            │
│ ┌─────────────────┐                    │
│ │ Step 2:         │                    │
│ │ Rule Engine     │                    │
│ │ Business Logic  │                    │
│ │ Matching        │                    │
│ └─────────┬───────┘                    │
│           │                            │
│           ▼                            │
│ ┌─────────────────┐                    │
│ │ Step 3:         │                    │
│ │ ML Analysis     │                    │
│ │ Anomaly         │                    │
│ │ Detection       │                    │
│ └─────────┬───────┘                    │
│           │                            │
└───────────┼────────────────────────────┘
            │
            ▼
┌─────────────────┐      ┌─────────────────┐
│ Data Storage    │      │ Dashboard       │
│ JSON +          │◄────►│ Risk            │
│ Compression     │      │ Visualization   │
│ Database        │      └─────────────────┘
└─────────────────┘              │
                                 ▼
                         ┌─────────────────┐
                         │ Case Management │
                         │ Manual Review   │
                         │ & Actions       │
                         └─────────────────┘
```

## Detailed Process Breakdown

### 1. File Upload Operations

**Components:**
- File validation (size, type, format)
- CSV header detection & validation
- Session UUID generation
- Memory allocation for processing

**Performance Notes:**
- ⚠️ **Bottleneck**: Large file processing (>10MB)
- ✅ **Optimization**: File size validation before processing

### 2. Four-Step Processing Pipeline

#### Step 0: Exclusion Rules
- **Purpose**: Filter unwanted records before processing
- **Operation**: Remove records that don't meet basic criteria
- **Performance**: Fast filtering, minimal overhead

#### Step 1: Whitelist Filtering
- **Purpose**: Remove emails from trusted domains
- **Operation**: Domain-based exclusion using whitelist
- **Configuration**: Admin-configurable trusted domains

#### Step 2: Rule Engine
- **Purpose**: Apply business logic and classification rules
- **Operation**: Pattern matching, keyword detection, business rules
- **Features**: Configurable rules, multiple condition types

#### Step 3: ML Analysis
- **Purpose**: AI-powered anomaly detection and risk scoring
- **Operation**: Machine learning classification and risk assessment
- **Performance**: ⚠️ **Bottleneck**: Memory usage during ML analysis

### 3. Data Storage Operations

**Process:**
1. JSON serialization
2. Automatic compression (files >5MB)
3. Database record creation
4. Session metadata storage

**Optimizations:**
- ✅ Chunked processing (2500 records/chunk)
- ✅ Automatic compression for large datasets
- ✅ Efficient JSON storage format

### 4. Dashboard Operations

**Features:**
- Real-time risk visualization
- Interactive case management
- Escalation handling
- Data export capabilities

**Performance:**
- ✅ Paginated data loading (50 records/page)
- ✅ Server-side filtering
- ⚠️ **Bottleneck**: Full dataset loading for complex queries

## Performance Metrics & Bottlenecks

### Current Bottlenecks

| Component | Issue | Impact |
|-----------|-------|--------|
| File Upload | Large files >10MB | Slow processing, timeout risk |
| ML Analysis | Memory usage | Performance degradation |
| Dashboard | Full dataset loading | Slow page loads |

### Implemented Optimizations

| Component | Optimization | Benefit |
|-----------|-------------|---------|
| Processing | Chunked processing (2500 records) | Reduced memory usage |
| Storage | Automatic compression >5MB | Reduced storage space |
| Dashboard | Paginated loading (50 records) | Faster page loads |
| Queries | Server-side filtering | Reduced data transfer |

## Admin Operations

### Configuration Management
- **Rules CRUD**: Create, update, delete business rules
- **Domain Management**: Classify domains by category
- **Whitelist Management**: Configure trusted domains
- **Session Management**: Monitor and manage processing sessions

### Administrative Workflows
```
Admin Action → Configuration Update → Automatic Reprocessing → Updated Results
```

## Frontend Interactions

### User Workflows
1. **Dashboard Navigation** → Triggers data queries
2. **Filter Changes** → Reloads processed data
3. **Case Actions** → Updates database & session
4. **Admin Changes** → Triggers system reprocessing

### System Responses
- Real-time data updates
- Progress indicators during processing
- Error handling and user feedback
- Automatic session management

## System Components Status

| Component | Status | Description |
|-----------|--------|-------------|
| 🤖 ML Engine | ✅ Active | Anomaly detection and risk scoring |
| ⚙️ Rule Engine | ✅ Active | Business logic processing |
| 🗄️ Data Storage | ✅ Available | JSON + database persistence |
| 🛡️ Security Engine | ✅ Active | Whitelist and domain filtering |

## Processing Statistics

- **File Processing**: CSV ingestion with dynamic column detection
- **Memory Management**: Efficient chunked processing
- **Storage Efficiency**: Automatic compression for large datasets
- **Query Performance**: Optimized database operations

## Optimization Recommendations

### Short-term Improvements
1. Implement streaming file upload for large files
2. Add progress indicators for long-running operations
3. Optimize ML model loading and caching

### Long-term Enhancements
1. Implement distributed processing for very large datasets
2. Add caching layer for frequently accessed data
3. Consider database optimization for complex queries

## Architecture Benefits

- **Modular Design**: Each component can be optimized independently
- **Scalable Processing**: Chunked processing handles large datasets
- **Configurable Rules**: Business logic can be adjusted without code changes
- **Performance Monitoring**: Built-in bottleneck identification
- **User Experience**: Real-time feedback and progress tracking

---

*This diagram represents the current system architecture as of July 2025. Performance metrics and optimizations are continuously monitored and improved.*
