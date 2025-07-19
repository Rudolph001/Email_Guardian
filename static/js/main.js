// Email Guardian - Main JavaScript Functions
// Professional Business Intelligence Interface

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initializeApp();

    // Initialize tooltips
    initializeTooltips();

    // Initialize file upload handling
    initializeFileUpload();

    // Initialize chart animations
    initializeCharts();

    // Initialize form validations
    initializeValidations();

    // Initialize auto-refresh for dashboard
    initializeAutoRefresh();

    // Initialize attachment risk intelligence
    initializeAttachmentRiskIntelligence();
});

/**
 * Initialize the main application
 */
function initializeApp() {
    console.log('Email Guardian - Initializing application...');

    // Reprocess rules for existing session data
    async function reprocessRules(sessionId) {
        if (!confirm('This will reprocess all existing data against the current rules. Continue?')) {
            return;
        }

        const button = event.target.closest('button');
        const originalText = button.innerHTML;

        try {
            // Show loading state
            button.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Reprocessing...';
            button.disabled = true;

            const response = await fetch(`/api/reprocess_rules/${sessionId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const result = await response.json();

            if (result.success) {
                // Show success message
                showAlert('success', `Successfully reprocessed ${result.total_records} records. ${result.updated_records} records had rule changes.`);

                // Reload the page to show updated data
                setTimeout(() => {
                    window.location.reload();
                }, 2000);
            } else {
                showAlert('error', `Error: ${result.error}`);
            }

        } catch (error) {
            console.error('Reprocessing error:', error);
            showAlert('error', 'Failed to reprocess rules. Please try again.');
        } finally {
            // Restore button state
            button.innerHTML = originalText;
            button.disabled = false;
        }
    }

    // Helper function to show alerts
    function showAlert(type, message) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type === 'success' ? 'success' : 'danger'} alert-dismissible fade show`;
        alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

        const container = document.querySelector('.container-fluid') || document.body;
        container.insertBefore(alertDiv, container.firstChild);

        // Auto-remove alert after 5 seconds
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }

    // Set up global error handling
    window.addEventListener('error', function(e) {
        console.error('Application error:', e.error);
        showNotification('An unexpected error occurred. Please try again.', 'error');
    });

    // Set up CSRF token for AJAX requests
    const csrfToken = document.querySelector('meta[name="csrf-token"]');
    if (csrfToken) {
        window.csrfToken = csrfToken.getAttribute('content');
    }

    // Initialize page-specific functionality
    const currentPage = getCurrentPage();
    initializePageSpecific(currentPage);
}

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
}

/**
 * Initialize file upload handling
 */
function initializeFileUpload() {
    const fileInput = document.getElementById('file');
    const uploadForm = document.querySelector('form[enctype="multipart/form-data"]');

    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                validateFile(file);
            }
        });
    }

    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            const file = fileInput ? fileInput.files[0] : null;
            if (!file) {
                e.preventDefault();
                showNotification('Please select a file to upload.', 'error');
                return;
            }

            if (!validateFile(file)) {
                e.preventDefault();
                return;
            }

            showUploadProgress();
        });
    }
}

/**
 * Validate uploaded file
 */
function validateFile(file) {
    const maxSize = 16 * 1024 * 1024; // 16MB
    const allowedTypes = ['text/csv', 'application/vnd.ms-excel'];

    if (file.size > maxSize) {
        showNotification('File size exceeds 16MB limit.', 'error');
        return false;
    }

    if (!allowedTypes.includes(file.type) && !file.name.toLowerCase().endsWith('.csv')) {
        showNotification('Please upload a CSV file.', 'error');
        return false;
    }

    return true;
}

/**
 * Show upload progress
 */
function showUploadProgress() {
    const uploadButton = document.querySelector('button[type="submit"]');
    if (uploadButton) {
        const originalText = uploadButton.innerHTML;
        uploadButton.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
        uploadButton.disabled = true;

        // Re-enable after 30 seconds as fallback
        setTimeout(() => {
            uploadButton.innerHTML = originalText;
            uploadButton.disabled = false;
        }, 30000);
    }
}

/**
 * Initialize chart animations
 */
function initializeCharts() {
    // Set default Chart.js options
    if (typeof Chart !== 'undefined') {
        Chart.defaults.font.family = 'Source Sans Pro, Roboto, sans-serif';
        Chart.defaults.font.size = 12;
        Chart.defaults.color = '#262730';
        Chart.defaults.backgroundColor = '#ffffff';
        Chart.defaults.borderColor = '#e1e5e9';

        // Add animation defaults
        Chart.defaults.animation.duration = 1000;
        Chart.defaults.animation.easing = 'easeOutCubic';

        // Add responsive defaults
        Chart.defaults.responsive = true;
        Chart.defaults.maintainAspectRatio = false;
    }
}

/**
 * Initialize form validations
 */
function initializeValidations() {
    const forms = document.querySelectorAll('form[data-validate]');

    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            if (!validateForm(form)) {
                e.preventDefault();
                e.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });
}

/**
 * Validate form
 */
function validateForm(form) {
    const inputs = form.querySelectorAll('input[required], select[required], textarea[required]');
    let isValid = true;

    inputs.forEach(input => {
        if (!input.value.trim()) {
            isValid = false;
            input.classList.add('is-invalid');
        } else {
            input.classList.remove('is-invalid');
        }
    });

    return isValid;
}

/**
 * Initialize auto-refresh for dashboard
 */
function initializeAutoRefresh() {
    const dashboardPage = document.querySelector('.dashboard-page');
    if (dashboardPage) {
        // Auto-refresh every 5 minutes
        setInterval(() => {
            refreshDashboardStats();
        }, 300000);
    }
}

/**
 * Refresh dashboard statistics
 */
function refreshDashboardStats() {
    const sessionId = getSessionIdFromURL();
    if (sessionId) {
        fetch(`/api/session_stats/${sessionId}`)
            .then(response => response.json())
            .then(data => {
                updateDashboardStats(data);
            })
            .catch(error => {
                console.error('Error refreshing dashboard stats:', error);
            });
    }
}

/**
 * Update dashboard statistics
 */
function updateDashboardStats(data) {
    const statsCards = document.querySelectorAll('.stats-card h5');
    if (statsCards.length >= 4) {
        statsCards[0].textContent = data.total_records || 0;
        statsCards[1].textContent = data.cases_cleared || 0;
        statsCards[2].textContent = data.cases_escalated || 0;
        statsCards[3].textContent = data.cases_open || 0;
    }
}

/**
 * Get current page identifier
 */
function getCurrentPage() {
    const path = window.location.pathname;
    if (path === '/') return 'dashboard';
    if (path.includes('/dashboard/')) return 'session';
    if (path.includes('/admin')) return 'admin';
    if (path.includes('/rules')) return 'rules';
    return 'unknown';
}

/**
 * Initialize page-specific functionality
 */
function initializePageSpecific(page) {
    switch (page) {
        case 'dashboard':
            initializeDashboard();
            break;
        case 'session':
            initializeSession();
            break;
        case 'admin':
            initializeAdmin();
            break;
        case 'rules':
            initializeRules();
            break;
    }
}

/**
 * Initialize dashboard page
 */
function initializeDashboard() {
    console.log('Initializing dashboard page...');

    // Initialize session cards hover effects
    const sessionCards = document.querySelectorAll('.card');
    sessionCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
        });

        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });

    // Initialize upload modal
    const uploadModal = document.getElementById('uploadModal');
    if (uploadModal) {
        uploadModal.addEventListener('shown.bs.modal', function() {
            const fileInput = document.getElementById('file');
            if (fileInput) {
                fileInput.focus();
            }
        });
    }
}

/**
 * Initialize session page
 */
function initializeSession() {
    console.log('Initializing session page...');

    // Initialize case action buttons
    const actionButtons = document.querySelectorAll('button[data-action]');
    actionButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            const action = this.dataset.action;
            const recordId = this.dataset.recordId;

            if (action === 'escalate') {
                if (!confirm('Are you sure you want to escalate this case?')) {
                    e.preventDefault();
                    return;
                }
            }

            // Show loading state
            this.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            this.disabled = true;
        });
    });

    // Initialize table sorting
    initializeTableSorting();

    // Initialize export functionality
    initializeExport();
}

/**
 * Initialize admin page
 */
function initializeAdmin() {
    console.log('Initializing admin page...');

    // Initialize whitelist textarea
    const domainsTextarea = document.getElementById('domains');
    if (domainsTextarea) {
        domainsTextarea.addEventListener('input', function() {
            const domains = this.value.split('\n').filter(d => d.trim());
            const count = domains.length;

            const countDisplay = document.getElementById('domain-count');
            if (countDisplay) {
                countDisplay.textContent = count;
            }
        });
    }

    // Initialize session management
    initializeSessionManagement();
}

/**
 * Initialize rules page
 */
function initializeRules() {
    console.log('Initializing rules page...');

    // Initialize rule builder
    initializeRuleBuilder();

    // Initialize rule actions
    initializeRuleActions();
}

/**
 * Initialize rule builder
 */
function initializeRuleBuilder() {
    const createRuleModal = document.getElementById('createRuleModal');
    if (createRuleModal) {
        createRuleModal.addEventListener('show.bs.modal', function() {
            resetRuleForm();
        });
    }
}

/**
 * Reset rule form
 */
function resetRuleForm() {
    const form = document.querySelector('#createRuleModal form');
    if (form) {
        form.reset();

        // Reset dynamic fields
        const conditionsContainer = document.getElementById('conditions-container');
        const actionsContainer = document.getElementById('actions-container');

        if (conditionsContainer) {
            conditionsContainer.innerHTML = conditionsContainer.children[0].outerHTML;
        }

        if (actionsContainer) {
            actionsContainer.innerHTML = actionsContainer.children[0].outerHTML;
        }
    }
}

/**
 * Initialize rule actions
 */
function initializeRuleActions() {
    const ruleTable = document.querySelector('.table');
    if (ruleTable) {
        ruleTable.addEventListener('click', function(e) {
            if (e.target.matches('.btn-outline-danger')) {
                const ruleId = e.target.dataset.ruleId;
                if (ruleId && !confirm('Are you sure you want to delete this rule?')) {
                    e.preventDefault();
                }
            }
        });
    }
}

/**
 * Initialize table sorting
 */
function initializeTableSorting() {
    const sortableHeaders = document.querySelectorAll('th[data-sort]');

    sortableHeaders.forEach(header => {
        header.style.cursor = 'pointer';
        header.addEventListener('click', function() {
            const sortBy = this.dataset.sort;
            const table = this.closest('table');
            sortTable(table, sortBy);
        });
    });
}

/**
 * Sort table by column
 */
function sortTable(table, sortBy) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));

    const sortedRows = rows.sort((a, b) => {
        const aValue = a.querySelector(`[data-value="${sortBy}"]`)?.textContent || '';
        const bValue = b.querySelector(`[data-value="${sortBy}"]`)?.textContent || '';

        return aValue.localeCompare(bValue);
    });

    tbody.innerHTML = '';
    sortedRows.forEach(row => tbody.appendChild(row));
}

/**
 * Initialize export functionality
 */
function initializeExport() {
    const exportButton = document.querySelector('a[href*="/export/"]');
    if (exportButton) {
        exportButton.addEventListener('click', function() {
            showNotification('Export started. Download will begin shortly.', 'info');
        });
    }
}

/**
 * Initialize session management
 */
function initializeSessionManagement() {
    const sessionTable = document.querySelector('.table');
    if (sessionTable) {
        sessionTable.addEventListener('click', function(e) {
            if (e.target.matches('.btn-outline-danger')) {
                const sessionId = e.target.dataset.sessionId;
                if (sessionId && !confirm('Are you sure you want to delete this session?')) {
                    e.preventDefault();
                }
            }
        });
    }
}

/**
 * Show notification
 */
function showNotification(message, type = 'info') {
    const alertClass = type === 'error' ? 'alert-danger' : `alert-${type}`;
    const alert = document.createElement('div');
    alert.className = `alert ${alertClass} alert-dismissible fade show`;
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    const container = document.querySelector('.container-fluid');
    if (container) {
        container.insertBefore(alert, container.firstChild);

        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            alert.remove();
        }, 5000);
    }
}

/**
 * Get session ID from URL
 */
function getSessionIdFromURL() {
    const path = window.location.pathname;
    const matches = path.match(/\/dashboard\/([^\/]+)/);
    return matches ? matches[1] : null;
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Format date
 */
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

/**
 * Debounce function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Throttle function
 */
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

/**
 * API utility functions
 */
const API = {
    /**
     * Make API request
     */
    request: function(url, options = {}) {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            }
        };

        if (window.csrfToken) {
            defaultOptions.headers['X-CSRFToken'] = window.csrfToken;
        }

        const mergedOptions = { ...defaultOptions, ...options };

        return fetch(url, mergedOptions)
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .catch(error => {
                console.error('API request failed:', error);
                throw error;
            });
    },

    /**
     * Get session statistics
     */
    getSessionStats: function(sessionId) {
        return this.request(`/api/session_stats/${sessionId}`);
    },

    /**
     * Get ML insights
     */
    getMLInsights: function(sessionId) {
        return this.request(`/api/ml_insights/${sessionId}`);
    }
};

/**
 * Export global utilities
 */
window.EmailGuardian = {
    API,
    showNotification,
    formatFileSize,
    formatDate,
    debounce,
    throttle
};

// Console banner
console.log(`
%c   _____ __  __          _____ _       _____                     _ _             
%c  |  ___|  \\/  |   /\\   |_   _| |     |  __ \\                   | (_)            
%c  | |___| |\\/| |  /  \\    | | | |     | |  \\/ _   _  __ _ _ __ __| |_  __ _ _ __  
%c  |  ___|  |  | | /    \\   | | | |     | | __ | | | |/ _' | '__/ _' | |/ _' | '_ \\ 
%c  | |___| |  | |/  /\\  \\  | | | |____ | |_\\ \\| |_| | (_| | | | (_| | | (_| | | | |
%c  \\____/|_|  |_/_ /    \\_\\ |_| |______| \\____/ \\__,_|\\__,_|_|  \\__,_|_|\\__,_|_| |_|
%c                                                                                   
%c  Email Guardian - Advanced Email Security Analysis
%c  Professional Business Intelligence Interface
`, 
'color: #1f77b4', 'color: #1f77b4', 'color: #1f77b4', 'color: #1f77b4', 
'color: #1f77b4', 'color: #1f77b4', 'color: #1f77b4', 'color: #ff7f0e', 'color: #6c757d');

/**
 * Initialize attachment risk intelligence
 */
function initializeAttachmentRiskIntelligence() {
    console.log('Initializing attachment risk intelligence...');
    const sessionId = getSessionIdFromURL();
    if (!sessionId) {
        console.warn('Session ID not found in URL.');
        return;
    }

    // Check if the attachment risk elements exist
    const criticalElement = document.getElementById('criticalAttachments');
    const highRiskElement = document.getElementById('highRiskAttachments');
    const avgScoreElement = document.getElementById('avgRiskScore');
    const topRiskFactorsElement = document.getElementById('topRiskFactors');

    if (!criticalElement || !highRiskElement || !avgScoreElement || !topRiskFactorsElement) {
        console.log('Attachment risk intelligence elements not found on this page');
        return;
    }

    // Load attachment risk analytics
    fetch(`/api/attachment_risk_analytics/${sessionId}`)
        .then(response => response.json())
        .then(data => {
            console.log('Attachment risk analytics data:', data);

            if (data.error) {
                topRiskFactorsElement.innerHTML = `<li><small class="text-danger">Error: ${data.error}</small></li>`;
                return;
            }

            updateAttachmentRiskDisplay(data);
        })
        .catch(error => {
            console.error('Error loading attachment risk analytics:', error);
            topRiskFactorsElement.innerHTML = `<li><small class="text-danger">Failed to load risk data</small></li>`;
        });
}

function updateAttachmentRiskDisplay(data) {
    // Update the individual elements
    const criticalElement = document.getElementById('criticalAttachments');
    const highRiskElement = document.getElementById('highRiskAttachments');
    const avgScoreElement = document.getElementById('avgRiskScore');
    const topRiskFactorsElement = document.getElementById('topRiskFactors');

    // Risk overview metrics
    const criticalRisk = data.critical_risk_count || 0;
    const highRisk = data.high_risk_count || 0;
    const avgRiskScore = data.average_risk_score || 0;

    console.log('Updating attachment risk display with:', { criticalRisk, highRisk, avgRiskScore });

    // Update the metric displays
    criticalElement.textContent = criticalRisk;
    highRiskElement.textContent = highRisk;
    avgScoreElement.textContent = avgRiskScore.toFixed(1);

    // Update top risk factors
    if (Object.keys(data.top_risk_factors || {}).length > 0) {
        const factorsList = Object.entries(data.top_risk_factors)
            .slice(0, 5)
            .map(([factor, count]) => 
                `<div class="d-flex justify-content-between align-items-start mb-2">
                    <div class="flex-grow-1 me-2">
                        <small class="text-dark" title="${factor}">${factor}</small>
                    </div>
                    <span class="badge bg-secondary">${count}</span>
                </div>`)
            .join('');
        
        topRiskFactorsElement.innerHTML = factorsList;
    } else {
        topRiskFactorsElement.innerHTML = '<div class="text-muted small">No risk factors found</div>';
    }
}