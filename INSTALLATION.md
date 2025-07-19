
# Email Guardian - Installation Guide

## For End Users Installing from GitHub

### First Time Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/email-guardian.git
   cd email-guardian
   ```

2. **Run the setup:**
   ```bash
   python setup.py
   ```

3. **Start the application:**
   ```bash
   python run.py
   ```

### Updating the Application (Preserving Your Data)

When updates are available, follow these steps to update without losing your uploaded files and processed data:

1. **Pull the latest changes:**
   ```bash
   git pull origin main
   ```
   
   Your `data/` and `uploads/` directories are protected by `.gitignore` and will not be overwritten.

2. **Check for new dependencies:**
   ```bash
   python setup.py
   ```

3. **Restart the application:**
   ```bash
   python run.py
   ```

### Data Backup (Recommended)

Before any major update, you can backup your data:

```bash
# Create backup directory
mkdir backup_$(date +%Y%m%d)

# Copy your data
cp -r data/ backup_$(date +%Y%m%d)/
cp -r uploads/ backup_$(date +%Y%m%d)/
```

### Troubleshooting

- If you encounter issues after an update, check the console output for any migration messages
- The application will automatically handle database schema updates when needed
- If problems persist, restore from your backup and contact support

### What Gets Updated vs. What's Preserved

**Updated on git pull:**
- Application code (Python files)
- Templates and static files
- Configuration files

**Preserved (never overwritten):**
- Your uploaded CSV files (`uploads/` directory)
- Processed session data (`data/` directory)
- Database files (`instance/` directory)
- Custom whitelists and rules you've configured
