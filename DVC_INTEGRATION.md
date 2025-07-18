# Data Versioning with DVC Integration

This document explains how DVC (Data Version Control) is integrated into the retraining pipeline for automated data versioning.

## Overview

The system now automatically handles data versioning during the retraining process using DVC. Here's how it works:

### Data Flow and Versioning Strategy

1. **Main Data File (`data.csv`)**:
   - This is the primary training dataset
   - Tracked by DVC for version control
   - Grows over time as new data is appended during retraining

2. **New Data File (`new_data.csv`)**:
   - Temporary storage for newly generated/collected data
   - NOT tracked by DVC (it's transient storage)
   - Contains data waiting to be processed during retraining
   - Only the latest 200 rows are retained after each retraining

### Retraining Pipeline with DVC Integration

When retraining is triggered, the following steps occur:

#### Step 1: Data Management and Versioning
1. **Ensure DVC Tracking**: Verify that `data.csv` is tracked by DVC
2. **Backup New Data**: Create a timestamped backup of `new_data.csv`
3. **Data Movement**: 
   - Move all but the last 200 rows from `new_data.csv` to `data.csv`
   - Keep the latest 200 rows in `new_data.csv` for monitoring
4. **Create DVC Snapshot**: Create a versioned snapshot of the updated `data.csv`

#### Step 2: Model Training
- Trigger the MLflow training pipeline with the updated data

#### Step 3: Model Loading
- Load the newly trained model if training was successful

## DVC Manager Features

The `DVCManager` class provides the following functionality:

### Core Methods

- `ensure_data_tracked()`: Ensures `data.csv` is tracked by DVC
- `create_data_snapshot(message)`: Creates a versioned snapshot with optional commit message
- `backup_new_data()`: Creates timestamped backups of `new_data.csv`
- `get_data_info()`: Returns status information about data files

### Dashboard Integration

The dashboard now displays:
- DVC tracking status for main data
- File sizes for both data files
- Data file existence status

## Benefits

1. **Data Provenance**: Every change to the training data is tracked and versioned
2. **Rollback Capability**: Can revert to any previous version of the data
3. **Audit Trail**: Complete history of data changes with timestamps and messages
4. **Storage Efficiency**: DVC deduplicates data and stores only changes
5. **Collaboration**: Multiple team members can work with the same data versions

## Usage Examples

### Manual DVC Operations

```bash
# Check current DVC status
dvc status

# View data file information
dvc data status

# Show data version history
git log --oneline Data/data.csv.dvc

# Checkout a specific version of data
git checkout <commit-hash> Data/data.csv.dvc
dvc checkout Data/data.csv.dvc
```

### Programmatic Usage

The DVC integration is automatic during retraining, but you can also use it programmatically:

```python
from app.services.dvc_manager import dvc_manager

# Get data information
info = await dvc_manager.get_data_info()

# Create a manual snapshot
success = await dvc_manager.create_data_snapshot("Manual checkpoint")

# Ensure tracking
tracked = await dvc_manager.ensure_data_tracked()
```

## File Structure

```
Data/
├── data.csv          # Main training data (DVC tracked)
├── data.csv.dvc      # DVC metadata file (Git tracked)
├── new_data.csv      # Temporary new data (not DVC tracked)
├── .gitignore        # Ignores data.csv (managed by DVC)
└── new_data_backup_* # Timestamped backups
```

## Configuration

### DVC Configuration
- Repository root: `/app` (inside container)
- Data directory: `Data/`
- Storage: Local cache (can be extended to remote storage)

### Environment Variables
No additional environment variables are required for basic DVC functionality.

## Monitoring

### Dashboard Indicators
- ✅ Green checkmarks: Files exist and are properly tracked
- ❌ Red crosses: Files missing or not tracked
- File sizes: Displayed in human-readable format

### Logging
The system logs all DVC operations including:
- Data tracking status
- Snapshot creation
- Backup operations
- Error conditions

## Troubleshooting

### Common Issues

1. **DVC not tracking data.csv**:
   ```bash
   dvc add Data/data.csv
   git add Data/data.csv.dvc Data/.gitignore
   git commit -m "Add data to DVC tracking"
   ```

2. **Git conflicts with DVC files**:
   ```bash
   git checkout Data/data.csv.dvc
   dvc checkout
   ```

3. **Large file performance**:
   - DVC handles large files efficiently
   - Consider setting up remote storage for production

### Best Practices

1. **Regular Snapshots**: The system automatically creates snapshots during retraining
2. **Meaningful Messages**: Automatic commit messages include row counts and timestamps
3. **Backup Strategy**: New data is backed up before processing
4. **Monitoring**: Use the dashboard to monitor DVC status

## Future Enhancements

Potential improvements to consider:

1. **Remote Storage**: Configure DVC with cloud storage (S3, GCS, Azure)
2. **Data Pipelines**: Use DVC pipelines for more complex data processing
3. **Metrics Tracking**: Track data quality metrics alongside versions
4. **Automated Cleanup**: Remove old backups after a certain period
5. **Data Validation**: Validate data integrity during versioning

## Security Considerations

- DVC metadata files are tracked in Git (small files)
- Actual data files are stored in DVC cache (local or remote)
- Sensitive data never enters Git repository
- Access control managed through Git permissions
