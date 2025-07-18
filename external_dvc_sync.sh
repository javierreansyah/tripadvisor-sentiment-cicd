#!/bin/bash
# External DVC operations script
# Run this script on the host machine after retraining to version the data

echo "Running external DVC operations..."

# Change to the repository directory
cd "$(dirname "$0")"

# Check if we're in a DVC repository
if [ ! -d ".dvc" ]; then
    echo "Error: Not in a DVC repository"
    exit 1
fi

# Check if data.csv exists
if [ ! -f "Data/data.csv" ]; then
    echo "Error: Data/data.csv not found"
    exit 1
fi

# Update DVC tracking
echo "Updating DVC tracking for data.csv..."
dvc add Data/data.csv

# Check for changes
if git diff --quiet HEAD Data/data.csv.dvc; then
    echo "No changes to data.csv detected"
else
    # Add to git and commit
    echo "Changes detected, creating snapshot..."
    git add Data/data.csv.dvc
    
    # Create commit with timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    commit_message="Data snapshot - automated DVC update - $timestamp"
    
    git commit -m "$commit_message"
    echo "Created DVC snapshot: $commit_message"
fi

echo "External DVC operations completed"
