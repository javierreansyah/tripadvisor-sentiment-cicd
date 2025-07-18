#!/usr/bin/env python3
"""
Test script to verify DVC integration functionality.
Run this to test the DVC manager without triggering a full retraining.
"""

import asyncio
import sys
import os
sys.path.append('FastAPI')

from app.services.dvc_manager import dvc_manager

async def test_dvc_integration():
    """Test the DVC integration functionality."""
    print("Testing DVC Integration...")
    print("=" * 50)
    
    # Test 1: Get data information
    print("Test 1: Getting data information...")
    data_info = await dvc_manager.get_data_info()
    print(f"Data info: {data_info}")
    print()
    
    # Test 2: Ensure data tracking
    print("Test 2: Ensuring data.csv is tracked...")
    tracking_result = await dvc_manager.ensure_data_tracked()
    print(f"Tracking result: {tracking_result}")
    print()
    
    # Test 3: Create a test snapshot
    print("Test 3: Creating test snapshot...")
    snapshot_result = await dvc_manager.create_data_snapshot("Test snapshot - DVC integration verification")
    print(f"Snapshot result: {snapshot_result}")
    print()
    
    print("DVC Integration Test Complete!")
    print("=" * 50)
    
    # Summary
    print("Summary:")
    print(f"✅ Data tracking: {'OK' if tracking_result else 'FAILED'}")
    print(f"✅ Snapshot creation: {'OK' if snapshot_result else 'FAILED'}")
    print(f"✅ Data info retrieval: {'OK' if data_info else 'FAILED'}")

if __name__ == "__main__":
    asyncio.run(test_dvc_integration())
