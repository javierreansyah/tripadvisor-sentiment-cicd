import os
import subprocess
import shutil
from datetime import datetime
from typing import Optional, Tuple
import asyncio

from app.config import DATA_DIR


class DVCManager:
    """Manages DVC operations for data versioning during retraining."""
    
    def __init__(self, repo_root: str = "/app"):
        """
        Initialize DVC manager.
        
        Args:
            repo_root: Root directory of the repository (inside container)
        """
        self.repo_root = repo_root
        self.data_dir = os.path.join(repo_root, DATA_DIR)
        self.main_data_path = os.path.join(self.data_dir, 'data.csv')
        self.new_data_path = os.path.join(self.data_dir, 'new_data.csv')
    
    async def _run_dvc_command(self, command: list) -> Tuple[bool, str]:
        """
        Run a DVC command asynchronously.
        
        Args:
            command: List of command parts
            
        Returns:
            Tuple of (success, output)
        """
        try:
            # Change to repo root directory for DVC commands
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.repo_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return True, stdout.decode()
            else:
                return False, stderr.decode()
                
        except Exception as e:
            return False, f"Error running DVC command: {str(e)}"
    
    async def ensure_data_tracked(self) -> bool:
        """
        Ensure that data.csv is tracked by DVC.
        Only tracks if not already tracked.
        
        Returns:
            bool: True if successfully tracked or already tracked
        """
        print("Ensuring data.csv is tracked by DVC...")
        
        # Check if data.csv is already tracked
        data_dvc_file = os.path.join(self.data_dir, 'data.csv.dvc')
        
        if os.path.exists(data_dvc_file):
            print("data.csv is already tracked by DVC")
            return True
        
        # Track data.csv with DVC
        success, output = await self._run_dvc_command(['dvc', 'add', self.main_data_path])
        
        if success:
            print(f"Successfully added data.csv to DVC tracking")
            
            # Add the .dvc file to git
            git_success, git_output = await self._run_dvc_command([
                'git', 'add', f'{self.main_data_path}.dvc', '.gitignore'
            ])
            
            if git_success:
                print("Added data.csv.dvc to git")
                return True
            else:
                print(f"Warning: Could not add .dvc file to git: {git_output}")
                return True  # DVC add was successful even if git add failed
        else:
            print(f"Error adding data.csv to DVC: {output}")
            return False
    
    async def create_data_snapshot(self, message: Optional[str] = None) -> bool:
        """
        Create a snapshot of the current data state.
        
        Args:
            message: Optional commit message
            
        Returns:
            bool: True if snapshot was created successfully
        """
        print("Creating data snapshot with DVC...")
        
        # Create timestamp for default message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = message or f"Data snapshot before retraining - {timestamp}"
        
        # First, ensure data is up to date in DVC
        success, output = await self._run_dvc_command(['dvc', 'add', self.main_data_path])
        
        if not success:
            print(f"Error updating DVC tracking: {output}")
            return False
        
        print("Updated DVC tracking for data.csv")
        
        # Add the updated .dvc file to git
        git_add_success, git_add_output = await self._run_dvc_command([
            'git', 'add', f'{self.main_data_path}.dvc'
        ])
        
        if not git_add_success:
            print(f"Warning: Could not stage .dvc file: {git_add_output}")
        
        # Create git commit with the snapshot
        commit_success, commit_output = await self._run_dvc_command([
            'git', 'commit', '-m', commit_message
        ])
        
        if commit_success:
            print(f"Created data snapshot: {commit_message}")
            return True
        else:
            # Check if it's just "nothing to commit"
            if "nothing to commit" in commit_output.lower():
                print("No changes to commit - data already up to date")
                return True
            else:
                print(f"Error creating git commit: {commit_output}")
                return False
    
    async def should_track_new_data(self) -> bool:
        """
        Determine if new_data.csv should be tracked.
        
        Based on your description, new_data.csv is temporary storage,
        but it might be useful to track it for debugging purposes.
        
        Returns:
            bool: Whether new_data.csv should be tracked
        """
        # Check if new_data.csv exists and has substantial content
        if not os.path.exists(self.new_data_path):
            return False
        
        try:
            # Get file size
            file_size = os.path.getsize(self.new_data_path)
            # Only track if it's larger than 1KB (has substantial content)
            return file_size > 1024
        except:
            return False
    
    async def backup_new_data(self) -> bool:
        """
        Create a backup of new_data.csv before processing.
        This is useful for debugging and recovery.
        
        Returns:
            bool: True if backup was created successfully
        """
        if not os.path.exists(self.new_data_path):
            return True  # Nothing to backup
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.data_dir, f'new_data_backup_{timestamp}.csv')
            shutil.copy2(self.new_data_path, backup_path)
            print(f"Created backup of new_data.csv: {backup_path}")
            return True
        except Exception as e:
            print(f"Error creating backup of new_data.csv: {e}")
            return False
    
    async def get_data_info(self) -> dict:
        """
        Get information about the current data state.
        
        Returns:
            dict: Information about data files and DVC status
        """
        info = {
            'data_csv_exists': os.path.exists(self.main_data_path),
            'new_data_csv_exists': os.path.exists(self.new_data_path),
            'data_csv_tracked': os.path.exists(f'{self.main_data_path}.dvc'),
            'data_csv_size': 0,
            'new_data_csv_size': 0
        }
        
        if info['data_csv_exists']:
            info['data_csv_size'] = os.path.getsize(self.main_data_path)
        
        if info['new_data_csv_exists']:
            info['new_data_csv_size'] = os.path.getsize(self.new_data_path)
        
        return info


# Global DVC manager instance
dvc_manager = DVCManager()
