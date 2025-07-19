import os
import subprocess
from app.config import DATA_DIR

class DVCManager:
    def __init__(self):
        # Find repo root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        while not os.path.exists(os.path.join(current_dir, '.dvc')):
            current_dir = os.path.dirname(current_dir)
        
        self.repo_root = current_dir
        self.data_path = os.path.join(current_dir, DATA_DIR, 'data.csv')
    
    def version_data(self, message="Data version update"):
        """Add data to DVC and commit changes."""
        os.chdir(self.repo_root)
        
        # Add to DVC
        subprocess.run(['dvc', 'add', self.data_path])
        
        # Add to git and commit
        subprocess.run(['git', 'add', f'{self.data_path}.dvc'])
        subprocess.run(['git', 'commit', '-m', message])

dvc_manager = DVCManager()