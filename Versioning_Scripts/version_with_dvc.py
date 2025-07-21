#!/usr/bin/env python3

import subprocess
from datetime import datetime
import pytz
import sys

def run_command(command):
    print(f"Executing: {' '.join(command)}")
    try:
        result = subprocess.run(
            command, 
            check=True, 
            text=True, 
            capture_output=True
        )
        if result.stdout:
            print(f"Output:\n{result.stdout.strip()}")
        if result.stderr:
            print(f"Stderr:\n{result.stderr.strip()}")
    except FileNotFoundError:
        print(f"ERROR: Command not found: '{command[0]}'. Is it installed and in your PATH?")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}.")
        print(f"    Command: {' '.join(e.cmd)}")
        if e.stdout:
            print(f"    Stdout:\n{e.stdout.strip()}")
        if e.stderr:
            print(f"    Stderr:\n{e.stderr.strip()}")
        sys.exit(1)

def main():
    print("Starting DVC data versioning process...")
    
    run_command(['dvc', 'add', 'Data/data.csv'])
    run_command(['git', 'add', 'Data/data.csv.dvc'])

    wib_tz = pytz.timezone('Asia/Jakarta')
    wib_time = datetime.now(wib_tz)
    commit_message = f"Update data version: {wib_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    print(f"Commit message: '{commit_message}'")

    run_command(['git', 'commit', '-m', commit_message])

    print("\n" + "="*50)
    response = input("Push changes to remote git repository? (y/N): ").lower().strip()
    
    if response == 'y':
        print("Pushing to remote...")
        run_command(['git', 'push'])
        print("\nVersioning process completed and changes pushed successfully!")
    else:
        print("\nSkipping push.")
        print("Local versioning process completed successfully!")
        print("Remember to run 'git push' manually when you are ready.")
    print("="*50)


if __name__ == "__main__":
    main()
