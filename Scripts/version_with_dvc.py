# 225150200111004_1 HAIKAL THORIQ ATHAYA_1
# 225150200111008_2 MUHAMMAD ARSYA ZAIN YASHIFA_2
# 225150201111001_3 JAVIER AAHMES REANSYAH_3
# 225150201111003_4 MUHAMMAD HERDI ADAM_4

import subprocess
from datetime import datetime
import pytz
import sys
import shutil
import os

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
    
    source_file = 'Data/data.csv'
    destination_file = 'Data/ci_train_data.csv'

    if not os.path.exists(source_file):
        print(f"ERROR: Source file not found at '{source_file}'. Aborting process.")
        sys.exit(1)
    
    print(f"Source file '{source_file}' found. Proceeding with versioning.")

    # Get current branch
    current_branch_result = subprocess.run(['git', 'branch', '--show-current'], 
                                         capture_output=True, text=True, check=True)
    current_branch = current_branch_result.stdout.strip()
    
    wib_tz = pytz.timezone('Asia/Jakarta')
    wib_time = datetime.now(wib_tz)
    timestamp_str = wib_time.strftime('%Y-%m-%d-%H-%M-%S')
    new_branch_name = f"data-update-{timestamp_str}"
    commit_message = f"Update data version: {wib_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"

    # Branch selection
    print("\n" + "="*50)
    print("BRANCH SELECTION:")
    print(f"Current branch: {current_branch}")
    print("Choose your branching strategy:")
    print("1. Commit to current branch")
    print("2. Create new branch from main")
    print("="*50)
    
    while True:
        branch_choice = input("Enter your choice (1 or 2): ").strip()
        if branch_choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    if branch_choice == '1':
        print(f"Using current branch: {current_branch}")
        working_branch = current_branch
    else:
        print("Switching to main branch and pulling latest changes...")
        run_command(['git', 'switch', 'main'])
        run_command(['git', 'pull', 'origin', 'main'])
        print(f"Creating and switching to new branch: {new_branch_name}")
        run_command(['git', 'checkout', '-b', new_branch_name])
        working_branch = new_branch_name

    try:
        print(f"Copying {source_file} to {destination_file}...")
        shutil.copy(source_file, destination_file)
        print(f"Successfully copied data to {destination_file}.")
    except Exception as e:
        print(f"ERROR: Failed to copy file: {e}")
        sys.exit(1)

    run_command(['dvc', 'add', source_file])
    run_command(['git', 'add', 'Data/data.csv.dvc', destination_file])

    print(f"Commit message: '{commit_message}'")
    run_command(['git', 'commit', '-m', commit_message])

    # Push options
    print("\n" + "="*50)
    print("PUSH OPTIONS:")
    print(f"Current working branch: {working_branch}")
    
    if branch_choice == '1':
        # Current branch - ask if want to push
        response = input(f"Push changes to remote branch '{working_branch}'? (y/N): ").lower().strip()
        if response == 'y':
            print("Pushing to remote...")
            run_command(['git', 'push', 'origin', working_branch])
            print(f"\nVersioning process completed and changes pushed to '{working_branch}' successfully!")
        else:
            print("\nSkipping push.")
            print("Local versioning process completed successfully!")
            print(f"Remember to run 'git push origin {working_branch}' manually when you are ready.")
    else:
        # New branch - ask if want to push
        response = input(f"Push new branch '{working_branch}' to remote git repository? (y/N): ").lower().strip()
        if response == 'y':
            print("Pushing to remote...")
            run_command(['git', 'push', '--set-upstream', 'origin', working_branch])
            print(f"\nVersioning process completed and new branch '{working_branch}' pushed successfully!")
            print("You can now create a Pull Request on GitHub/GitLab.")
        else:
            print("\nSkipping push.")
            print("Local versioning process completed successfully!")
            print(f"Remember to run 'git push origin {working_branch}' manually when you are ready.")
    
    print("="*50)


if __name__ == "__main__":
    main()
