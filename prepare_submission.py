#!/usr/bin/env python3
"""
Final Submission Preparation Guide
Follow this step-by-step to prepare your best agent for submission.
"""

import subprocess
import shutil
import sys
from pathlib import Path
from datetime import datetime

WORKSPACE = Path(__file__).parent

def check_agent_exists(agent_name):
    """Check if agent directory exists."""
    agent_path = WORKSPACE / agent_name
    if not agent_path.exists():
        print(f"✗ Agent directory not found: {agent_path}")
        return False
    
    sudokuai = agent_path / 'sudokuai.py'
    if not sudokuai.exists():
        print(f"✗ sudokuai.py not found in {agent_path}")
        return False
    
    init_py = agent_path / '__init__.py'
    if not init_py.exists():
        print(f"✗ __init__.py not found in {agent_path}")
        return False
    
    print(f"✓ Agent directory structure is valid")
    return True

def verify_syntax(agent_name):
    """Verify Python syntax."""
    sudokuai = WORKSPACE / agent_name / 'sudokuai.py'
    try:
        result = subprocess.run(
            ['python', '-m', 'py_compile', str(sudokuai)],
            capture_output=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"✓ Python syntax is valid")
            return True
        else:
            print(f"✗ Syntax error:")
            print(result.stderr.decode())
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_agent(agent_name, opponent='greedy_player', think_time=3):
    """Test agent with single game."""
    print(f"\nTesting {agent_name} against {opponent} with {think_time}s think time...")
    try:
        result = subprocess.run(
            ['python', str(WORKSPACE / 'play_match.py'), agent_name, opponent, str(think_time)],
            capture_output=True,
            text=True,
            timeout=think_time * 2 + 60
        )
        
        output = result.stdout + result.stderr
        
        if result.returncode == 0 and ('move' in output.lower() or 'win' in output.lower() or 'score' in output.lower()):
            print(f"✓ Test game completed successfully")
            return True
        else:
            print(f"✗ Test game failed")
            print(f"Output: {output[:300]}")
            return False
    
    except subprocess.TimeoutExpired:
        print(f"✗ Test game timed out")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def backup_agent(agent_name):
    """Create backup of current team32_A1."""
    backup_path = WORKSPACE / f'team32_A1_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    original = WORKSPACE / 'team32_A1'
    
    if original.exists():
        try:
            shutil.copytree(original, backup_path)
            print(f"✓ Created backup at: {backup_path}")
            return True
        except Exception as e:
            print(f"✗ Backup failed: {e}")
            return False
    
    return True

def copy_agent_to_submission(source_agent):
    """Copy agent to team32_A1 for submission."""
    source = WORKSPACE / source_agent
    dest = WORKSPACE / 'team32_A1'
    
    if not source.exists():
        print(f"✗ Source agent not found: {source}")
        return False
    
    try:
        # Backup current team32_A1
        backup_agent('team32_A1')
        
        # Remove old team32_A1
        if dest.exists():
            shutil.rmtree(dest)
        
        # Copy new agent
        shutil.copytree(source, dest)
        print(f"✓ Copied {source_agent} to team32_A1")
        return True
    
    except Exception as e:
        print(f"✗ Copy failed: {e}")
        return False

def create_submission_zip(agent_name='team32_A1'):
    """Create zip file for submission."""
    agent_path = WORKSPACE / agent_name
    
    if not agent_path.exists():
        print(f"✗ Agent not found: {agent_path}")
        return False
    
    try:
        zip_name = WORKSPACE / f'{agent_name}.zip'
        shutil.make_archive(str(zip_name.with_suffix('')), 'zip', WORKSPACE, agent_name)
        
        if zip_name.exists():
            size_mb = zip_name.stat().st_size / (1024 * 1024)
            print(f"✓ Created submission: {zip_name} ({size_mb:.2f} MB)")
            return True
        else:
            print(f"✗ Zip file not created")
            return False
    
    except Exception as e:
        print(f"✗ Zip creation failed: {e}")
        return False

def main():
    print("=" * 80)
    print("A2 SUBMISSION PREPARATION GUIDE")
    print("=" * 80)
    
    print("\nStep 1: Choose Your Agent")
    print("-" * 80)
    
    agents = [
        'team32_A1',
        'team32_A2_heuristic',
        'team32_A2_mcts',
        'team32_A2_positional',
        'team32_A2_hybrid',
    ]
    
    print("\nAvailable agents for submission:")
    for i, agent in enumerate(agents, 1):
        print(f"  {i}. {agent}")
    
    while True:
        try:
            choice = input("\nEnter the number of your best agent (1-5): ")
            agent_idx = int(choice) - 1
            if 0 <= agent_idx < len(agents):
                selected_agent = agents[agent_idx]
                break
            else:
                print("Invalid choice. Please enter 1-5.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    print(f"\n✓ Selected: {selected_agent}")
    
    print("\n\nStep 2: Validate Selected Agent")
    print("-" * 80)
    
    if not check_agent_exists(selected_agent):
        print("✗ Agent validation failed. Cannot proceed.")
        return False
    
    if not verify_syntax(selected_agent):
        print("✗ Syntax check failed. Fix errors before proceeding.")
        return False
    
    print("\n\nStep 3: Test Agent")
    print("-" * 80)
    
    if not test_agent(selected_agent, think_time=3):
        proceed = input("\nWarning: Test failed. Proceed anyway? (y/n): ")
        if proceed.lower() != 'y':
            print("Submission cancelled.")
            return False
    
    print("\n\nStep 4: Prepare Submission")
    print("-" * 80)
    
    if selected_agent != 'team32_A1':
        print(f"\nCopying {selected_agent} to team32_A1 for submission...")
        if not copy_agent_to_submission(selected_agent):
            print("✗ Copy failed. Submission preparation cancelled.")
            return False
    
    print("\n\nStep 5: Create Submission Package")
    print("-" * 80)
    
    if not create_submission_zip('team32_A1'):
        print("✗ Zip creation failed.")
        return False
    
    print("\n\n" + "=" * 80)
    print("SUBMISSION READY!")
    print("=" * 80)
    
    print(f"\nSelected Agent: {selected_agent}")
    print(f"Submission Package: team32_A1.zip")
    print(f"\nNext steps:")
    print("1. Download team32_A1.zip")
    print("2. Submit to Momotor before deadline: Friday Jan 9, 2026 21:00")
    print("3. You can resubmit multiple times - last submission counts")
    print("\nFor exam preparation, remember:")
    print("- Be able to explain your heuristics")
    print("- Document your experimental methodology")
    print("- Understand why this agent is strongest")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSubmission preparation cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        sys.exit(1)
