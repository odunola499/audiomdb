import subprocess
import sys
import os

def run_tests():
    os.chdir('/home/odunola/Documents/papers/current_projects/audiomdb')
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'
        ], capture_output=True, text=True, timeout=120)
        
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")  
        print(result.stderr)
        print(f"\nReturn code: {result.returncode}")
        
    except subprocess.TimeoutExpired:
        print("Tests timed out after 120 seconds")
    except Exception as e:
        print(f"Error running tests: {e}")

if __name__ == "__main__":
    run_tests()