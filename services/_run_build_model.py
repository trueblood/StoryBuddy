
import subprocess
import sys

def run_script():
    try:
        # Running the third_transformer.py script
        subprocess.check_call([sys.executable, 'third_transformer.py'])
    except subprocess.CalledProcessError:
        # If third_transformer.py crashes, rerun it
        print("Script crashed, rerunning...")
        run_script()

if __name__ == "__main__":
    run_script()
