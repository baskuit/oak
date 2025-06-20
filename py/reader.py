import subprocess
import time
import signal
import sys
import os
print("Python cwd:", os.getcwd())

# Start the process
proc = subprocess.Popen(sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

time.sleep(3)  # or however long you want

proc.send_signal(signal.SIGINT)

# Wait for the process to finish and capture output
stdout, stderr = proc.communicate()

# The C++ program should print the float value before exiting.
# For example, if it prints the float as the last line:
lines = stdout.strip().splitlines()
if lines:
    try:
        # Try converting the last line to float
        result = float(lines[-1])
        print("Received float value:", result)
    except ValueError:
        print("Could not parse float from output:", lines[-1])
else:
    print("No output from process")

# Optionally handle stderr if needed
if stderr:
    print("Error output from program:", stderr)
