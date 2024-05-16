import datetime
import os
import subprocess

# Directories where the scripts are located
TRAIN_SCRIPT_DIR = "/Users/mchildress/Code/dreamers/Models"
TEST_SCRIPT_DIR = "/Users/mchildress/Code/dreamers/Models"

# Directory to store logs
# Update this path to a writable location
LOG_DIR = os.path.expanduser("/Users/mchildress/Code/dreamers/Analysis/logs")

# Ensure log directory exists
print(f"Creating log directory: {LOG_DIR}")
os.makedirs(LOG_DIR, exist_ok=True)

# Get current time for the log file
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(LOG_DIR, f"log_{current_time}.txt")
print(f"Log file will be saved to: {log_file}")

# Commands to run with full paths
commands = [
    f"python3 {os.path.join(TRAIN_SCRIPT_DIR, 'train.py')}",
    f"python3 {os.path.join(TEST_SCRIPT_DIR, 'test.py')}",
    f"python3 {os.path.join(TRAIN_SCRIPT_DIR, 'train_tensorflow.py')}",
    f"python3 {os.path.join(TEST_SCRIPT_DIR, 'test_tensorflow.py')}"
]

# Run commands and log output
with open(log_file, "w") as log:
    for command in commands:
        print(f"Running command: {command}")
        log.write(f"Running command: {command}\n")
        log.write("="*50 + "\n")

        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        final_output = []
        for line in process.stdout:
            print(line, end='')  # Print to terminal
            final_output.append(line)

        for line in process.stderr:
            print(line, end='')  # Print to terminal
            final_output.append(line)

        process.wait()

        # Write final output to log file
        # Adjust this to capture necessary final output
        log.write("\n".join(final_output[-20:]))

        if process.returncode != 0:
            log.write(
                f"\nCommand '{command}' failed with return code {process.returncode}\n\n")
            print(
                f"Command '{command}' failed with return code {process.returncode}")

print("All commands executed. Logs are saved in:", log_file)
