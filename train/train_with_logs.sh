#!/bin/bash


# Pub IP
PUBLIC_IP=$(curl -s ifconfig.me)

if [ -z "$PUBLIC_IP" ]; then
  echo "No valid IP get!"
  exit 1
fi

LAST_TWO_OCTETS=$(echo "$PUBLIC_IP" | awk -F'.' '{print $3"."$4}')

# --- Configuration Section ---
# 1. Define the log directory
LOG_DIR="./task_logs/${PUBLIC_IP}/train_log"

# 2. Define the prefix for output files
LOG_PREFIX="training_output_"

# 3. Define the maximum size of a single log file (1MB)
LOG_SIZE='1M'

# 4. Define the maximum number of log files to keep (mimics backupCount behavior)
MAX_LOG_FILES=5000

# 5. The Python command you want to run (enclosed in quotes)
PYTHON_COMMAND="python train/train_agent.py"
# Or a more complex command, for example:
# PYTHON_COMMAND="python -m agentflow.verl algorithm.adv_estimator=grpo data.train_batch_size=8"

# --- Main Logic ---

# Remove and recreate the log directory for a clean start
rm -rf $LOG_DIR
mkdir -p $LOG_DIR # Using -p is safer, it doesn't error if the directory exists

# Calculate the required number of digits for the suffix based on MAX_LOG_FILES
# For example, if MAX_LOG_FILES is 5000, the largest index is 4999, which has 4 digits.
MAX_INDEX=$((MAX_LOG_FILES - 1))
SUFFIX_DIGITS=${#MAX_INDEX}

echo "Starting the task... Log files will use $SUFFIX_DIGITS-digit suffixes (e.g., 0001, 0002...)"

# Execute the Python command and pipe its output to 'split'
# -d uses numeric suffixes
# -a specifies the length of the suffix, automatically adding leading zeros
PYTHONUNBUFFERED=1 $PYTHON_COMMAND 2>&1 | \
    split -b "$LOG_SIZE" -d -a "$SUFFIX_DIGITS" - "$LOG_DIR/$LOG_PREFIX"

# Get the exit status of the 'split' command from the pipe
SPLIT_EXIT_CODE=${PIPESTATUS[1]}

# Check if the command executed successfully
if [ $SPLIT_EXIT_CODE -eq 0 ]; then
    echo "Task completed successfully."
else
    echo "Error: The task or log splitting failed with exit code $SPLIT_EXIT_CODE."
    exit $SPLIT_EXIT_CODE
fi

# Clean up: keep only the newest MAX_LOG_FILES files (by modification time)
# This is a robust way to handle cleanup for a single, long-running process
echo "Cleaning up old log files, keeping the latest $MAX_LOG_FILES..."
ls -1t "$LOG_DIR"/"$LOG_PREFIX"* 2>/dev/null | \
    tail -n +$((MAX_LOG_FILES + 1)) | \
    xargs rm -f

echo "Log files are saved in: $LOG_DIR"