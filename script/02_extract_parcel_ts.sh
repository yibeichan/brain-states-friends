#!/bin/bash
#SBATCH --job-name=02_extract_parcel_ts
#SBATCH --partition=mit_normal
#SBATCH --output=logs/02_extract_parcel_ts_%A_%a.out
#SBATCH --error=logs/02_extract_parcel_ts_%A_%a.err
#SBATCH --array=0-291
# IMPORTANT: Set --array based on the number of lines in your episode file (292 episodes = 0-291)
#SBATCH --time=00:05:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Shared preamble: PROJECT_DIR/SCRIPT_DIR, uv on PATH, logs/ (utils/_env.sh)
source "${SLURM_SUBMIT_DIR:-.}/script/utils/_env.sh" 2>/dev/null || source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/script/utils/_env.sh" || exit 1


# --- HARDCODED VARIABLES ---
# Change these values directly instead of using command line arguments
SUBJECT_ID="sub-04"
PARCELLATION="atlas-4S156Parcels"
# EPISODE_FILE="${PROJECT_DIR}/02_parcel_ts_missing_episodes_${PARCELLATION}_${SUBJECT_ID}.txt"
EPISODE_FILE="${PROJECT_DIR}/${SUBJECT_ID}_episode_ids.txt"

# Alternative: If you want to specify the exact path to the episode file:
# EPISODE_FILE="${PROJECT_DIR}/path/to/your/episode_list.txt"

# Check if SLURM_ARRAY_TASK_ID is set (i.e., running as part of a SLURM array)
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: This script is intended to be run as a SLURM array job via sbatch."
    echo "SLURM_ARRAY_TASK_ID is not set. If testing, call a single instance with a defined SLURM_ARRAY_TASK_ID."
    # Example for testing a single task: SLURM_ARRAY_TASK_ID=0 bash run_episode.sh
    exit 1
fi

# Check if the episode file exists
if [ ! -f "$EPISODE_FILE" ]; then
    echo "Error: Episode file '$EPISODE_FILE' does not exist."
    echo "Please check the path and filename in the script."
    exit 1
fi

# --- SLURM Array Task Logic ---
# SLURM_ARRAY_TASK_ID is 0-indexed, sed is 1-indexed
LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 1))
EPISODE_ID=$(sed -n "${LINE_NUMBER}p" "$EPISODE_FILE")

echo "--- SLURM Job Info (Task: $SLURM_ARRAY_TASK_ID) ---"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Subject: $SUBJECT_ID"
echo "Parcellation: $PARCELLATION"
echo "Episode File: $EPISODE_FILE"
echo "Attempting to process line $LINE_NUMBER from episode file."
echo "-----------------------------------------"

if [ -z "$EPISODE_ID" ]; then
    echo "No episode ID found on line $LINE_NUMBER of '$EPISODE_FILE'."
    echo "Task $SLURM_ARRAY_TASK_ID will now exit gracefully (this is normal for tasks beyond the actual file length if --array is set too large)."
    exit 0 
fi

echo "Processing Episode ID: '$EPISODE_ID'"

# Python script path (SCRIPT_DIR already defined at top of script)
PYTHON_SCRIPT_PATH="$SCRIPT_DIR/02_extract_parcel_ts.py"

if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "Error: Python script 02_extract_parcel_ts.py not found in script directory $SCRIPT_DIR."
    exit 1
fi

uv run --project "${PROJECT_DIR}" --no-sync python "$PYTHON_SCRIPT_PATH" --subject_id "$SUBJECT_ID" --parcellation "$PARCELLATION" --episode_id "$EPISODE_ID"

EXIT_STATUS=$?
if [ $EXIT_STATUS -eq 0 ]; then
    echo "Successfully processed episode: '$EPISODE_ID'"
else
    echo "Error processing episode: '$EPISODE_ID' (Python script exit status: $EXIT_STATUS)"
fi

echo "Task $SLURM_ARRAY_TASK_ID completed."

# Optional: Add seff for job resource usage, only if $SLURM_JOB_ID is available and meaningful
# if [ -n "$SLURM_JOB_ID" ]; then
#    echo "--- Resource Usage for Job $SLURM_JOB_ID (reported by seff) ---"
#    seff "$SLURM_JOB_ID"
# fi
