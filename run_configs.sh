#!/usr/bin/env bash
# Run all configs sequentially and tee output to a timestamped log file.
# Usage: ./run_configs.sh

set -euo pipefail

CONFIGS=(
  "configs/pipeline_configs/pipeline_AAV1.yaml"
  "configs/pipeline_configs/pipeline_AAV2.yaml"
  "configs/pipeline_configs/pipeline_best_of_5.yaml"
  "configs/pipeline_configs/pipeline_coarse_fine.yaml"
  "configs/pipeline_configs/pipeline_debateV1.yaml"
  "configs/pipeline_configs/pipeline_debateV2.yaml"
  "configs/pipeline_configs/pipeline_per_score.yaml"
)

VENV_PATH="./venv"
REQUIREMENTS_FILE="requirements.txt"

## Detection and Activation
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "No active virtual environment detected."
    
    if [[ -d "$VENV_PATH" ]]; then
        echo "Found existing venv at $VENV_PATH. Activating..."
        source "$VENV_PATH/bin/activate"
    else
        echo "Venv not found. Creating one at $VENV_PATH..."
        python3 -m venv "$VENV_PATH"
        source "$VENV_PATH/bin/activate"
    fi
else
    echo "Already running inside venv: $VIRTUAL_ENV"
fi

## Verify and Install Requirements
if [[ -f "$REQUIREMENTS_FILE" ]]; then
    # We use 'pip install' with the requirements file. 
    # Pip is smart enough to skip packages that are already satisfied,
    # making this both a 'verify' and an 'install' step.
    echo "Verifying dependencies..."
    pip install -r "$REQUIREMENTS_FILE" --quiet
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install requirements."
        exit 1
    fi
else
    echo "Warning: $REQUIREMENTS_FILE not found. Skipping dependency check."
fi

## 3. Continue Execution
echo "------------------------------------------"
echo "Environment ready. Starting config runs..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/responses/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/run_$TIMESTAMP.log"

log() {
  echo "$@" | tee -a "$LOG_FILE"
}

log "========================================================"
log "Run started: $(date)"
log "Log file: $LOG_FILE"
log "========================================================"
log ""

TOTAL=${#CONFIGS[@]}
for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  num=$((i + 1))
  log "--------------------------------------------------------"
  log "[$num/$TOTAL] Running: $cfg"
  log "Started: $(date)"
  log "--------------------------------------------------------"

  if python3 "$SCRIPT_DIR/llm_scoring.py" --config "$cfg" 2>&1 | tee -a "$LOG_FILE"; then
    log ""
    log "[$num/$TOTAL] DONE: $cfg"
  else
    log ""
    log "[$num/$TOTAL] FAILED: $cfg (exit code $?)"
    log "Continuing with next config..."
  fi

  log ""
done

log "========================================================"
log "All configs finished: $(date)"
log "Log saved to: $LOG_FILE"
log "========================================================"
