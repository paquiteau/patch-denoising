#!/bin/bash

# Usage: ./denoise_bids.sh -d /path/to/bids -t [TASK] --threshold 0.5 --method highpass
# ------------------------------------------------------------------------------
set -e 
# Default values
DATASET_PATH=""
TASK_NAME=""

# 1. Parse script-specific arguments
while getopts "d:t:" opt; do
  case $opt in
    d) DATASET_PATH="$OPTARG" ;;
    t) TASK_NAME="$OPTARG" ;;
    *) echo "Usage: $0 -d <dataset_path> -t <task_name> [tool_args]"; exit 1 ;;
  esac
done

# Shift the positional parameters so $@ only contains the remaining arguments
shift $((OPTIND-1))
TOOL_ARGS="$@"

# 2. Validation
if [[ -z "$DATASET_PATH" || -z "$TASK_NAME" ]]; then
    echo "Error: Missing dataset path (-d) or task name (-t)."
    exit 1
fi

# 3. Define Derivative Path
DERIV_DIR="$DATASET_PATH/derivatives/denoised"
mkdir -p "$DERIV_DIR"

# Create dataset_description.json if it doesn't exist (BIDS compliance)
if [ ! -f "$DERIV_DIR/dataset_description.json" ]; then
    echo '{"Name": "Denoised Data", "BIDSVersion": "1.8.0", "DatasetType": "derivative"}' > "$DERIV_DIR/dataset_description.json"
fi

echo "Starting denoising for task: $TASK_NAME"
echo "Passing to patch-denoise: $TOOL_ARGS"

# 4. Loop through Subjects
for sub_path in "$DATASET_PATH"/sourcedata/sub-*; do
    sub=$(basename "$sub_path")
    
    for ses_path in "$sub_path"/ses-*; do
        [ -d "$ses_path" ] || continue
        ses=$(basename "$ses_path")
        # Check if subject has a func directory
        if [ -d "$ses_path/func" ]; then
            
            # Find all runs for the specified task
            for run_file in "$ses_path/func"/*_task-"$TASK_NAME"*_bold.nii.gz; do
                # Skip if no files match
                [ -e "$run_file" ] || continue
                # Prepare Derivative Path: derivatives/pipeline/sub-xx/ses-yy/func/
                out_func_dir="$DERIV_DIR/$sub/$ses/func"
                mkdir -p "$out_func_dir"

                # Build Output Filename
                base_name=$(basename "$run_file" _bold.nii.gz)
                out_file="$out_func_dir/${base_name}_desc-denoised_bold.nii.gz"

                echo "Processing: $sub | $ses | $(basename "$run_file")"
                
                # 4. EXECUTE COMMAND
                # 5. EXECUTE CLI COMMAND
                # Replace 'my_denoise_tool' with your actual command
                uv run patch-denoise --gpu $TOOL_ARGS "$run_file"  "$out_file" 
                
                # 6. Copy Sidecar JSON (Essential for BIDS)
                json_src="${run_file%.nii.gz}.json"
                json_dest="${out_file%.nii.gz}.json"
                if [ -f "$json_src" ]; then
                    cp "$json_src" "$json_dest"
                fi
            done
        fi
    done
done

echo "Workflow complete. Results are in $DERIV_DIR"
