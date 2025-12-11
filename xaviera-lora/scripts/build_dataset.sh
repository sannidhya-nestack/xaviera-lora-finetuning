#!/usr/bin/env bash
set -euo pipefail

# Arguments:
#   $1 = prepared_manifest.csv
#   $2 = output_dataset_dir
#   $3 = repeat_count (optional, default: 1)

REPEAT_COUNT=${3:-1}

python /mnt/c/Users/Test/Desktop/ACE-Step/xaviera-lora/scripts/manifest_to_hf_dataset.py \
  --prepared_csv "$1" \
  --output_dir "$2" \
  --repeat_count "$REPEAT_COUNT"


