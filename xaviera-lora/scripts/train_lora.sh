 #!/usr/bin/env bash
 set -euo pipefail

 # Usage:
 #   train_lora.sh DATASET_DIR EXP_NAME MAX_STEPS
 # Example:
 #   train_lora.sh \
 #     "/mnt/c/Users/Test/Desktop/ACE-Step/xaviera-lora/data/datasets/alternative_subset_10" \
 #     "lora_alt_10" \
 #     6000

 DATASET_DIR=${1:?dataset dir required}
 EXP_NAME=${2:?exp name required}
 MAX_STEPS=${3:-6000}

 export CUDA_VISIBLE_DEVICES=0

 python /mnt/c/Users/Test/Desktop/ACE-Step/trainer.py \
   --dataset_path "$DATASET_DIR" \
   --exp_name "$EXP_NAME" \
   --lora_config_path "/mnt/c/Users/Test/Desktop/ACE-Step/xaviera-lora/config/lora_config.json" \
   --max_steps "$MAX_STEPS" \
   --every_n_train_steps 3 \
   --every_plot_step 3 \
   --accumulate_grad_batches 15 \
   --learning_rate 1e-4 \
   --devices 1 \
   --precision bf16-mixed \
   --logger_dir "/mnt/c/Users/Test/Desktop/ACE-Step/xaviera-lora/logs" \  
   --checkpoint_dir "/mnt/c/Users/Test/Desktop/ACE-Step/xaviera-lora/checkpoints"


 #  every_n_train_steps 200 ------ step save checkpoint. convert to 200 in production.
 #  every_plot_step 200 ------ step save audio samples. convert to 200 in production.