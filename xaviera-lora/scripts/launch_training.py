"""
Launch SageMaker LoRA fine-tuning using ACE-Step code in ../ACE-Step and
the dataset index created by prepare_s3_index.py.

Assumes you run from /home/ec2-user/SageMaker/xaviera-lora:
    python scripts/launch_training.py
"""

import os
import shutil
import sagemaker
from sagemaker.pytorch import PyTorch


# --- PATHS ---
ROOT = "/home/ec2-user/SageMaker"
XAVIERA_ROOT = os.path.join(ROOT, "xaviera-lora")
ACE_STEP_ROOT = os.path.join(ROOT, "ACE-Step")

LOCAL_DATASET_DIR = os.path.join(XAVIERA_ROOT, "data", "datasets", "pop_500_index")
LOCAL_LORA_CONFIG = os.path.join(XAVIERA_ROOT, "config", "lora_config.json")

# Copy config into ACE-Step so it is bundled with source_dir upload
ACE_LORA_CONFIG = os.path.join(ACE_STEP_ROOT, "config", "lora_config.json")

# --- S3 INPUTS ---
# The audio already lives on S3; we just point to it as a channel.
S3_AUDIO_URI = "s3://xaviera-training-file/000001/Pop/"


def main():
    sess = sagemaker.Session()
    role = sagemaker.get_execution_role()

    # 1) Upload dataset index to S3
    print(f"Uploading dataset index from {LOCAL_DATASET_DIR} ...")
    dataset_s3_uri = sess.upload_data(
        path=LOCAL_DATASET_DIR, key_prefix="xaviera-lora/datasets/pop_500_index"
    )
    print(f"Dataset index uploaded to: {dataset_s3_uri}")

    # 2) Ensure LoRA config is inside ACE-Step before upload
    os.makedirs(os.path.dirname(ACE_LORA_CONFIG), exist_ok=True)
    shutil.copyfile(LOCAL_LORA_CONFIG, ACE_LORA_CONFIG)
    print(f"Copied LoRA config into ACE-Step: {ACE_LORA_CONFIG}")

    # 3) Define input channels
    inputs = {
        "dataset": dataset_s3_uri,  # small index
        "audio": S3_AUDIO_URI,  # large audio
    }

    # 4) Configure estimator
    estimator = PyTorch(
        entry_point="trainer.py",
        source_dir=ACE_STEP_ROOT,  # upload ACE-Step code (contains trainer.py & acestep/)
        role=role,
        instance_count=1,
        instance_type="ml.g6.2xlarge",
        framework_version="2.1",
        py_version="py310",
        output_path="s3://xaviera-lora-checkpoints/",
        hyperparameters={
            "dataset_path": "/opt/ml/input/data/dataset",
            "exp_name": "pop_finetune",
            "lora_config_path": "config/lora_config.json",
            "max_steps": 2500,
            "learning_rate": 1e-4,
            "every_plot_step": 500,
            "logger_dir": "/opt/ml/output/data/logs",
            "checkpoint_dir": "/opt/ml/model",
            "num_workers": 1,          # Reduce memory from data loaders
            "precision": "bf16-mixed", # Use bf16 to halve memory
            "s3_output_path": "s3://xaviera-lora-checkpoints",  # Save LoRA adapters directly to S3
        },
        environment={
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        # Use an absolute path so the file is found when packaging the code.
        dependencies=[os.path.join(ACE_STEP_ROOT, "requirements.txt")],
    )

    # 5) Launch training
    print("Starting SageMaker training job...")
    estimator.fit(inputs)
    print("Training job submitted.")


if __name__ == "__main__":
    main()

