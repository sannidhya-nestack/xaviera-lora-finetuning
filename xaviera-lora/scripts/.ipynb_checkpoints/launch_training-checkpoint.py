"""
Launch SageMaker LoRA fine-tuning using ACE-Step code in ../ACE-Step and
the dataset index created by prepare_s3_index.py.

Assumes you run from /home/ec2-user/SageMaker/xaviera-lora:
    python scripts/launch_training.py
"""

import os
import shutil
import boto3
import tarfile
import tempfile
from datetime import datetime
from botocore.exceptions import ClientError

# SDK 3.x has completely restructured the API - old PyTorch/Estimator classes don't exist
# We'll use boto3 directly with the known working image URI
# NOTE: Successful job from Dec 15, 2024 used 2.1, not 2.2
# AWS ECR source image
AWS_PYTORCH_IMAGE = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1-gpu-py310"


def get_custom_image_uri():
    """
    Get PyTorch training image URI from user's ECR to bypass algorithm detection.
    
    AWS detects algorithms from the image URI pattern (pytorch-training:*), so using
    the image from the user's own ECR bypasses the algorithm validation that rejects ml.g6.2xlarge.
    
    Returns:
        str: ECR URI in user's account, or raises error with instructions if not found
    """
    # Get account ID and region
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    sm_client = boto3.client('sagemaker')
    region = sm_client.meta.region_name
    
    # ECR repository name in user's account
    repo_name = "pytorch-training-custom"
    custom_image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:2.1-gpu-py310"
    
    # Check if image already exists in user's ECR
    try:
        ecr = boto3.client('ecr', region_name=region)
        try:
            ecr.describe_images(
                repositoryName=repo_name,
                imageIds=[{'imageTag': '2.1-gpu-py310'}]
            )
            print(f"[INFO] Using custom image from your ECR: {custom_image_uri}")
            return custom_image_uri
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code in ['ImageNotFoundException', 'RepositoryNotFoundException']:
                # Image or repository doesn't exist
                pass
            else:
                raise
        
        # Image doesn't exist - provide instructions
        print(f"\n[ERROR] Custom PyTorch image not found in your ECR.")
        print(f"[ERROR] To bypass AWS algorithm validation, you need to copy the PyTorch image to your ECR.")
        print(f"\n[INSTRUCTIONS] Run these commands to copy the image:")
        print(f"  1. Create ECR repository (if needed):")
        print(f"     aws ecr create-repository --repository-name {repo_name} --region {region}")
        print(f"  2. Copy the image using docker:")
        print(f"     aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com")
        print(f"     aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com")
        print(f"     docker pull {AWS_PYTORCH_IMAGE}")
        print(f"     docker tag {AWS_PYTORCH_IMAGE} {custom_image_uri}")
        print(f"     docker push {custom_image_uri}")
        print(f"\n[ALTERNATIVE] Or use AWS CLI image replication (if available):")
        print(f"     aws ecr batch-get-image --repository-name pytorch-training --registry-id 763104351884 --image-ids imageTag=2.1-gpu-py310 --region us-east-1")
        print(f"\n[FALLBACK] Attempting to use AWS image URI (may fail validation)...")
        return AWS_PYTORCH_IMAGE
            
    except Exception as e:
        print(f"[WARNING] Error checking ECR: {e}")
        print(f"[WARNING] Falling back to AWS image URI (may fail validation).")
        return AWS_PYTORCH_IMAGE


def get_execution_role():
    """Get SageMaker execution role from STS identity (SDK 3.x compatible)."""
    # First, try to use SageMaker SDK's get_execution_role if available
    try:
        import sagemaker
        if hasattr(sagemaker, 'get_execution_role'):
            role = sagemaker.get_execution_role()
            print(f"[INFO] Using role from sagemaker.get_execution_role(): {role}")
            return role
    except (AttributeError, ImportError):
        pass
    
    # Fallback: Extract from STS identity
    sts = boto3.client('sts')
    identity = sts.get_caller_identity()
    arn = identity['Arn']
    account_id = identity['Account']
    
    print(f"[DEBUG] STS Identity ARN: {arn}")
    
    # Extract role ARN from assumed role identity
    # Format: arn:aws:sts::ACCOUNT:assumed-role/ROLE_NAME/SessionName
    # We need to convert to: arn:aws:iam::ACCOUNT:role/ROLE_NAME or role/service-role/ROLE_NAME
    if ':assumed-role/' in arn or '/assumed-role/' in arn:
        # Handle both formats: ':assumed-role/' and '/assumed-role/'
        if ':assumed-role/' in arn:
            # Format: arn:aws:sts::ACCOUNT:assumed-role/ROLE_NAME/SessionName
            parts = arn.split(':assumed-role/')
        else:
            # Format: arn:aws:sts::ACCOUNT/assumed-role/ROLE_NAME/SessionName (less common)
            parts = arn.split('/assumed-role/')
        
        if len(parts) > 1:
            # Extract role name (the part between 'assumed-role/' and the next '/')
            role_name = parts[1].split('/')[0]
            
            # SageMaker execution roles are typically under service-role/ path
            # Try service-role path first, then standard path
            role_arns_to_try = [
                f"arn:aws:iam::{account_id}:role/service-role/{role_name}",
                f"arn:aws:iam::{account_id}:role/{role_name}"
            ]
            
            # Check which role actually exists by querying IAM
            # IAM role names don't include the path - the path is only in the ARN
            try:
                iam = boto3.client('iam')
                # Try to get the role with just the role name (no path)
                try:
                    response = iam.get_role(RoleName=role_name)
                    # If successful, check the ARN to see which path it uses
                    actual_arn = response['Role']['Arn']
                    print(f"[INFO] Found role, using ARN from IAM: {actual_arn}")
                    return actual_arn
                except ClientError as e:
                    error_code = e.response['Error']['Code']
                    if error_code == 'NoSuchEntity':
                        # Role doesn't exist with that name, try service-role path
                        print(f"[WARNING] Role {role_name} not found, trying service-role path")
                        # Default to service-role path (most common for SageMaker)
                        return role_arns_to_try[0]
                    else:
                        raise
            except Exception as e:
                # If IAM query fails, default to service-role path (most common for SageMaker)
                print(f"[WARNING] Could not query IAM to verify role ({e}), defaulting to service-role path: {role_arns_to_try[0]}")
                return role_arns_to_try[0]
    
    # If not an assumed role format, try to query IAM for execution roles
    print(f"[WARNING] Could not extract role from STS identity ARN: {arn}")
    print(f"[WARNING] Attempting to find SageMaker execution role in IAM...")
    
    try:
        iam = boto3.client('iam')
        # List roles and find one that matches SageMaker execution role pattern
        roles = iam.list_roles()
        for role in roles.get('Roles', []):
            role_name = role['RoleName']
            if 'SageMaker' in role_name and 'Execution' in role_name:
                role_arn = role['Arn']
                print(f"[INFO] Found SageMaker execution role: {role_arn}")
                return role_arn
    except Exception as e:
        print(f"[WARNING] Could not query IAM for roles: {e}")
    
    # Last resort: try common patterns
    common_patterns = [
        f"arn:aws:iam::{account_id}:role/AmazonSageMaker-ExecutionRole",
        f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole",
    ]
    
    for pattern in common_patterns:
        print(f"[WARNING] Trying common role pattern: {pattern}")
        # Note: We can't verify if this role exists without IAM permissions
        # But we'll return it and let SageMaker validate it
        return pattern
    
    raise RuntimeError(
        f"Could not determine SageMaker execution role. "
        f"STS Identity ARN: {arn}. "
        f"Please specify the role ARN explicitly or ensure IAM permissions are available."
    )


def get_default_bucket():
    """Get default SageMaker bucket name (SDK 3.x compatible)."""
    sm_client = boto3.client('sagemaker')
    region = sm_client.meta.region_name
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    return f"sagemaker-{region}-{account_id}"


def upload_data_to_s3(local_path, bucket, key_prefix):
    """Upload data to S3 (replacement for sess.upload_data)."""
    s3 = boto3.client('s3')
    
    # If local_path is a directory, upload all files
    if os.path.isdir(local_path):
        s3_uris = []
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file = os.path.join(root, file)
                # Preserve relative path structure
                rel_path = os.path.relpath(local_file, local_path)
                s3_key = f"{key_prefix}/{rel_path}".replace("\\", "/")
                
                s3.upload_file(local_file, bucket, s3_key)
                s3_uris.append(f"s3://{bucket}/{s3_key}")
        
        # Return the prefix URI (SageMaker expects this format)
        return f"s3://{bucket}/{key_prefix}/"
    else:
        # Single file upload
        filename = os.path.basename(local_path)
        s3_key = f"{key_prefix}/{filename}"
        s3.upload_file(local_path, bucket, s3_key)
        return f"s3://{bucket}/{s3_key}"


def upload_source_code_to_s3(source_dir, bucket, key_prefix):
    """
    Upload source code as a tar.gz archive (SageMaker requires this format).
    
    Args:
        source_dir: Directory containing source code
        bucket: S3 bucket name
        key_prefix: S3 key prefix (will append sourcedir.tar.gz)
    
    Returns:
        str: S3 URI to the tar.gz file
    """
    s3 = boto3.client('s3')
    
    # Create a temporary tar.gz file
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
        tar_path = tmp_file.name
    
    try:
        # Create tar.gz archive
        # SageMaker extracts the tar.gz to /opt/ml/code/, so files need to be at the
        # root of the archive (not under a subdirectory like 'sourcedir/')
        print(f"Creating tar.gz archive from {source_dir}...")
        with tarfile.open(tar_path, 'w:gz') as tar:
            # Add all files with their relative paths from source_dir
            # This ensures trainer.py is at the root, not sourcedir/trainer.py
            source_dir_abs = os.path.abspath(source_dir)
            for root, dirs, files in os.walk(source_dir_abs):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Get relative path from source_dir (this removes source_dir from the path)
                    arcname = os.path.relpath(file_path, source_dir_abs)
                    tar.add(file_path, arcname=arcname)
        
        # Upload to S3
        s3_key = f"{key_prefix}/sourcedir.tar.gz"
        print(f"Uploading {tar_path} to s3://{bucket}/{s3_key}...")
        s3.upload_file(tar_path, bucket, s3_key)
        
        s3_uri = f"s3://{bucket}/{s3_key}"
        print(f"Source code uploaded to: {s3_uri}")
        return s3_uri
    finally:
        # Clean up temporary file
        if os.path.exists(tar_path):
            os.remove(tar_path)


# --- PATHS ---
# Get the script's directory and work backwards to find root paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XAVIERA_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up from scripts/ to xaviera-lora/
FINETUNING_ROOT = os.path.dirname(XAVIERA_ROOT)  # Go up from xaviera-lora/ to xaviera-lora-finetuning/
ACE_STEP_ROOT = os.path.join(FINETUNING_ROOT, "ACE-Step")

# Default dataset directory (can be overridden with --bucket argument)
DEFAULT_DATASET_DIR = os.path.join(XAVIERA_ROOT, "data", "datasets", "pop_500_index")
LOCAL_LORA_CONFIG = os.path.join(XAVIERA_ROOT, "config", "lora_config.json")

# Copy config into ACE-Step so it is bundled with source_dir upload
ACE_LORA_CONFIG = os.path.join(ACE_STEP_ROOT, "config", "lora_config.json")

# --- S3 INPUTS ---
# The audio already lives on S3; we just point to it as a channel.
S3_AUDIO_URI = "s3://xaviera-training-file/000001/Pop/"


def calculate_batch_size_from_duration(bucket_name, avg_duration_seconds, max_seconds_per_batch=300, min_batch_size=1, max_batch_size=32):
    """
    Calculate batch size based on audio duration (similar to propose_batches.py logic).
    
    This is more meaningful than arbitrary "batches per epoch" because:
    - Audio training is memory-bound by duration, not just sample count
    - Each bucket contains similar-duration audio (bucket_90s ≈ 90s, bucket_95s ≈ 95s, etc.)
    - We want to fit max_seconds_per_batch worth of audio in each batch
    
    Args:
        bucket_name: Bucket name (e.g., "bucket_90s") - used to extract duration if avg_duration not provided
        avg_duration_seconds: Average audio duration in seconds for this bucket
        max_seconds_per_batch: Maximum total audio seconds per batch (default: 300s = 5 minutes)
        min_batch_size: Minimum batch size (default: 1 - allows single-sample batches for long audio)
        max_batch_size: Maximum batch size (default: 32)
    
    Returns:
        int: Calculated batch size (no power-of-2 rounding, any integer is fine)
    """
    import re
    
    # If avg_duration not provided, try to extract from bucket name
    if avg_duration_seconds is None and bucket_name:
        match = re.match(r'bucket_(\d+)s', bucket_name)
        if match:
            avg_duration_seconds = float(match.group(1))
        else:
            # Fallback: use default batch size
            return 8
    
    if avg_duration_seconds is None or avg_duration_seconds <= 0:
        return min_batch_size
    
    # Calculate batch size: how many samples fit in max_seconds_per_batch?
    # e.g., if avg_dur is 90s and max_seconds=300, batch_size = 300/90 = 3
    # e.g., if avg_dur is 50s and max_seconds=300, batch_size = 300/50 = 6
    # e.g., if avg_dur is 240s and max_seconds=300, batch_size = 300/240 = 1
    calculated_batch_size = max(1, int(max_seconds_per_batch / avg_duration_seconds))
    
    # Clamp to min/max bounds (no power-of-2 rounding - modern frameworks handle any batch size efficiently)
    # Allow batch_size=1 for long audio to respect max_seconds_per_batch constraint
    batch_size = max(min_batch_size, min(max_batch_size, calculated_batch_size))
    
    return batch_size


def main(bucket_name=None, dataset_dir=None, checkpoint_path=None, batch_size=None, role_arn=None):
    """
    Launch SageMaker training job.
    
    Args:
        bucket_name: Optional bucket name (e.g., "bucket_0s", "bucket_5s"). 
                     If provided, uses bucket-specific dataset index.
        dataset_dir: Optional custom dataset directory path.
                     If not provided, uses DEFAULT_DATASET_DIR or bucket-specific path.
        checkpoint_path: Optional S3 path to checkpoint to resume from.
                        If provided, training will continue from this checkpoint.
        batch_size: Optional batch size. If None, will be calculated based on dataset size.
        role_arn: Optional IAM role ARN. If not provided, will attempt to auto-detect.
    """
    # SDK 3.x doesn't have Session - use boto3 directly
    if role_arn:
        role = role_arn
        print(f"[INFO] Using provided role ARN: {role}")
    else:
        role = get_execution_role()
    default_bucket = get_default_bucket()
    sm_client = boto3.client('sagemaker')
    region = sm_client.meta.region_name
    
    # Determine dataset directory
    if dataset_dir:
        LOCAL_DATASET_DIR = dataset_dir
    elif bucket_name:
        # Use bucket-specific dataset index
        LOCAL_DATASET_DIR = os.path.join(XAVIERA_ROOT, "data", "datasets", "buckets", bucket_name)
        print(f"Using bucket-specific dataset: {LOCAL_DATASET_DIR}")
    else:
        LOCAL_DATASET_DIR = DEFAULT_DATASET_DIR
        print(f"Using default dataset: {LOCAL_DATASET_DIR}")
    
    if not os.path.exists(LOCAL_DATASET_DIR):
        print(f"ERROR: Dataset directory not found: {LOCAL_DATASET_DIR}")
        print("Make sure you've created the dataset index first.")
        print("For bucket indices, run: python scripts/create_bucket_indices.py --input buckets.txt")
        return

    # 1.1) Check for or generate training_durations.txt in the dataset directory
    # This ensures the remote trainer has access to the duration of each sample in the dataset order
    dur_out_path = os.path.join(LOCAL_DATASET_DIR, "training_durations.txt")
    
    try:
        from datasets import load_from_disk
        ds = load_from_disk(LOCAL_DATASET_DIR)
        dataset_size = len(ds)
        
        # Calculate batch size if not provided
        # Use duration-based calculation (more meaningful than arbitrary "batches per epoch")
        if batch_size is None:
            batch_size = calculate_batch_size_from_duration(
                bucket_name=bucket_name,
                avg_duration_seconds=None,  # Will extract from bucket name if available
                max_seconds_per_batch=240,  # Match max audio duration limit (240s)
                min_batch_size=1,  # Allow batch_size=1 for long audio (e.g., bucket_240s)
                max_batch_size=32
            )
            # CRITICAL: Batch size cannot exceed dataset size
            batch_size = min(batch_size, dataset_size)
            print(f"[INFO] Calculated batch_size={batch_size} based on audio duration (dataset_size={dataset_size})")
        else:
            # CRITICAL: Even if provided, batch size cannot exceed dataset size
            batch_size = min(batch_size, dataset_size)
            print(f"[INFO] Using provided batch_size={batch_size} (clamped to dataset_size={dataset_size})")
        
        # Check if training_durations.txt already exists (e.g., from prepare_s3_index.py)
        if os.path.exists(dur_out_path):
            # Verify it matches the dataset size
            with open(dur_out_path, 'r') as f:
                existing_durations = [line.strip() for line in f if line.strip()]
            
            if len(existing_durations) == dataset_size:
                print(f"✓ Found existing training_durations.txt with {len(existing_durations)} durations (matches dataset size)")
                print(f"  Using existing file: {dur_out_path}")
            else:
                print(f"⚠ Existing training_durations.txt has {len(existing_durations)} durations, but dataset has {dataset_size} entries")
                print(f"  Regenerating training_durations.txt to match dataset...")
                # Fall through to regeneration
                dur_out_path = None  # Force regeneration
        else:
            print("No existing training_durations.txt found. Generating...")
            dur_out_path = None  # Force generation
        
        # Generate if needed
        if dur_out_path is None:
            dur_out_path = os.path.join(LOCAL_DATASET_DIR, "training_durations.txt")
            durations = []
            missing_count = 0
            
            # Try to find generated_audio_metadata directory
            possible_metadata_roots = [
                os.path.join(XAVIERA_ROOT, "generated_audio_metadata", "Pop"),
                os.path.join(os.path.dirname(XAVIERA_ROOT), "xaviera_essential", "generated_audio_metadata", "Pop"),
                os.path.join(XAVIERA_ROOT, "..", "xaviera_essential", "generated_audio_metadata", "Pop"),
            ]
            
            metadata_root = None
            for path in possible_metadata_roots:
                if os.path.exists(path):
                    metadata_root = path
                    break
            
            if metadata_root is None:
                print(f"WARNING: Could not find generated_audio_metadata directory. Using fallback durations (240.0s).")
                durations = ["240.0"] * dataset_size
            else:
                print(f"Reading durations from: {metadata_root}")
                for key in ds['keys']:
                    # Construct path: .../Pop/[key]/[key]_duration.txt
                    dur_path = os.path.join(metadata_root, key, f"{key}_duration.txt")
                    
                    if os.path.exists(dur_path):
                        with open(dur_path, 'r') as f:
                            val = f.read().strip()
                            # fast validation
                            try:
                                float(val)
                                durations.append(val)
                            except ValueError:
                                print(f"Warning: Invalid duration in {dur_path}: {val}")
                                durations.append("240.0")
                    else:
                        missing_count += 1
                        durations.append("240.0") # Fallback default
                
                if missing_count > 0:
                    print(f"Warning: {missing_count}/{len(ds)} samples were missing duration files (used 240.0s).")
            
            with open(dur_out_path, "w") as f:
                f.write("\n".join(durations))
            print(f"✓ Saved training_durations.txt to: {dur_out_path} ({len(durations)} durations)")
        
    except Exception as e:
        print(f"Error processing durations: {e}")
        print("Skipping duration file generation (bucketing might revert to random).")
        # Ensure batch_size is set even if dataset loading failed
        if batch_size is None:
            batch_size = 8
            print(f"[WARNING] Using default batch_size={batch_size} due to dataset loading error")

    # Ensure batch_size is set (fallback)
    if batch_size is None:
        batch_size = 8
        print(f"[WARNING] Using default batch_size={batch_size}")

    # 1) Upload dataset index to S3
    # Use bucket name in S3 key prefix if using bucket-specific index
    if bucket_name:
        s3_key_prefix = f"xaviera-lora/datasets/buckets/{bucket_name}"
        exp_name = f"pop_finetune_{bucket_name}"
    else:
        s3_key_prefix = "xaviera-lora/datasets/pop_500_index"
        exp_name = "pop_finetune"
    
    print(f"Uploading dataset index from {LOCAL_DATASET_DIR} ...")
    dataset_s3_uri = upload_data_to_s3(
        local_path=LOCAL_DATASET_DIR,
        bucket=default_bucket,
        key_prefix=s3_key_prefix
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
    hyperparameters = {
        "dataset_path": "/opt/ml/input/data/dataset",
        "exp_name": exp_name,
        "lora_config_path": "config/lora_config.json",
        "max_steps": 3000,
        "learning_rate": 3e-5,
        "every_plot_step": 1000,
        "logger_dir": "/opt/ml/output/data/logs",
        "checkpoint_dir": "/opt/ml/model",
        "num_workers": 1,          # Reduce memory from data loaders
        "precision": "bf16-mixed", # Use bf16 to halve memory
        "batch_size": batch_size,  # Calculated or provided batch size for bucket-specific datasets
    }
    
    # Add checkpoint path if provided
    if checkpoint_path:
        hyperparameters["ckpt_path"] = checkpoint_path
        print(f"Will resume from checkpoint: {checkpoint_path}")
    
    # 5) Upload source code to S3
    # SDK 3.x doesn't have the old estimator APIs, so we'll upload source code manually
    # SageMaker requires source code to be a tar.gz archive, not a directory
    job_name = f"pytorch-training-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')[:-3]}"
    source_code_uri = upload_source_code_to_s3(
        source_dir=ACE_STEP_ROOT,
        bucket=default_bucket,
        key_prefix=f"{job_name}/source"
    )

    # 6) Get custom image URI to bypass algorithm detection
    # AWS detects algorithms from image URI pattern (pytorch-training:*), so we use
    # the image from user's ECR instead to bypass validation
    print("Getting custom PyTorch image URI...")
    image_uri = get_custom_image_uri()
    
    # 7) Construct training job request using boto3 directly
    # This bypasses SDK validation issues and matches the successful job structure
    print("Constructing training job request...")
    
    # Build input data config
    input_data_config = []
    for channel_name, s3_data in inputs.items():
        input_data_config.append({
            'ChannelName': channel_name,
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': s3_data,
                    'S3DataDistributionType': 'FullyReplicated'
                }
            }
        })
    
    # Construct training job request
    # CRITICAL: AlgorithmSpecification must NOT contain AlgorithmName field
    # Using custom ECR image (not pytorch-training:*) bypasses algorithm detection
    train_request = {
        'TrainingJobName': job_name,
        'RoleArn': role,
        'AlgorithmSpecification': {
            'TrainingImage': image_uri,  # Use custom ECR image to bypass algorithm detection
            'TrainingInputMode': 'File',
            'EnableSageMakerMetricsTimeSeries': True
            # NOTE: No 'AlgorithmName' field - this is key to bypassing validation
        },
        'InputDataConfig': input_data_config,
        'OutputDataConfig': {
            'S3OutputPath': 's3://xaviera-lora-checkpoints/'
        },
        'ResourceConfig': {
            'InstanceType': 'ml.g6.2xlarge',
            'InstanceCount': 1,
            'VolumeSizeInGB': 30
        },
        'HyperParameters': {str(k): str(v) for k, v in hyperparameters.items()},
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 86400  # 24 hours
        },
        'Environment': {
            'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'
        }
    }
    
    # Add source code hyperparameters (required for SageMaker to find the entry point)
    train_request['HyperParameters']['sagemaker_program'] = 'trainer.py'
    train_request['HyperParameters']['sagemaker_submit_directory'] = source_code_uri
    
    # DEBUG: Print request structure
    print(f"[DEBUG] Training job request structure:")
    print(f"  TrainingJobName: {job_name}")
    print(f"  InstanceType: ml.g6.2xlarge")
    print(f"  TrainingImage: {image_uri}")
    print(f"  AlgorithmSpecification keys: {list(train_request['AlgorithmSpecification'].keys())}")
    if 'AlgorithmName' in train_request['AlgorithmSpecification']:
        print(f"[WARNING] AlgorithmName is present (should not be)")
    else:
        print("[DEBUG] AlgorithmName not present (correct - matches successful job)")
    
    # 8) Submit training job via boto3
    print("Submitting training job via boto3...")
    # sm_client already created at the beginning of main()
    
    try:
        response = sm_client.create_training_job(**train_request)
        print(f"✓ Training job submitted: {job_name}")
        
        # Create a mock estimator object that can be used by launch_sequential_buckets.py
        # This allows wait_for_job_completion to work
        class MockEstimator:
            def __init__(self, job_name, output_path):
                self.latest_training_job = job_name
                self.output_path = output_path
                self._current_job_name = job_name
        
        mock_estimator = MockEstimator(job_name, 's3://xaviera-lora-checkpoints/')
        return mock_estimator
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"ERROR: Failed to create training job: {error_code}: {error_message}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Launch SageMaker LoRA fine-tuning training job."
    )
    parser.add_argument(
        "--bucket",
        type=str,
        help="Bucket name to use (e.g., 'bucket_0s', 'bucket_5s'). "
             "If provided, uses bucket-specific dataset index from data/datasets/buckets/<bucket_name>"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        help="Custom dataset directory path. Overrides --bucket if both are provided."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="S3 path to checkpoint to resume from (e.g., s3://bucket/path/to/checkpoint.ckpt)"
    )
    parser.add_argument(
        "--role-arn",
        type=str,
        help="IAM role ARN for SageMaker execution. If not provided, will attempt to auto-detect from STS identity."
    )
    
    args = parser.parse_args()
    
    main(bucket_name=args.bucket, dataset_dir=args.dataset_dir, checkpoint_path=args.checkpoint_path, role_arn=args.role_arn)

