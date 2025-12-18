"""
Launch sequential training across all buckets, where each bucket continues
from the previous bucket's checkpoint.

Usage:
    python xaviera-lora/scripts/launch_sequential_buckets.py
    
    # Start from a specific bucket
    python xaviera-lora/scripts/launch_sequential_buckets.py --start-bucket bucket_90s
    
    # Start from a specific checkpoint
    python xaviera-lora/scripts/launch_sequential_buckets.py --checkpoint-path s3://bucket/path/to/checkpoint
"""

import os
import re
import sys
import time
import argparse
import subprocess
import boto3
import threading
from pathlib import Path
from datetime import datetime, timedelta

# Get paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XAVIERA_LORA_ROOT = os.path.dirname(SCRIPT_DIR)
BUCKETS_DIR = os.path.join(XAVIERA_LORA_ROOT, "data", "datasets", "buckets")


def discover_buckets(buckets_dir):
    """
    Discover all bucket directories and sort them by duration.
    
    Returns:
        list: Sorted list of bucket names (e.g., ['bucket_90s', 'bucket_95s', ...])
    """
    if not os.path.exists(buckets_dir):
        print(f"ERROR: Buckets directory not found: {buckets_dir}")
        return []
    
    buckets = []
    for item in os.listdir(buckets_dir):
        bucket_path = os.path.join(buckets_dir, item)
        if os.path.isdir(bucket_path) and item.startswith("bucket_") and item.endswith("s"):
            buckets.append(item)
    
    # Sort by duration (extract number from bucket name)
    def extract_duration(bucket_name):
        match = re.match(r'bucket_(\d+)s', bucket_name)
        return int(match.group(1)) if match else 0
    
    buckets.sort(key=extract_duration)
    return buckets


def get_bucket_dataset_size(bucket_path):
    """
    Get the dataset size for a bucket.
    
    Args:
        bucket_path: Path to bucket dataset directory
    
    Returns:
        int: Number of samples in the dataset, or 0 if error
    """
    try:
        from datasets import load_from_disk
        if os.path.exists(bucket_path):
            ds = load_from_disk(bucket_path)
            return len(ds)
    except Exception as e:
        print(f"WARNING: Could not load dataset from {bucket_path}: {e}")
    return 0


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
    # If avg_duration not provided, try to extract from bucket name
    if avg_duration_seconds is None:
        match = re.match(r'bucket_(\d+)s', bucket_name)
        if match:
            avg_duration_seconds = float(match.group(1))
        else:
            # Fallback: use default batch size
            return 8
    
    if avg_duration_seconds <= 0:
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


def get_checkpoint_path_from_job(job_name, output_bucket="s3://xaviera-lora-checkpoints"):
    """
    Construct the checkpoint path from a SageMaker training job name.
    
    The LoRA adapter is saved to /opt/ml/model during training, which SageMaker
    uploads to: s3://output-bucket/job-name/output/model.tar.gz
    
    After extraction, the LoRA adapter files are inside model.tar.gz.
    
    Args:
        job_name: SageMaker training job name
        output_bucket: S3 bucket where outputs are saved
    
    Returns:
        str: S3 path to model.tar.gz containing LoRA adapter
    """
    # SageMaker saves model artifacts to output/model.tar.gz
    checkpoint_path = f"{output_bucket}/{job_name}/output/model.tar.gz"
    return checkpoint_path


def stream_training_logs(job_name, stop_event):
    """
    Stream CloudWatch logs from a SageMaker training job.
    
    Args:
        job_name: Training job name
        stop_event: threading.Event to signal when to stop streaming
    """
    logs_client = boto3.client('logs')
    log_group = f'/aws/sagemaker/TrainingJobs'
    
    # Wait a bit for logs to start appearing (SageMaker takes time to initialize)
    print(f"[LOG] Waiting for CloudWatch logs to appear (this may take 1-2 minutes)...")
    time.sleep(30)  # Increased wait time
    
    start_time = int((datetime.utcnow() - timedelta(minutes=10)).timestamp() * 1000)
    seen_streams = set()
    no_logs_count = 0
    
    while not stop_event.is_set():
        try:
            # List all log streams for this job (can't use orderBy with prefix)
            streams_response = logs_client.describe_log_streams(
                logGroupName=log_group,
                logStreamNamePrefix=f'{job_name}/'
            )
            
            log_streams = streams_response.get('logStreams', [])
            
            if not log_streams:
                no_logs_count += 1
                if no_logs_count % 10 == 0:  # Print message every 20 seconds (10 * 2s)
                    print(f"[LOG] Still waiting for log streams to appear... (job may still be initializing)")
                time.sleep(2)
                continue
            
            # Reset counter if we found streams
            no_logs_count = 0
            
            # Process each log stream (usually algo-1, but could be multiple)
            for stream_info in log_streams:
                log_stream = stream_info['logStreamName']
                
                # Get log events from this stream
                try:
                    response = logs_client.get_log_events(
                        logGroupName=log_group,
                        logStreamName=log_stream,
                        startTime=start_time
                    )
                    
                    events = response.get('events', [])
                    for event in events:
                        # Print the log message
                        message = event['message'].rstrip()
                        if message:  # Only print non-empty messages
                            print(message)
                        # Update start_time to avoid reprinting old logs
                        start_time = max(start_time, event['timestamp'] + 1)
                    
                    seen_streams.add(log_stream)
                except Exception as e:
                    # Skip this stream if there's an error
                    continue
            
            # Wait before checking for new logs
            time.sleep(2)
            
        except logs_client.exceptions.ResourceNotFoundException:
            # Log group doesn't exist yet, wait and retry
            print(f"[LOG] Log group not found yet, waiting...")
            time.sleep(5)
        except Exception as e:
            # If there's an error, just wait and retry
            # Don't spam errors if logs aren't available yet
            if no_logs_count % 20 == 0:  # Print error every 40 seconds
                print(f"[LOG] Error accessing logs (may still be initializing): {type(e).__name__}")
            time.sleep(2)


def wait_for_job_completion(estimator):
    """
    Wait for a SageMaker training job to complete while streaming logs.
    
    Args:
        estimator: Mock estimator object with latest_training_job as a string (job name)
    
    Returns:
        str: Job name
        str: Checkpoint path (S3 URI)
    """
    # latest_training_job is now just a string (job name), not an object
    job_name = estimator.latest_training_job
    print(f"\nWaiting for training job to complete...")
    print(f"Job name: {job_name}")
    print(f"Streaming training logs (epochs, batches, etc.)...")
    print(f"\n💡 To manually check logs, run:")
    print(f"   aws logs tail /aws/sagemaker/TrainingJobs --follow --filter-pattern '{job_name}' --region us-east-1")
    print(f"   Or view in AWS Console: https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FTrainingJobs")
    print()
    
    # Start log streaming in a separate thread
    stop_logs = threading.Event()
    log_thread = threading.Thread(
        target=stream_training_logs,
        args=(job_name, stop_logs),
        daemon=True
    )
    log_thread.start()
    
    try:
        # Use boto3 to wait for job completion
        sm_client = boto3.client('sagemaker')
        waiter = sm_client.get_waiter('training_job_completed_or_stopped')
        waiter.wait(TrainingJobName=job_name)
    finally:
        # Stop log streaming
        stop_logs.set()
        log_thread.join(timeout=5)
    
    # Get job status
    response = sm_client.describe_training_job(TrainingJobName=job_name)
    status = response['TrainingJobStatus']
    
    if status == 'Completed':
        print(f"\n✓ Training job completed successfully!")
        checkpoint_path = get_checkpoint_path_from_job(job_name)
        print(f"Checkpoint saved to: {checkpoint_path}")
        return job_name, checkpoint_path
    else:
        print(f"\n✗ Training job failed with status: {status}")
        return None, None


def main(start_bucket=None, checkpoint_path=None, max_steps_per_bucket=3000):
    """
    Launch sequential training across all buckets.
    
    Args:
        start_bucket: Optional bucket name to start from (e.g., 'bucket_90s').
                     If None, starts from the first bucket.
        checkpoint_path: Optional S3 path to initial checkpoint.
                        If None and start_bucket is not the first bucket,
                        will try to find checkpoint from previous bucket.
        max_steps_per_bucket: Number of training steps per bucket.
    """
    # Discover all buckets
    buckets = discover_buckets(BUCKETS_DIR)
    
    if not buckets:
        print("ERROR: No buckets found. Make sure you've created bucket indices first:")
        print("  python xaviera-lora/scripts/create_bucket_indices.py --input buckets.txt")
        return
    
    print(f"Found {len(buckets)} buckets: {', '.join(buckets)}")
    
    # Find starting point
    start_index = 0
    if start_bucket:
        if start_bucket in buckets:
            start_index = buckets.index(start_bucket)
            print(f"Starting from bucket: {start_bucket} (index {start_index})")
        else:
            print(f"WARNING: Start bucket '{start_bucket}' not found. Starting from first bucket.")
    
    # Process buckets sequentially
    current_checkpoint = checkpoint_path
    buckets_to_process = buckets[start_index:]
    
    print(f"\n{'='*80}")
    print(f"SEQUENTIAL BUCKET TRAINING")
    print(f"{'='*80}")
    print(f"Total buckets to process: {len(buckets_to_process)}")
    if current_checkpoint:
        print(f"Initial checkpoint: {current_checkpoint}")
    print(f"{'='*80}\n")
    
    for i, bucket_name in enumerate(buckets_to_process):
        bucket_num = start_index + i + 1
        print(f"\n{'='*80}")
        print(f"BUCKET {bucket_num}/{len(buckets)}: {bucket_name}")
        print(f"{'='*80}")
        print(f"\n🔄 BUCKET CHANGE: Starting training on bucket '{bucket_name}'")
        print(f"   This bucket contains audio samples with duration ~{bucket_name.replace('bucket_', '').replace('s', '')} seconds")
        print(f"   Bucket {bucket_num} of {len(buckets)} total buckets\n")
        
        # Get bucket dataset size and calculate batch size based on audio duration
        bucket_path = os.path.join(BUCKETS_DIR, bucket_name)
        dataset_size = get_bucket_dataset_size(bucket_path)
        
        # Calculate batch size based on audio duration (not arbitrary "batches per epoch")
        # Each bucket name indicates duration: bucket_90s ≈ 90s audio, bucket_95s ≈ 95s, etc.
        calculated_batch_size = calculate_batch_size_from_duration(
            bucket_name=bucket_name,
            avg_duration_seconds=None,  # Will extract from bucket name
            max_seconds_per_batch=240,  # Match max audio duration limit (240s)
            min_batch_size=1,  # Allow batch_size=1 for long audio (e.g., bucket_240s)
            max_batch_size=32
        )
        
        # CRITICAL: Batch size cannot exceed dataset size
        if dataset_size > 0:
            calculated_batch_size = min(calculated_batch_size, dataset_size)
            print(f"[INFO] Bucket dataset size: {dataset_size} samples")
            print(f"[INFO] Calculated batch size: {calculated_batch_size} (based on audio duration, clamped to dataset size)")
        else:
            print(f"[WARNING] Could not determine dataset size for {bucket_name}, using calculated batch_size={calculated_batch_size}")
        
        # Import launch_training and call it directly
        # This is cleaner than subprocess since we need the estimator object
        sys.path.insert(0, SCRIPT_DIR)
        from launch_training import main as launch_main
        
        try:
            # Launch training job with calculated batch size
            estimator = launch_main(
                bucket_name=bucket_name,
                dataset_dir=None,
                checkpoint_path=current_checkpoint,
                batch_size=calculated_batch_size
            )
            
            # Wait for job to complete and get checkpoint
            job_name, next_checkpoint = wait_for_job_completion(estimator)
            
            if job_name and next_checkpoint:
                current_checkpoint = next_checkpoint
                print(f"\n✓ Bucket {bucket_name} completed. Checkpoint: {current_checkpoint}")
            else:
                print(f"\n✗ Bucket {bucket_name} failed. Stopping sequential training.")
                return
                
        except KeyboardInterrupt:
            print(f"\n\nTraining interrupted by user.")
            print(f"Last checkpoint: {current_checkpoint}")
            print(f"To resume, run:")
            print(f"  python {__file__} --start-bucket {bucket_name} --checkpoint-path {current_checkpoint}")
            return
        except Exception as e:
            print(f"\n✗ Error training bucket {bucket_name}: {e}")
            print(f"Stopping sequential training.")
            return
        
        # Small delay between jobs
        if i < len(buckets_to_process) - 1:
            print(f"\nWaiting 10 seconds before starting next bucket...")
            time.sleep(10)
    
    print(f"\n{'='*80}")
    print(f"✓ ALL BUCKETS COMPLETED!")
    print(f"{'='*80}")
    print(f"Final checkpoint: {current_checkpoint}")
    print(f"Total buckets trained: {len(buckets_to_process)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch sequential training across all buckets."
    )
    parser.add_argument(
        "--start-bucket",
        type=str,
        help="Bucket name to start from (e.g., 'bucket_90s'). If not provided, starts from first bucket."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="S3 path to initial checkpoint to resume from."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3000,
        help="Maximum training steps per bucket (default: 3000)"
    )
    
    args = parser.parse_args()
    
    main(
        start_bucket=args.start_bucket,
        checkpoint_path=args.checkpoint_path,
        max_steps_per_bucket=args.max_steps
    )

