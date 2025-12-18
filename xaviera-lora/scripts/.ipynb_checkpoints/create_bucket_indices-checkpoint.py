"""
Create bucket-specific dataset indices from propose_batches.py output.

This script:
1. Parses the output from propose_batches.py (or reads from a file)
2. Extracts catalog IDs for each duration bucket
3. Creates separate dataset indices for each bucket using prepare_s3_index.py

Usage:
    # From propose_batches.py output (piped)
    python propose_batches.py | python create_bucket_indices.py
    
    # From a saved file
    python propose_batches.py > buckets.txt
    python create_bucket_indices.py --input buckets.txt
    
    # Specify output base directory
    python create_bucket_indices.py --input buckets.txt --output-base data/datasets/buckets
"""

import os
import re
import sys
import argparse
import subprocess
from pathlib import Path


# Get paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XAVIERA_LORA_ROOT = os.path.dirname(SCRIPT_DIR)
PREPARE_SCRIPT = os.path.join(SCRIPT_DIR, "prepare_s3_index.py")
DEFAULT_OUTPUT_BASE = os.path.join(XAVIERA_LORA_ROOT, "data", "datasets", "buckets")


def parse_bucket_output(text):
    """
    Parse output from propose_batches.py to extract bucket information.
    
    Format:
        Bucket (s)    | Count | BSz | Batches (Catalog IDs)
        --------------------------------------------------------------------
        0-4 s        | 5     | 60  | 606727 806272 , 1131440 629288 743703
    
    Returns:
        dict: {bucket_key: [list of catalog_id lists (batches)]}
        Example: {0: [['606727', '806272'], ['1131440', '629288', '743703']]}
    """
    buckets = {}
    lines = text.strip().split('\n')
    
    # Find the header line
    header_found = False
    for i, line in enumerate(lines):
        if 'Bucket (s)' in line or 'Bucket' in line:
            header_found = True
            continue
        
        if not header_found or not line.strip() or line.startswith('-'):
            continue
        
        # Parse bucket line
        # Format: "0-4 s        | 5     | 60  | 606727 806272 , 1131440 629288"
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 4:
            continue
        
        bucket_range = parts[0].strip()
        batches_str = parts[3].strip()
        
        # Extract bucket key (e.g., "0-4 s" -> 0)
        match = re.match(r'(\d+)-', bucket_range)
        if not match:
            continue
        bucket_key = int(match.group(1))
        
        # Parse batches (comma-separated, space-separated catalog IDs)
        batches = []
        for batch_str in batches_str.split(','):
            batch_str = batch_str.strip()
            if batch_str:
                # Split by space to get individual catalog IDs
                catalog_ids = [cid.strip() for cid in batch_str.split() if cid.strip()]
                if catalog_ids:
                    batches.append(catalog_ids)
        
        if batches:
            buckets[bucket_key] = batches
    
    return buckets


def create_bucket_index(bucket_key, catalog_ids, output_dir, metadata_dir):
    """
    Create a dataset index for a specific bucket.
    
    Args:
        bucket_key: Bucket identifier (e.g., 0 for 0-4s bucket)
        catalog_ids: List of catalog IDs to include in this bucket
        output_dir: Directory to save the dataset index
        metadata_dir: Directory containing pop_metadata
    
    Returns:
        str: Path to created dataset index, or None if failed
    """
    print(f"\n{'='*80}")
    print(f"Creating index for bucket {bucket_key}s (catalog IDs: {len(catalog_ids)})")
    print(f"{'='*80}")
    
    # Create output directory for this bucket
    bucket_output_dir = os.path.join(output_dir, f"bucket_{bucket_key}s")
    os.makedirs(bucket_output_dir, exist_ok=True)
    
    # Call prepare_s3_index.py with catalog IDs filter
    # Duration-based batching is enabled by default (groups similar durations together)
    cmd = [
        sys.executable,
        PREPARE_SCRIPT,
        "--catalog-ids"] + catalog_ids + [
        "--output-dir",
        bucket_output_dir,
        "--max-seconds-per-batch", "241"  # Enable duration-based batching
    ]
    
    print(f"Running: {' '.join(cmd[:5])} ... ({len(catalog_ids)} IDs)")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        
        print(f"✓ Successfully created index: {bucket_output_dir}")
        return bucket_output_dir
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error creating index for bucket {bucket_key}s:")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Error: {e.stderr}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Create bucket-specific dataset indices from propose_batches.py output."
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input file containing propose_batches.py output. "
             "If not provided, reads from stdin."
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default=DEFAULT_OUTPUT_BASE,
        help=f"Base directory for bucket indices (default: {DEFAULT_OUTPUT_BASE})"
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        help="Path to pop_metadata directory. "
             "If not provided, uses default from prepare_s3_index.py"
    )
    parser.add_argument(
        "--bucket",
        type=int,
        help="Create index for a specific bucket only (e.g., --bucket 0 for 0-4s bucket)"
    )
    
    args = parser.parse_args()
    
    # Read input
    if args.input:
        if not os.path.exists(args.input):
            print(f"ERROR: Input file not found: {args.input}")
            return
        with open(args.input, 'r') as f:
            input_text = f.read()
    else:
        # Read from stdin
        print("Reading from stdin... (paste propose_batches.py output and press Ctrl+D)")
        input_text = sys.stdin.read()
    
    if not input_text.strip():
        print("ERROR: No input provided.")
        return
    
    # Parse buckets
    print("Parsing bucket information...")
    buckets = parse_bucket_output(input_text)
    
    if not buckets:
        print("ERROR: Could not parse any buckets from input.")
        print("Make sure the input follows propose_batches.py output format.")
        return
    
    print(f"Found {len(buckets)} buckets:")
    for bucket_key in sorted(buckets.keys()):
        total_ids = sum(len(batch) for batch in buckets[bucket_key])
        print(f"  Bucket {bucket_key}s: {len(buckets[bucket_key])} batches, {total_ids} total catalog IDs")
    
    # Create output directory
    os.makedirs(args.output_base, exist_ok=True)
    
    # Create indices for each bucket
    created_indices = {}
    metadata_dir = args.metadata_dir if args.metadata_dir else None
    
    buckets_to_process = [args.bucket] if args.bucket is not None else sorted(buckets.keys())
    
    for bucket_key in buckets_to_process:
        if bucket_key not in buckets:
            print(f"WARNING: Bucket {bucket_key}s not found in parsed buckets.")
            continue
        
        # Flatten all batches into a single list of catalog IDs
        all_catalog_ids = []
        for batch in buckets[bucket_key]:
            all_catalog_ids.extend(batch)
        
        # Remove duplicates while preserving order
        catalog_ids = list(dict.fromkeys(all_catalog_ids))
        
        # Create index for this bucket
        index_path = create_bucket_index(
            bucket_key,
            catalog_ids,
            args.output_base,
            metadata_dir
        )
        
        if index_path:
            created_indices[bucket_key] = index_path
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Created {len(created_indices)} bucket indices:")
    for bucket_key in sorted(created_indices.keys()):
        print(f"  Bucket {bucket_key}s: {created_indices[bucket_key]}")
    
    print(f"\nTo use a bucket index for training:")
    print(f"  Update launch_training.py to use: {args.output_base}/bucket_<N>s")
    print(f"  Or modify dataset_path in training hyperparameters.")


if __name__ == "__main__":
    main()

