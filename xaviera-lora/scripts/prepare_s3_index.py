"""
Build a Hugging Face dataset index for Pop songs by reading metadata from
local pop_metadata/[catalog_id]/ directories and mapping to S3 audio files.

Data Sources:
- Metadata: /home/ec2-user/SageMaker/xaviera-lora/pop_metadata/[catalog_id]/
  - [catalog_id]_prompt.txt  (comma-separated tags)
  - [catalog_id]_lyrics.txt  (song lyrics)
- Audio: s3://xaviera-training-file/000001/Pop/[catalog_id].wav

The resulting dataset folder is saved under:
    xaviera-lora/data/datasets/pop_500_index
"""

import os
from datasets import Dataset


# --- CONFIG ---
# Get paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XAVIERA_LORA_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up from scripts/ to xaviera-lora/

# Local metadata location on SageMaker notebook
METADATA_DIR = os.path.join(XAVIERA_LORA_ROOT, "pop_metadata")

# Where to save the Hugging Face dataset on the notebook filesystem
OUTPUT_DIR = os.path.join(XAVIERA_LORA_ROOT, "data", "datasets", "pop_500_index")


def read_file_content(filepath):
    """Read file content, return empty string if file doesn't exist."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def parse_prompt_to_tags(prompt_text):
    """
    Parse comma-separated prompt text into a list of tags.
    Example: "pop, female vocal, piano" -> ["pop", "female vocal", "piano"]
    """
    if not prompt_text:
        return []
    tags = [tag.strip() for tag in prompt_text.split(",") if tag.strip()]
    return tags


def get_duration_for_catalog_id(catalog_id, metadata_root):
    """
    Get duration for a catalog ID from metadata directory.
    
    Returns:
        float: Duration in seconds, or None if not found
    """
    if metadata_root is None:
        return None
    
    dur_path = os.path.join(metadata_root, catalog_id, f"{catalog_id}_duration.txt")
    if os.path.exists(dur_path):
        try:
            with open(dur_path, 'r') as f:
                val = f.read().strip()
                return float(val)
        except (ValueError, IOError):
            return None
    return None


def find_metadata_root():
    """Try to find generated_audio_metadata directory."""
    possible_paths = [
        os.path.join(XAVIERA_LORA_ROOT, "generated_audio_metadata", "Pop"),
        os.path.join(os.path.dirname(XAVIERA_LORA_ROOT), "xaviera_essential", "generated_audio_metadata", "Pop"),
        os.path.join(XAVIERA_LORA_ROOT, "..", "xaviera_essential", "generated_audio_metadata", "Pop"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def group_by_similar_duration(catalog_ids_with_durations, duration_tolerance=2.0):
    """
    Group catalog IDs by exact or very similar duration.
    
    Args:
        catalog_ids_with_durations: List of (catalog_id, duration) tuples
        duration_tolerance: Maximum difference in seconds to consider durations "similar" (default: 2.0s)
    
    Returns:
        dict: {rounded_duration: [list of (catalog_id, duration) tuples]}
    """
    # Sort by duration first
    sorted_items = sorted(catalog_ids_with_durations, key=lambda x: x[1])
    
    groups = {}
    for catalog_id, duration in sorted_items:
        # Round to nearest 2s for grouping (or use exact match)
        # This groups items with similar durations together (within 2 seconds)
        rounded_dur = round(duration / duration_tolerance) * duration_tolerance  # Round to 2s precision
        
        if rounded_dur not in groups:
            groups[rounded_dur] = []
        groups[rounded_dur].append((catalog_id, duration))
    
    return groups


def create_batches_from_groups(duration_groups, max_seconds_per_batch=300):
    """
    Create batches from duration groups, ensuring similar-duration items are together.
    
    Args:
        duration_groups: dict from group_by_similar_duration
        max_seconds_per_batch: Maximum total audio seconds per batch
    
    Returns:
        list: Ordered list of catalog_ids, grouped into batches (batches are contiguous)
    """
    ordered_catalog_ids = []
    
    # Process each duration group
    for rounded_dur in sorted(duration_groups.keys()):
        items = duration_groups[rounded_dur]
        
        # Calculate batch size for this duration group
        if items:
            avg_dur = sum(dur for _, dur in items) / len(items)
            batch_size = max(1, int(max_seconds_per_batch / avg_dur))
        else:
            batch_size = 1
        
        # Create batches from this group
        catalog_ids = [cat_id for cat_id, _ in items]
        for i in range(0, len(catalog_ids), batch_size):
            batch = catalog_ids[i:i + batch_size]
            ordered_catalog_ids.extend(batch)
    
    return ordered_catalog_ids


def generate_training_durations(dataset, output_dir, metadata_root=None):
    """
    Generate training_durations.txt file for a dataset.
    
    Args:
        dataset: Hugging Face dataset with 'keys' field
        output_dir: Directory where to save training_durations.txt
        metadata_root: Root directory containing duration files. 
                      If None, tries to auto-detect.
    """
    if metadata_root is None:
        metadata_root = find_metadata_root()
        
        if metadata_root is None:
            print("WARNING: Could not find generated_audio_metadata directory. Skipping training_durations.txt generation.")
            return
    
    print(f"\nGenerating training_durations.txt from {metadata_root}...")
    durations = []
    missing_count = 0
    
    for key in dataset['keys']:
        dur = get_duration_for_catalog_id(key, metadata_root)
        if dur is not None:
            durations.append(str(dur))
        else:
            missing_count += 1
            durations.append("240.0")  # Fallback default
    
    if missing_count > 0:
        print(f"Warning: {missing_count}/{len(dataset)} samples were missing duration files (used 240.0s).")
    
    dur_out_path = os.path.join(output_dir, "training_durations.txt")
    with open(dur_out_path, "w") as f:
        f.write("\n".join(durations))
    print(f"Saved training_durations.txt to: {dur_out_path} ({len(durations)} durations)")


def main(catalog_ids_filter=None, output_dir=None, generate_durations=True, max_seconds_per_batch=300, batch_by_duration=True):
    """
    Build dataset index for specified catalog IDs.
    
    Args:
        catalog_ids_filter: Optional set/list of catalog IDs to include. 
                           If None, includes all catalog IDs found in METADATA_DIR.
        output_dir: Optional output directory. If None, uses default OUTPUT_DIR.
        generate_durations: Whether to generate training_durations.txt file (default: True).
        max_seconds_per_batch: Maximum total audio seconds per batch for batching (default: 300).
        batch_by_duration: If True, groups similar-duration items together to minimize padding (default: True).
    """
    # Use provided output_dir or fall back to default
    final_output_dir = output_dir if output_dir else OUTPUT_DIR
    print(f"Scanning metadata directory: {METADATA_DIR}")
    
    if not os.path.exists(METADATA_DIR):
        print(f"ERROR: Metadata directory not found: {METADATA_DIR}")
        return
    
    # Get all catalog_id directories (exclude hidden directories like .ipynb_checkpoints)
    all_catalog_dirs = [
        d for d in os.listdir(METADATA_DIR)
        if os.path.isdir(os.path.join(METADATA_DIR, d)) and not d.startswith(".")
    ]
    
    # Filter to requested catalog IDs if provided
    if catalog_ids_filter is not None:
        # Convert to set for fast lookup
        if isinstance(catalog_ids_filter, list):
            catalog_ids_filter = set(catalog_ids_filter)
        elif not isinstance(catalog_ids_filter, set):
            catalog_ids_filter = set(str(catalog_ids_filter))
        
        catalog_dirs = [d for d in all_catalog_dirs if d in catalog_ids_filter]
        print(f"Found {len(all_catalog_dirs)} total catalog directories.")
        print(f"Filtering to {len(catalog_ids_filter)} requested catalog IDs.")
        print(f"Matched {len(catalog_dirs)} catalog IDs in metadata directory.")
        
        # Warn about missing catalog IDs
        missing = catalog_ids_filter - set(catalog_dirs)
        if missing:
            print(f"WARNING: {len(missing)} requested catalog IDs not found in metadata:")
            print(f"  Missing: {sorted(list(missing))[:10]}{'...' if len(missing) > 10 else ''}")
    else:
        catalog_dirs = all_catalog_dirs
        print(f"Found {len(catalog_dirs)} catalog directories (no filter applied).")
    
    # If batch_by_duration is enabled, we need to load durations first to order catalog IDs
    catalog_ids_with_durations = []
    metadata_root = find_metadata_root() if batch_by_duration else None
    
    if batch_by_duration and metadata_root:
        print(f"\nLoading durations for batching from: {metadata_root}")
        for catalog_id in catalog_dirs:
            dur = get_duration_for_catalog_id(catalog_id, metadata_root)
            if dur is not None:
                catalog_ids_with_durations.append((catalog_id, dur))
            else:
                # If duration not found, use fallback (will be sorted last)
                catalog_ids_with_durations.append((catalog_id, 240.0))
        
        # Group by similar duration and create batches
        print(f"Grouping {len(catalog_ids_with_durations)} catalog IDs by similar duration...")
        duration_groups = group_by_similar_duration(catalog_ids_with_durations, duration_tolerance=2.0)
        print(f"Created {len(duration_groups)} duration groups")
        
        # Create batches from groups
        ordered_catalog_ids = create_batches_from_groups(duration_groups, max_seconds_per_batch)
        print(f"Ordered {len(ordered_catalog_ids)} catalog IDs into batches (similar durations grouped together)")
        
        # Use ordered catalog IDs for processing
        catalog_dirs_to_process = ordered_catalog_ids
    else:
        # No batching - use sorted order
        catalog_dirs_to_process = sorted(catalog_dirs)
        if batch_by_duration:
            print("WARNING: batch_by_duration=True but metadata_root not found. Using sorted order instead.")
    
    data = []
    missing_prompt = 0
    missing_lyrics = 0
    
    print(f"\nProcessing {len(catalog_dirs_to_process)} catalog directories...")
    if catalog_ids_filter:
        print(f"Expected catalog IDs: {sorted(list(catalog_ids_filter))[:10]}{'...' if len(catalog_ids_filter) > 10 else ''}")
        print(f"Catalog directories to process: {len(catalog_dirs_to_process)} items")
    
    for catalog_id in catalog_dirs_to_process:
        catalog_path = os.path.join(METADATA_DIR, catalog_id)
        
        # Expected file paths
        prompt_file = os.path.join(catalog_path, f"{catalog_id}_prompt.txt")
        lyrics_file = os.path.join(catalog_path, f"{catalog_id}_lyrics.txt")
        
        # Read prompt and lyrics
        prompt_text = read_file_content(prompt_file)
        lyrics_text = read_file_content(lyrics_file)
        
        if not prompt_text:
            missing_prompt += 1
        if not lyrics_text:
            missing_lyrics += 1
        
        # Parse prompt into tags
        tags = parse_prompt_to_tags(prompt_text)
        if not tags:
            tags = ["Pop"]  # Fallback if no prompt found
        
        # Path inside the SageMaker training container after download
        # Audio files are [catalog_id].wav on S3
        container_audio_path = f"/opt/ml/input/data/audio/{catalog_id}.wav"
        
        data.append({
            "keys": catalog_id,
            "filename": container_audio_path,
            "tags": tags,
            "speaker_emb_path": "",
            "norm_lyrics": lyrics_text,
            "recaption": {},
        })
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(data)}")
    print(f"  Missing prompt files: {missing_prompt}")
    print(f"  Missing lyrics files: {missing_lyrics}")
    
    if len(data) == 0:
        print("ERROR: No valid samples found. Check your metadata directory structure.")
        return
    
    # Show a sample entry
    print(f"\nSample entry (first item):")
    sample = data[0]
    print(f"  keys: {sample['keys']}")
    print(f"  filename: {sample['filename']}")
    print(f"  tags: {sample['tags'][:5]}{'...' if len(sample['tags']) > 5 else ''}")
    print(f"  lyrics: {sample['norm_lyrics'][:100]}..." if sample['norm_lyrics'] else "  lyrics: (empty)")
    
    # Save dataset
    print(f"\nSaving dataset index to {final_output_dir} ...")
    os.makedirs(final_output_dir, exist_ok=True)
    dataset = Dataset.from_list(data)
    dataset.save_to_disk(final_output_dir)
    print("Done.")
    
    # Generate training_durations.txt if requested
    if generate_durations:
        generate_training_durations(dataset, final_output_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build Hugging Face dataset index for Pop songs."
    )
    parser.add_argument(
        "--catalog-ids",
        nargs="+",
        help="Optional: List of catalog IDs to include (space-separated). "
             "If not provided, includes all catalog IDs found in metadata directory."
    )
    parser.add_argument(
        "--catalog-ids-file",
        help="Optional: Path to file containing catalog IDs (one per line or space/comma-separated)."
    )
    parser.add_argument(
        "--output-dir",
        help="Optional: Override default output directory."
    )
    parser.add_argument(
        "--max-seconds-per-batch",
        type=int,
        default=300,
        help="Maximum total audio seconds per batch for duration-based batching (default: 300)"
    )
    parser.add_argument(
        "--no-batch-by-duration",
        action="store_true",
        help="Disable duration-based batching (use sorted order instead)"
    )
    
    args = parser.parse_args()
    
    # Determine catalog IDs filter
    catalog_ids_filter = None
    if args.catalog_ids:
        catalog_ids_filter = set(args.catalog_ids)
    elif args.catalog_ids_file:
        with open(args.catalog_ids_file, 'r') as f:
            content = f.read().strip()
            # Support multiple formats: one per line, space-separated, or comma-separated
            ids = []
            for line in content.split('\n'):
                ids.extend(line.replace(',', ' ').split())
            catalog_ids_filter = set([id.strip() for id in ids if id.strip()])
        print(f"Loaded {len(catalog_ids_filter)} catalog IDs from {args.catalog_ids_file}")
    
    # Pass output directory and batching options to main function
    main(
        catalog_ids_filter=catalog_ids_filter, 
        output_dir=args.output_dir,
        max_seconds_per_batch=args.max_seconds_per_batch,
        batch_by_duration=not args.no_batch_by_duration
    )
