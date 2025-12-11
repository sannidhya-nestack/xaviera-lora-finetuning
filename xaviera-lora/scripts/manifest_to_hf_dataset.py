"""
Build a HuggingFace dataset from a prepared CSV without copying audio.

Input CSV columns (from select_subset.py):
- keys
- filename  (absolute path to existing mp3)
- tags      (pipe-separated string)
- norm_lyrics

Usage:
python manifest_to_hf_dataset.py \
    --prepared_csv /mnt/c/.../xaviera-lora/manifests/prepared_manifest.csv \
    --output_dir   /mnt/c/.../xaviera-lora/data/datasets/alternative_subset_10
"""

import argparse
import json
import os
from datasets import Dataset
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--repeat_count", type=int, default=1, help="Repeat each example N times (default: 1)")
    args = parser.parse_args()

    df = pd.read_csv(args.prepared_csv)
    examples = []
    for _, row in df.iterrows():
        tags_list = []
        if isinstance(row.get("tags"), str) and row["tags"].strip():
            tags_list = [t.strip() for t in row["tags"].split("|") if t.strip()]

        example = {
            "keys": str(row["keys"]),
            "filename": str(row["filename"]),
            "tags": tags_list,
            "speaker_emb_path": "",
            "norm_lyrics": str(row.get("norm_lyrics", "") or ""),
            "recaption": {},
        }
        
        # Repeat the example repeat_count times
        for _ in range(args.repeat_count):
            examples.append(example)

    ds = Dataset.from_list(examples)
    os.makedirs(args.output_dir, exist_ok=True)
    ds.save_to_disk(args.output_dir)
    print(json.dumps({
        "num_songs": len(df),
        "repeat_count": args.repeat_count,
        "total_examples": len(examples),
        "saved_to": args.output_dir
    }))


if __name__ == "__main__":
    main()


