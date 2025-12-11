"""
Build a Hugging Face dataset index for Pop songs directly from S3 without
downloading audio locally. The resulting dataset folder is saved under
`xaviera-lora/data/datasets/pop_500_index`.
"""

import os
import boto3
from datasets import Dataset


# --- CONFIG ---
# Source audio location (existing on S3)
BUCKET_NAME = "xaviera-training-file"
PREFIX = "000001/Pop/"

# Where to save the Hugging Face dataset on the notebook filesystem
OUTPUT_DIR = "/home/ec2-user/SageMaker/xaviera-lora/data/datasets/pop_500_index"


def main():
    print(f"Scanning s3://{BUCKET_NAME}/{PREFIX} ...")
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=PREFIX)

    audio_files = []
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith((".mp3", ".wav", ".flac")):
                filename = os.path.basename(key)
                # Path inside the SageMaker training container after download
                container_path = f"/opt/ml/input/data/audio/{filename}"
                audio_files.append(container_path)

    print(f"Found {len(audio_files)} audio files.")

    data = []
    for i, filepath in enumerate(audio_files):
        data.append(
            {
                "keys": str(i),
                "filename": filepath,
                "tags": ["Pop"],
                "norm_lyrics": "",
                "recaption": {},
            }
        )

    print(f"Saving dataset index to {OUTPUT_DIR} ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset = Dataset.from_list(data)
    dataset.save_to_disk(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()

