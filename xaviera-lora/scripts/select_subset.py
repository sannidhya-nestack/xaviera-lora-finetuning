"""
 Select a subset of CatalogIDs from a manifest and prepare a minimal manifest
 that points directly to audio files on the E: drive (no copying).

 Outputs a CSV with columns compatible with our HF dataset builder:
   - keys (string CatalogID)
   - filename (absolute path to mp3 on /mnt/e/...)
   - tags (pipe-separated string; will be split to list later)
   - norm_lyrics (string)

 Usage:
   python select_subset.py \
     --manifest /mnt/c/.../xaviera_training/analysis_outputs/available_manifest_genre_Alternative_vocal_high_with_lyrics.csv \
     --audio_dir /mnt/e/Xaviera/AudioSparx_Training_Data \
     --output_csv /mnt/c/.../xaviera-lora/manifests/prepared_manifest.csv \
     --number_of_songs 10
"""
import argparse
import os
import sys
import pandas as pd


def first_nonempty(*values: str) -> str:
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def build_tags(row: pd.Series) -> list:
    tags = []
    # Core musical attributes
    for col in ["Genre", "Subgenre", "VocalType"]:
        val = str(row.get(col, "") or "").strip()
        if val:
            tags.append(val)

    # BPM preference: ai_BpmValue else BPM
    bpm_ai = str(row.get("ai_BpmValue", "") or "").strip()
    bpm = str(row.get("BPM", "") or "").strip()
    if bpm_ai:
        tags.append(f"{bpm_ai} bpm")
    elif bpm:
        tags.append(f"{bpm} bpm")

    # Musical key
    key = str(row.get("ai_MusicalKey", "") or "").strip()
    if key:
        tags.append(key)

    # Instruments, Moods
    for col in ["Instruments", "Moods"]:
        val = str(row.get(col, "") or "").strip()
        if val:
            parts = val.replace("|", ",").split(",")
            tags.extend([p.strip() for p in parts if p.strip()])

    # AI tags
    for col in ["ai_InstrumentTags", "ai_MoodAdvancedTags", "ai_GenreTags"]:
        val = str(row.get(col, "") or "").strip()
        if val:
            parts = val.replace("|", ",").split(",")
            tags.extend([p.strip() for p in parts if p.strip()])

    # Deduplicate, preserve order
    seen = set()
    uniq = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def find_audio_path(catalog_id: str, audio_dir: str, verbose: bool = False) -> str | None:
    extensions = [".mp3", ".MP3", ".wav", ".WAV", ".flac", ".m4a", ".aac", ".ogg"]
    # Check root first
    for ext in extensions:
        p = os.path.join(audio_dir, f"{catalog_id}{ext}")
        if os.path.exists(p):
            return p
    # Fallback: recursive search
    for root, _, files in os.walk(audio_dir):
        for ext in extensions:
            fname = f"{catalog_id}{ext}"
            if fname in files:
                return os.path.join(root, fname)
    if verbose:
        print({"catalog_id": catalog_id, "reason": "audio_not_found"})
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--number_of_songs", type=int, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest, low_memory=False)

    # Ensure columns exist
    if "CatalogID" not in df.columns:
        print("Manifest must contain CatalogID column", file=sys.stderr)
        sys.exit(2)

    rows = []
    picked = 0
    for _, row in df.iterrows():
        # Robust numeric extraction (handles floats like 12345.0)
        try:
            cid_num = pd.to_numeric(row["CatalogID"], errors="coerce")
        except Exception:
            cid_num = None
        if pd.isna(cid_num):
            continue
        try:
            catalog_id = str(int(cid_num))
        except Exception:
            continue

        mp3_path = find_audio_path(catalog_id, args.audio_dir, verbose=args.verbose)
        if mp3_path is None:
            continue

        lyrics = first_nonempty(str(row.get("Lyrics", "") or ""), str(row.get("ai_LyricTranscription", "") or ""))
        tag_list = build_tags(row)

        rows.append(
            {
                "keys": catalog_id,
                "filename": mp3_path,
                "tags": "|".join(tag_list),  # pipe-separated, split later
                "norm_lyrics": lyrics,
            }
        )
        picked += 1
        if picked >= args.number_of_songs:
            break

    out_dir = os.path.dirname(args.output_csv)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print({
        "prepared_count": len(rows),
        "output_csv": args.output_csv,
        "audio_dir": args.audio_dir,
    })


if __name__ == "__main__":
    main()


