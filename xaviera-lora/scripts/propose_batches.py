import os
import argparse
import math
import re
import subprocess

# Target catalog IDs to analyze (same as analyze_durations.py)
TARGET_CATALOG_IDS = {
    "606727", "806272", "1131440", "629288", "743703", "1252753", "884820", "857661", "780355", "579114",
    "477011", "668663", "734353", "459031", "743095", "846269", "804448", "260216", "1271020", "988739",
    "578707", "598650", "701181", "673135", "808432", "673368", "748765", "474389", "967621", "693479",
    "804447", "812959", "744224", "786723", "796190", "820463", "808836", "1271025", "920083", "462750",
    "500897", "706386", "789708", "635226", "844344", "1270342", "856939", "1267982", "860575", "703853",
    "556410", "598646", "866172", "455620", "470647", "542906", "642051", "259160", "876931", "1270305",
    "1271651", "777283", "527432", "360286", "490815", "620073", "520187", "730477", "865743", "598640",
    "790105", "532453", "755744", "602172", "640454", "1270762", "363319", "455619", "743624", "538286",
    "635888", "620664", "610123", "527434", "1135470", "804450", "692418", "358163", "452852", "512905",
    "762677", "695869", "635231", "465491", "593703", "519110", "754243", "1270339", "804453", "355050",
    "1193429", "741431", "428833", "587339", "778275", "809057", "818393", "851733", "632275", "748811",
    "1293406", "773660", "629125", "803924", "678513", "835712", "578947", "261318", "524844", "920317",
    "1081270", "527414", "802315", "795543", "723954", "741683", "782522", "1271038", "880563", "1251850",
    "891548", "1254164", "1188345", "557198", "1188263", "511991", "848446", "519690", "793139", "519443",
    "1071672", "678270", "556419", "804454", "629287", "169081", "685828", "1251476", "512475", "313135",
    "620670", "1271037", "419275", "577697", "851764", "928869", "1169034", "820122", "740351", "459316",
    "724373", "342643", "804452", "813524", "797460", "418574", "806275", "575423", "861481", "635342",
    "777306", "840463", "787796", "573947", "746125", "714861", "635223", "527440", "616103", "491616",
    "1270398", "1251253", "452249", "780132", "1252201", "1270309", "636368", "527441", "668343", "863336",
    "820310", "748813", "771423", "836194", "761198", "1271023", "711294", "879786", "608621", "739878",
    "842380", "751066", "852355", "1270359", "868633", "806277", "794876", "576060", "748591", "607688",
    "556476", "489829", "835714", "756407", "1129843", "556413", "743988", "631668", "398141", "695559",
    "1271039", "525876", "561327", "377488", "804145", "1162578", "816492", "784778", "886033", "509619",
    "851742", "517927", "543368", "1270759", "1070299", "718300", "795563", "470559", "639035", "701891",
    "741401", "723949", "866163", "754591", "866883", "1243460", "1270400", "700742", "525161", "1233477",
    "271303", "878310", "801399", "754519", "1250130", "821929", "714716", "168651", "878311", "813729",
    "645208", "707638", "672039", "804146", "1251485", "679413", "527436", "793926", "803278", "565244",
    "440098", "866047", "527516", "343839", "678192", "1312203", "664419", "847059", "793712", "420200",
    "692610", "519892", "802399", "1057391", "786734", "836070", "740102", "882457", "1247342", "420164",
    "739291", "517137", "1270401", "860646", "759486", "357158", "771422", "671868", "728763", "786726",
    "701895", "1128255", "405078", "874734", "1270307", "780688", "519505", "490340", "724163", "1103928",
    "834817", "740363", "419885", "1173224", "804158", "881261", "804456", "793748", "630511", "1268202",
    "851385", "474062", "761438", "760453", "851748"
}

def find_generated_audio_metadata():
    """Try to automatically find the generated_audio_metadata directory."""
    # Get script directory - this is the most reliable since script is in xaviera_essential
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Common search locations (prioritize relative to script location)
    search_paths = [
        # Relative to script location (most reliable - script is in xaviera_essential)
        os.path.join(script_dir, "generated_audio_metadata"),
        # Current working directory
        os.path.join(os.getcwd(), "generated_audio_metadata"),
        # Common absolute paths (fallback)
        "/home/ec2-user/SageMaker/xaviera_essential/generated_audio_metadata",
        "/home/ec2-user/SageMaker/xaviera-lora-finetuning/xaviera-lora/generated_audio_metadata",
    ]
    
    # Also try to find it using find command if available
    try:
        result = subprocess.run(
            ["find", "/home/ec2-user", "-type", "d", "-name", "generated_audio_metadata", "-maxdepth", "5"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            found_paths = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            # Filter out trash and virtual documents
            for path in found_paths:
                if 'Trash' not in path and '.virtual_documents' not in path:
                    search_paths.insert(0, path)
                    break
    except Exception:
        pass
    
    # Check each path
    for path in search_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return path
    
    return None

def propose_batches(root_dir, max_seconds_per_batch=300, target_catalog_ids=None):
    print(f"Scanning for duration files in: {root_dir}")
    print(f"Configuration: Batching songs to fit approx {max_seconds_per_batch}s audio per batch.")
    
    filter_msgs = ["Filtering: Removing songs > 240s."]
    if target_catalog_ids:
        filter_msgs.append(f"Filtering to {len(target_catalog_ids)} target catalog IDs.")
    print(" ".join(filter_msgs) + "\n")

    # Store: buckets[lower_bound] = [ (cat_id, duration), ... ]
    buckets = {}
    
    # Regex to capture catalog_id from path
    # Pattern: .../Pop/{cat_id}/{cat_id}_duration.txt
    # We'll just look for the file and infer cat_id from the parent folder name
    
    if not os.path.exists(root_dir):
        print(f"Error: Directory not found: {root_dir}")
        return

    files_processed = 0
    dropped_count = 0
    skipped_count = 0
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith("_duration.txt"):
                path = os.path.join(root, file)
                # Infer cat_id from parent folder name
                # path is .../Pop/CAT_ID/CAT_ID_duration.txt
                cat_id = os.path.basename(os.path.dirname(path))
                
                # Filter by target catalog IDs if specified
                if target_catalog_ids and cat_id not in target_catalog_ids:
                    skipped_count += 1
                    continue
                
                try:
                    with open(path, 'r') as f:
                        content = f.read().strip()
                        if content:
                            dur = float(content)
                            
                            # Filter > 240s
                            if dur > 240.0:
                                dropped_count += 1
                                continue
                                
                            # Bucket by 5 seconds: 0-4.99, 5-9.99, ...
                            bucket_key = int(dur // 5) * 5
                            
                            if bucket_key not in buckets:
                                buckets[bucket_key] = []
                            buckets[bucket_key].append((cat_id, dur))
                            files_processed += 1
                            
                except ValueError:
                    pass
                except Exception:
                    pass

    print(f"Processed {files_processed} songs (Dropped {dropped_count} > 240s).")
    if target_catalog_ids:
        print(f"Skipped {skipped_count} files (not in target catalog IDs).")
        print(f"Found {files_processed} / {len(target_catalog_ids)} target catalog IDs.")
    print("\n" + "="*140)
    # Header
    # Bucket | Count | BatchSize | Batches (Catalog IDs)
    header = f"{'Bucket (s)':<12} | {'Count':<5} | {'BSz':<3} | {'Batches (Catalog IDs)':<80}"
    print(header)
    print("="*140)

    # Sort buckets
    sorted_keys = sorted(buckets.keys())
    
    for key in sorted_keys:
        songs = buckets[key]
        # Sort songs by duration within bucket for tighter packing (optional, but good practice)
        songs.sort(key=lambda x: x[1])
        
        count = len(songs)
        avg_dur = sum(s[1] for s in songs) / count
        
        # Determine Batch Size
        # Heuristic: Fit max_seconds_per_batch per batch
        # e.g. if avg_dur is 200s, max_audio=300 -> batch_size = 1
        # if avg_dur is 50s, max_audio=300 -> batch_size = 6
        suggested_bsz = max(1, int(max_seconds_per_batch // avg_dur))
        
        # Create batches
        # We just chunk the list of cat_ids
        cat_ids = [s[0] for s in songs]
        batches_str_list = []
        
        for i in range(0, count, suggested_bsz):
            batch_chunk = cat_ids[i : i + suggested_bsz]
            # Format batch as "id1 id2"
            batch_str = " ".join(batch_chunk)
            batches_str_list.append(batch_str)
            
        # Join batches with comma
        all_batches_display = " , ".join(batches_str_list)
        
        bucket_range = f"{key}-{key+4} s"
        
        print(f"{bucket_range:<12} | {count:<5} | {suggested_bsz:<3} | {all_batches_display}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Propose batching strategy."
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        help="Root directory to search for *_duration.txt files (e.g., /path/to/generated_audio_metadata/Pop). "
             "If not provided, will attempt to auto-detect."
    )
    parser.add_argument(
        "--genre",
        default="Pop",
        help="Genre subdirectory name (default: Pop)"
    )
    parser.add_argument(
        "--max_audio",
        type=int,
        default=300,
        help="Target max total audio seconds per batch (controls batch size)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all files (ignore target catalog IDs filter)"
    )
    
    args = parser.parse_args()
    
    # Determine the root directory path
    if args.root_dir:
        root_dir = args.root_dir
        # Check if it's a base directory (doesn't end with genre) or full path
        if os.path.exists(root_dir):
            # If it exists and is a directory, check if it contains the genre subdirectory
            genre_path = os.path.join(root_dir, args.genre)
            if os.path.exists(genre_path):
                # User provided base directory, append genre
                root_dir = genre_path
            # Otherwise, assume user provided full path including genre
    else:
        # Try to auto-detect
        print("No path provided, attempting to auto-detect generated_audio_metadata directory...")
        base_dir = find_generated_audio_metadata()
        if base_dir:
            root_dir = os.path.join(base_dir, args.genre)
            print(f"Auto-detected base directory: {base_dir}")
            print(f"Using genre subdirectory: {args.genre}")
        else:
            # Fall back to relative path from script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.join(script_dir, "generated_audio_metadata", args.genre)
    
    root_dir = os.path.abspath(root_dir)
    
    if not os.path.exists(root_dir):
        print(f"\n[ERROR] The directory does not exist!")
        print(f"Path looked for: {root_dir}")
        print("\nTroubleshooting:")
        print("1. Check if the path is correct")
        print("2. Try providing the full path explicitly:")
        print(f"   python propose_batches.py /home/ec2-user/SageMaker/xaviera_essential/generated_audio_metadata/Pop")
        print("3. Or provide just the base directory and use --genre:")
        print(f"   python propose_batches.py /home/ec2-user/SageMaker/xaviera_essential/generated_audio_metadata --genre Pop")
        
        # Try to suggest the correct path
        base_dir = find_generated_audio_metadata()
        if base_dir:
            suggested_path = os.path.join(base_dir, args.genre)
            print(f"\nSuggested path (auto-detected): {suggested_path}")
            if os.path.exists(suggested_path):
                print("  ^ This path exists! Try using it.")
        exit(1)
    
    # Use target catalog IDs unless --all flag is set
    target_ids = None if args.all else TARGET_CATALOG_IDS
    
    propose_batches(root_dir, args.max_audio, target_catalog_ids=target_ids)
