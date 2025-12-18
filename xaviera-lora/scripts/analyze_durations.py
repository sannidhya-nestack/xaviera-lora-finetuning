import os
import argparse
from collections import Counter
import numpy as np
import subprocess

# Target catalog IDs to analyze
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

def analyze_durations(root_dir, target_catalog_ids=None):
    print(f"Scanning for duration files in: {root_dir}")
    
    if target_catalog_ids:
        print(f"Filtering to {len(target_catalog_ids)} target catalog IDs.")
    
    durations = []
    files_processed = 0
    skipped_count = 0
    
    if not os.path.exists(root_dir):
        print(f"Error: Directory not found: {root_dir}")
        return

    # Recursively find all *duration.txt files
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith("_duration.txt"):
                # Extract catalog ID from filename (e.g., "606727_duration.txt" -> "606727")
                catalog_id = file.replace("_duration.txt", "")
                
                # Filter by target catalog IDs if specified
                if target_catalog_ids and catalog_id not in target_catalog_ids:
                    skipped_count += 1
                    continue
                
                path = os.path.join(root, file)
                try:
                    with open(path, 'r') as f:
                        content = f.read().strip()
                        if content:
                            val = float(content)
                            durations.append(val)
                            files_processed += 1
                except ValueError:
                    print(f"Warning: Could not parse duration in {path}")
                except Exception as e:
                    print(f"Error reading {path}: {e}")

    if not durations:
        print("No valid duration files found.")
        if target_catalog_ids:
            print(f"Searched for {len(target_catalog_ids)} catalog IDs, skipped {skipped_count} files.")
        return

    print(f"\nProcessed {files_processed} files.")
    if target_catalog_ids:
        print(f"Skipped {skipped_count} files (not in target catalog IDs).")
        print(f"Found {files_processed} / {len(target_catalog_ids)} target catalog IDs.")
    
    # 1. Round to 1 decimal place for grouping (as per user examples: 9.3s, 1.8s)
    rounded_durations = [round(d, 1) for d in durations]
    
    # 2. Count frequencies
    counts = Counter(rounded_durations)
    
    # 3. Sort by duration
    sorted_items = sorted(counts.items(), key=lambda x: x[0])
    
    print("\n--- Duration Distribution (rounded to 0.1s) ---")
    print(f"{'Count':<10} | {'Duration (s)':<15}")
    print("-" * 30)
    
    for duration, count in sorted_items:
        print(f"{count:<10} | {duration:<15}")

    # 4. Summary Statistics
    dur_array = np.array(durations)
    print("\n--- Summary Statistics ---")
    print(f"Total Songs:      {len(dur_array)}")
    print(f"Min Duration:     {np.min(dur_array):.4f} s")
    print(f"Max Duration:     {np.max(dur_array):.4f} s")
    print(f"Mean Duration:    {np.mean(dur_array):.4f} s")
    print(f"Median Duration:  {np.median(dur_array):.4f} s")
    print(f"Std Dev:          {np.std(dur_array):.4f} s")

    # 5. Batching Recommendation (Heuristic)
    print("\n--- Batching Recommendation ---")
    print("If creating batches of similar duration:")
    # Group into rough buckets (e.g. integer buckets)
    int_counts = Counter([int(d) for d in durations])
    sorted_int_buckets = sorted(int_counts.items())
    
    print(f"{'Bucket (Int)':<15} | {'Songs':<10} | {'Est. Batches (bsz=32)':<20}")
    print("-" * 50)
    for bucket_sec, count in sorted_int_buckets:
        est_batches = max(1, count // 32)
        print(f"{bucket_sec}s - {bucket_sec+0.99}s   | {count:<10} | {est_batches:<20}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze audio duration distribution."
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
        print(f"   python analyze_durations.py /home/ec2-user/SageMaker/xaviera_essential/generated_audio_metadata/Pop")
        print("3. Or provide just the base directory and use --genre:")
        print(f"   python analyze_durations.py /home/ec2-user/SageMaker/xaviera_essential/generated_audio_metadata --genre Pop")
        
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
    
    analyze_durations(root_dir, target_catalog_ids=target_ids)
