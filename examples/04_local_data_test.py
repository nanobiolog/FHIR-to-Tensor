import sys
import os
import json
import glob
from concurrent.futures import ProcessPoolExecutor

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurofhir.temporal_builder import FHIRTemporalGraphBuilder

def process_file(file_path):
    """
    Process a single FHIR bundle file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            bundle = json.load(f)
            
        # Extract entries from bundle
        if bundle.get("resourceType") == "Bundle" and "entry" in bundle:
            resources = [e["resource"] for e in bundle["entry"] if "resource" in e]
        else:
            # Maybe it's a single resource or list
            # For this example, let's assume it's a bundle or skip
            return 0, 0
            
        if not resources:
            return 0, 0
            
        builder = FHIRTemporalGraphBuilder(time_window="1d")
        snapshots = list(builder.build_snapshots(resources))
        
        return len(resources), len(snapshots)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, 0

def main():
    print("=== Example 4: Processing Local Data from 'data/fhir' ===")
    
    # Locate data directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "fhir")
    
    # 1. Find Files
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    if not json_files:
        # Fallback to just "data/" if "data/fhir" is empty or doesn't exist
        data_dir = os.path.join(base_dir, "data")
        json_files = glob.glob(os.path.join(data_dir, "*.json"))

    if not json_files:
        print(f"No JSON files found in {data_dir}. Please run 'download_sample_data.py' or add your own data.")
        return

    print(f"Found {len(json_files)} JSON files in {data_dir}. Processing subset...")
    
    # 2. Process Subset (First 1 for demonstration/inspection)
    # We will save the first one to disk to show "results"
    subset = json_files[:1]
    
    for file_path in subset:
        print(f"Processing {os.path.basename(file_path)}...")
        recs, snapshots_list = process_file_get_snapshots(file_path)
        
        if not snapshots_list:
            continue
            
        print(f"  -> {recs} resources, {len(snapshots_list)} temporal snapshots")
        
        # Save output to a file
        output_file = os.path.join(os.path.dirname(__file__), "output_graph_sample.json")
        
        # Convert to serializable format (if dict)
        serializable_snaps = []
        for i, snap in enumerate(snapshots_list):
            # If it's a dict, it's already close, just need to handle sets/tuples if any
            # The builder returns edges as dict with tuple keys (src, rel, dst) which isn't JSON friendly
            # We strictly convert for display
            snap_clean = {}
            if isinstance(snap, dict):
                # Convert edge keys from ('Patient', 'refers_to', 'Organization') -> "Patient__refers_to__Organization"
                edges_clean = {}
                for k, v in snap.get("edge_index_dict", {}).items():
                    k_str = f"{k[0]}__{k[1]}__{k[2]}"
                    edges_clean[k_str] = {"src": v[0], "dst": v[1]}
                snap_clean["edges"] = edges_clean
                
                # num_nodes might be int or dict. If dict, keys might be tuples or strings.
                nn = snap.get("num_nodes")
                if isinstance(nn, dict):
                    nn_clean = {}
                    for k, v in nn.items():
                        # Key could be string "Patient" or tuple ("Patient",) depending on logic
                        k_str = str(k)
                        if isinstance(k, tuple) and len(k) == 1:
                            k_str = str(k[0])
                        elif isinstance(k, str):
                            k_str = k
                        nn_clean[k_str] = v
                    snap_clean["num_nodes"] = nn_clean
                else:
                    snap_clean["num_nodes"] = nn
                    
                snap_clean["timestamp"] = str(snap.get("timestamp"))
                
            serializable_snaps.append(snap_clean)
            
        with open(output_file, 'w') as f:
            json.dump(serializable_snaps, f, indent=2)
            
        print(f"\n[SUCCESS] Results saved to: {output_file}")
        print("Open this file to inspect the generated graph structure!")

def process_file_get_snapshots(file_path):
    """
    Process a single FHIR bundle file and return snapshots.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            bundle = json.load(f)
            
        if bundle.get("resourceType") == "Bundle" and "entry" in bundle:
            resources = [e["resource"] for e in bundle["entry"] if "resource" in e]
        else:
            return 0, []
            
        if not resources:
            return 0, []
            
        builder = FHIRTemporalGraphBuilder(time_window="1d")
        snapshots = list(builder.build_snapshots(resources))
        
        return len(resources), snapshots
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, []


if __name__ == "__main__":
    main()
