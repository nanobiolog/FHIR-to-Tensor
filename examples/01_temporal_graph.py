import sys
import os
import datetime
# Add parent directory to path to allow importing neurofhir from source
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurofhir.temporal_builder import FHIRTemporalGraphBuilder

def main():
    print("=== Example 1: Temporal Patient Modeling ===")
    
    # 1. Initialize the Time Machine
    # We want daily snapshots to capture the progression of vitals.
    builder = FHIRTemporalGraphBuilder(time_window="1d")
    print("Initialized FHIRTemporalGraphBuilder with 1d window.")

    # 2. Load Raw FHIR Data (simulated)
    # In a real scenario, this would come from a FHIR server or JSON dump.
    patient_history = [
        {"resourceType": "Patient", "id": "p1", "recordedDate": "2025-01-01T08:00:00Z"},
        # Day 1: Infection suspected
        {"resourceType": "Condition", "id": "c1", "subject": {"reference": "Patient/p1"}, "code": {"text": "Sepsis"}, "recordedDate": "2025-01-01T09:00:00Z"},
        # Day 2: Antibiotics administered
        {"resourceType": "MedicationRequest", "id": "m1", "subject": {"reference": "Patient/p1"}, "code": {"text": "Antibiotic"}, "authoredOn": "2025-01-02T09:00:00Z"},
    ]
    print(f"Loaded {len(patient_history)} FHIR resources.")

    # 3. Build Dynamic Graph
    # Returns an iterator of PyG HeteroData objects (or dicts if torch is missing)
    snapshots = builder.build_snapshots(patient_history)
    
    # We convert iterator to list to inspect it multiple times (warning: consumes iterator)
    snapshots_list = list(snapshots)

    # 4. Inspect Graph Statistics
    print(f"\nGenerated {len(snapshots_list)} distinct time steps.")
    builder.summary(snapshots_list)

    # 5. Accessing Data
    if snapshots_list:
        first_snap = snapshots_list[0]
        print("\nFirst Snapshot Details:")
        if isinstance(first_snap, dict):
            print("Nodes:", first_snap.get("num_nodes"))
            print("Edges:", first_snap.get("edges").keys())
        else:
            print(first_snap)

if __name__ == "__main__":
    main()
