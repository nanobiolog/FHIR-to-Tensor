import sys
import os
# Add parent directory to path to allow importing neurofhir from source
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurofhir.causal_edge_miner import CausalEdgeMiner
import datetime

def main():
    print("=== Example 3: Causal Edge Mining ===")
    
    miner = CausalEdgeMiner()
    print("Initialized CausalEdgeMiner.")

    # 1. Define Patient History with Potential Causality
    # Scenario: Med X given -> Symptom Y recorded later
    base_time = datetime.datetime(2025, 1, 1, 10, 0, 0)
    
    patient_history = [
        # T=0: Admission
        {"resourceType": "Encounter", "id": "enc1", "period": {"start": base_time.isoformat()}},
        
        # T=1: Drug Administered (Antibiotic)
        {"resourceType": "MedicationRequest", "id": "med1", "code": {"text": "Antibiotic"}, "authoredOn": (base_time + datetime.timedelta(hours=1)).isoformat()},
        
        # T=2: Symptom Reported (Rash) - Potential Side Effect?
        {"resourceType": "Observation", "id": "obs1", "code": {"text": "Skin Rash"}, "effectiveDateTime": (base_time + datetime.timedelta(hours=4)).isoformat()},
        
        # T=3: Condition Improved (Fever gone) - Potential Treatment Effect?
        {"resourceType": "Observation", "id": "obs2", "code": {"text": "Body Temperature"}, "valueQuantity": {"value": 36.6, "unit": "C"}, "effectiveDateTime": (base_time + datetime.timedelta(hours=6)).isoformat()},
    ]

    # 2. Mine Relationships
    # Auto-detects patterns based on temporal precedence and rules
    print("Mining relationships from history...")
    edges_df = miner.mine_relationships(patient_history)

    # 3. Inspect the Results
    if edges_df is not None and not edges_df.is_empty():
        print("\nFound Potential Causal Edges:")
        print(edges_df.select(["source", "relation", "target", "weight"]))
        
        # 4. Export to DAG (if networkx is installed)
        try:
            dag = miner.create_dag(edges_df)
            print(f"\nSuccessfully created NetworkX DAG with {dag.number_of_nodes()} nodes and {dag.number_of_edges()} edges.")
        except ImportError:
            print("\nNetworkX not installed, skipping DAG creation.")
    else:
        print("\nNo significant causal edges found with default rules.")

if __name__ == "__main__":
    main()
