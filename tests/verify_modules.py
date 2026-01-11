# Copyright (C) 2026 ATIL İHSAN YALI
# This file is part of NeuroFHIR.
# AGPL-3.0 License.

import sys
import os
import datetime
# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurofhir import FHIRTemporalGraphBuilder, PoincareEmbedding, CausalEdgeMiner

def test_temporal_builder():
    print("\n--- Testing FHIRTemporalGraphBuilder ---")
    try:
        builder = FHIRTemporalGraphBuilder(time_window="1d")
        
        # Mock Data
        now = datetime.datetime.now()
        resources = [
            {"resourceType": "Patient", "id": "p1", "recordedDate": now.isoformat()},
            {"resourceType": "Observation", "id": "o1", "effectiveDateTime": now.isoformat()},
            {"resourceType": "Observation", "id": "o2", "effectiveDateTime": (now + datetime.timedelta(days=2)).isoformat()},
        ]

        snapshots = list(builder.build_snapshots(resources))
        print(f"Generated {len(snapshots)} snapshots.")
        for i, snap in enumerate(snapshots):
            print(f"Snapshot {i}: {snap}")
    except ImportError as e:
        print(f"Skipped due to missing dependencies: {e}")
    except Exception as e:
        print(f"Error: {e}")

def test_hyperbolic_encoder():
    print("\n--- Testing PoincareEmbedding ---")
    try:
        encoder = PoincareEmbedding(num_embeddings=100, embedding_dim=8)
        print("Initialized PoincareEmbedding.")
        print("Embeddings shape:", encoder.weight.shape)
    except ImportError as e:
        print(f"Skipped due to missing dependencies: {e}")
    except Exception as e:
        print(f"Error: {e}")

def test_causal_miner():
    print("\n--- Testing CausalEdgeMiner ---")
    try:
        miner = CausalEdgeMiner()
        
        # Mock Data: Infection -> Antibiotic -> Fever Drop
        base_time = datetime.datetime(2025, 1, 1, 10, 0, 0)
        
        resources = [
            # Condition: Infection
            {
                "resourceType": "Condition", "id": "c1", 
                "subject": {"reference": "Patient/1"},
                "code": {"text": "Sepsis infection"},
                "recordedDate": base_time.isoformat()
            },
            # Med: Antibiotic (1 hour later)
            {
                "resourceType": "MedicationRequest", "id": "m1",
                "subject": {"reference": "Patient/1"},
                "code": {"text": "IV Antibiotic"},
                "authoredOn": (base_time + datetime.timedelta(hours=1)).isoformat()
            },
            # Obs: Fever 39.0 (Before Med) - Ignore in miner logic or use for context
            {
                "resourceType": "Observation", "id": "obs_high",
                "subject": {"reference": "Patient/1"},
                "code": {"text": "Body temperature"},
                "valueQuantity": {"value": 39.0},
                "effectiveDateTime": base_time.isoformat()
            },
            # Obs: Fever Drop 37.0 (2 hours after Med)
            {
                "resourceType": "Observation", "id": "obs_low",
                "subject": {"reference": "Patient/1"},
                "code": {"text": "Body temperature"},
                "valueQuantity": {"value": 37.0},
                "effectiveDateTime": (base_time + datetime.timedelta(hours=3)).isoformat()
            }
        ]
        
        edges = miner.mine_relationships(resources)
        print("Mined Edges DataFrame:")
        print(edges)
    except ImportError as e:
        print(f"Skipped due to missing dependencies: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_temporal_builder()
    test_hyperbolic_encoder()
    test_causal_miner()
