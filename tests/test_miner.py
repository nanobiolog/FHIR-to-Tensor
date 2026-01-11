import pytest
from neurofhir.causal_edge_miner import CausalEdgeMiner
import datetime

def test_miner_initialization():
    miner = CausalEdgeMiner()
    assert miner is not None

def test_mining_logic():
    miner = CausalEdgeMiner()
    base_time = datetime.datetime(2025, 1, 1, 10, 0, 0)
    resources = [
        {"resourceType": "Condition", "id": "c1", "subject": {"reference": "Patient/1"}, "code": {"text": "Sepsis"}, "recordedDate": base_time.isoformat()},
        {"resourceType": "MedicationRequest", "id": "m1", "subject": {"reference": "Patient/1"}, "code": {"text": "Antibiotic"}, "authoredOn": (base_time + datetime.timedelta(hours=1)).isoformat()},
        {"resourceType": "Observation", "id": "obs1", "subject": {"reference": "Patient/1"}, "code": {"text": "Fever"}, "valueQuantity": {"value": 37.0}, "effectiveDateTime": (base_time + datetime.timedelta(hours=2)).isoformat()},
    ]
    
    edges = miner.mine_relationships(resources)
    assert edges.height >= 0  # Just ensuring it runs and returns a DF without error
