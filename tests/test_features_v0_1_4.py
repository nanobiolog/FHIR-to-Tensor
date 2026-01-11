
import pytest
import torch
import datetime
from neurofhir.hyperbolic_encoder import PoincareEmbedding
from neurofhir.temporal_builder import FHIRTemporalGraphBuilder
from neurofhir.causal_edge_miner import CausalEdgeMiner
import polars as pl

def test_hyperbolic_distance():
    """Test that the hyperbolic distance logic works and matches expected behavior."""
    # 1. Initialize logic
    # We can use the class directly if logic is pure, but it inherits nn.Module
    encoder = PoincareEmbedding(num_embeddings=10, embedding_dim=2)
    
    # 2. Create two vectors close to origin
    u = torch.tensor([0.0, 0.0])
    v = torch.tensor([0.1, 0.0])
    
    # Dist near origin should be approx Euclidean: 0.1 * 2 = 0.2? 
    # Formula: acosh(1 + 2 * 0.01 / (1 * 0.99)) ~ acosh(1 + 0.02) ~ sqrt(0.04) ~ 0.2
    dist_uv = encoder.dist(u, v)
    assert dist_uv > 0
    assert torch.isclose(dist_uv, torch.tensor(0.2), atol=0.01)
    
    # 3. Create vectors near boundary
    x = torch.tensor([0.9, 0.0])
    y = torch.tensor([0.95, 0.0])
    dist_xy = encoder.dist(x, y)
    
    # Euclidean diff is 0.05. Hyperbolic should be Huge.
    # 0.9^2 = 0.81, 0.95^2 = 0.9025
    # denoms: 0.19, 0.0975.  prod ~ 0.018.
    # num: 2 * 0.0025 = 0.005.
    # arg = 1 + 0.005 / 0.018 ~ 1.27
    # acosh(1.27) is not huge yet, but larger than euclidean.
    
    assert dist_xy > torch.norm(x - y)

def test_builder_summary(capsys):
    """Test that summary produces output."""
    builder = FHIRTemporalGraphBuilder()
    
    # Create fake snapshots (dicts for simplicity/non-torch env safety)
    snapshots = [
        {"timestamp": datetime.datetime(2025, 1, 1), "num_nodes": 10, "edges": {("A", "r", "B"): ([0], [1])}},
        {"timestamp": datetime.datetime(2025, 1, 2), "num_nodes": 15, "edges": {}}
    ]
    
    builder.summary(snapshots)
    captured = capsys.readouterr()
    assert "Snapshot 0: 10 nodes, 1 edges" in captured.out
    assert "Avg Nodes/Snap:  12.5" in captured.out

def test_temporal_mining():
    """Test generic temporal precedence mining."""
    miner = CausalEdgeMiner()
    
    # Patient p1: Surgery (Day 1) -> Complication (Day 2)
    # Patient p2: Complication (Day 1) -> Surgery (Day 2) [Should NOT match]
    resources = [
        {"resourceType": "Condition", "id": "e1", "code": {"text": "Surgery"}, "recordedDate": "2025-01-01T10:00:00Z", "subject": {"reference": "Patient/p1"}},
        {"resourceType": "Observation", "id": "e2", "code": {"text": "Complication"}, "recordedDate": "2025-01-02T10:00:00Z", "subject": {"reference": "Patient/p1"}},
        
        {"resourceType": "Observation", "id": "e3", "code": {"text": "Complication"}, "recordedDate": "2025-01-01T10:00:00Z", "subject": {"reference": "Patient/p2"}},
        {"resourceType": "Condition", "id": "e4", "code": {"text": "Surgery"}, "recordedDate": "2025-01-02T10:00:00Z", "subject": {"reference": "Patient/p2"}},
    ]
    
    df = miner.mine_temporal_precedence(resources, "surgery", "complication")
    
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["source"] == "e1"
    assert row["target"] == "e2"
    assert row["relation"] == "PRECEDES"
