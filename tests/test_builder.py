import pytest
import datetime
from neurofhir.temporal_builder import FHIRTemporalGraphBuilder

def test_builder_initialization():
    builder = FHIRTemporalGraphBuilder(time_window="1d")
    assert builder.time_window == "1d"

def test_simple_snapshot_creation():
    builder = FHIRTemporalGraphBuilder(time_window="1d")
    now = datetime.datetime.now()
    # Mock resources
    resources = [
        {"resourceType": "Patient", "id": "p1", "recordedDate": now.isoformat()},
        {"resourceType": "Observation", "id": "o1", "subject": {"reference": "Patient/p1"}, "effectiveDateTime": now.isoformat()},
    ]
    
    snapshots = list(builder.build_snapshots(resources))
    assert len(snapshots) > 0
    # Add more specific assertions after refactor
