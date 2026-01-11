# Copyright (C) 2026 ATIL İHSAN YALI
# This file is part of NeuroFHIR.
#
# NeuroFHIR is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Commercial licensing is available. Contact nano.carbay@gmail.com for details.

import logging
from typing import List, Dict, Any, Iterator, Optional, Union
import datetime

try:
    import polars as pl
except ImportError:
    pl = None

try:
    import torch
    from torch_geometric.data import HeteroData
except ImportError:
    torch = None
    HeteroData = Any

logger = logging.getLogger(__name__)

class FHIRTemporalGraphBuilder:
    """
    The Time Machine: Converts FHIR resources into a sequence of temporal graph snapshots.
    Builds a DynamicHeteroGraph compatible with torch_geometric_temporal.
    """

    def __init__(self, time_window: str = "1d"):
        """
        Initialize the builder.
        
        Args:
            time_window: The time bucket size for snapshots (e.g., '1d', '1h').
                         Must be in Polars duration string format.
        """
        if pl is None:
            raise ImportError("NeuroFHIR requires 'polars' for high-performance temporal alignment.")
        
        # Global node mapping: {ResourceType: {OriginalID: Index}}
        self.node_mapping: Dict[str, Dict[str, int]] = {}
        self.time_window = time_window

    def _get_node_index(self, resource_type: str, resource_id: str) -> int:
        """Get or create a stable integer index for a node."""
        if resource_type not in self.node_mapping:
            self.node_mapping[resource_type] = {}
        
        mapping = self.node_mapping[resource_type]
        if resource_id not in mapping:
            mapping[resource_id] = len(mapping)
        return mapping[resource_id]

    # ... (rest of simple methods like parse_timestamp can stay, assuming they are not in the hunk) ...
    # We focus on the placeholder replacement.

    def _construct_hetero_data(self, df: pl.DataFrame) -> Union[HeteroData, Dict]:
        """
        Constructs a HeteroData object (or dict) from the dataframe of the current snapshot.
        Resolves references to build edge indices.
        """
        if torch is None:
            logger.warning("Torch not found, returning dict representation.") 
            data = {}
        else:
            data = HeteroData()

        # 1. Identify Nodes in this snapshot
        # We must iterate rows to map them to global indices
        # In a very large scale, we'd use polars joins with the mapping DF, 
        # but for iterating logic, row-based is clear.
        
        # Group by type for efficiency
        partitioned = df.partition_by("resourceType", as_dict=True)
        
        snapshot_node_indices: Dict[str, List[int]] = {}
        
        for r_type, sub_df in partitioned.items():
            indices = []
            for row in sub_df.iter_rows(named=True):
                idx = self._get_node_index(r_type, row["id"])
                indices.append(idx)
            snapshot_node_indices[r_type] = indices
            
            # Set num_nodes in HeteroData
            # Note: HeteroData usually expects contiguous 0..N indices for features.
            # If we strictly use global indices, the feature matrix must be global size.
            # For 'Dynamic' graphs, usually we just include the active nodes *or* all nodes.
            # Here we assume we mark the *active* mask or just set num_nodes to max_global so far.
            current_max = len(self.node_mapping[r_type])
            
            if isinstance(data, dict):
                data.setdefault("num_nodes", {})[r_type] = current_max
            else:
                data[r_type].num_nodes = current_max

        # 2. Identify Edges (Reference Resolution)
        # Scan columns that look like References (e.g., subject.reference, context.reference)
        # We look into the 'payload' dict.
        
        edge_lists: Dict[Tuple[str, str, str], Tuple[List[int], List[int]]] = {}

        for row in df.iter_rows(named=True):
            src_type = row["resourceType"]
            src_id = row["id"]
            src_idx = self._get_node_index(src_type, src_id)
            payload = row["payload"]
            
            # Helper to add edge
            def add_edge_if_exists(ref_key: str, edge_type: str):
                ref = payload.get(ref_key, {})
                ref_str = ref.get("reference")
                if not ref_str or "/" not in ref_str:
                    return
                
                parts = ref_str.split("/")
                if len(parts) < 2: 
                    return
                tgt_type, tgt_id = parts[-2], parts[-1]
                
                # Get target index (updating mapping if it's a new node seen in ref)
                tgt_idx = self._get_node_index(tgt_type, tgt_id)
                
                # Canonical edge type: (Source, Relation, Target)
                # In PyG: (src, rel, dst) -> edge_index [2, E] (src_indices, dst_indices)
                # But typically PyG edges are directed src->dst? 
                # Yes. e.g. Observation -> has_subject -> Patient
                
                key = (src_type, edge_type, tgt_type)
                if key not in edge_lists:
                    edge_lists[key] = ([], [])
                
                edge_lists[key][0].append(src_idx)
                edge_lists[key][1].append(tgt_idx)

            # Common FHIR references
            add_edge_if_exists("subject", "refers_to")
            add_edge_if_exists("encounter", "occurs_in")
            add_edge_if_exists("performer", "performed_by")
            # Can extend to other standard fields

        # 3. Assign to Data
        timestamp = df["timestamp"].min()
        if isinstance(data, dict):
            # Dict Mode
            data["edges"] = edge_lists
            data["timestamp"] = timestamp
        else:
            # Torch Mode
            for (src, rel, dst), (srcs, dsts) in edge_lists.items():
                data[src, rel, dst].edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
            
            setattr(data, "timestamp", timestamp)

        return data
