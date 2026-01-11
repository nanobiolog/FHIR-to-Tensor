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

    def build_snapshots(self, fhir_resources: List[Dict[str, Any]]) -> Iterator[Union[HeteroData, Dict]]:
        """
        Main entry point: temporal alignment and graph construction.
        """
        if not fhir_resources:
            return iter([])

        # 1. Parse Resources into a flat DataFrame with timestamps
        data_dicts = []
        for res in fhir_resources:
            # Timestamp priority: recordedDate, effectiveDateTime, issued, authoredOn
            ts_str = res.get("recordedDate") or res.get("effectiveDateTime") or res.get("issued") or res.get("authoredOn")
            
            if not ts_str:
                continue
                
            try:
                if ts_str.endswith("Z"):
                     ts_str = ts_str[:-1] + "+00:00"
                ts = datetime.datetime.fromisoformat(ts_str)
            except ValueError:
                continue

            data_dicts.append({
                "resourceType": res.get("resourceType"),
                "id": res.get("id"),
                "timestamp": ts,
                "payload": res # Store full payload for edge extraction
            })
            
        if not data_dicts:
            return iter([])

        df = pl.DataFrame(data_dicts).sort("timestamp")
        
        # 2. Group by Time Window (Downsampling)
        # using dynamic grouper or manual iteration.
        # For simplicity/robustness: use group_by_dynamic
        
        # We want snapshots.
        # df = df.with_columns(pl.col("timestamp").dt.truncate(self.time_window).alias("window_start"))
        # partitioned = df.partition_by("window_start", maintain_order=True)
        
        # Dynamic grouping usually requires aggregation. Here we just want to split.
        # Let's use dt.truncate to assign specific windows.
        
        df = df.with_columns(pl.col("timestamp").dt.truncate(self.time_window).alias("snapshot_idx"))
        
        # 3. Construct Graphs per Window
        for _, window_df in df.group_by("snapshot_idx", maintain_order=True):
             yield self._construct_hetero_data(window_df)

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
        
        edges_dict: Dict[Tuple[str, str, str], Tuple[List[int], List[int]]] = {}

        # 3. Vectorized Edge Construction
        # We process each relation type we care about
        # Defined as: (ReferenceFieldPath, EdgeType, TargetResourceType)
        # Note: In generic FHIR, we might not know TargetType ahead of time if reference string is "Type/123".
        # We can extract it processing the whole payload column.
        
        # Ensure we have a mapping from (Type, ID) -> GlobalIdx
        # Create a DF for joining:
        # We need a unified DF of all nodes in this snapshot with their global indices.
        # Since _get_node_index is stateful, we must ensure all IDs in this 'df' are mapped.
        # The previous loop (Step 1) already populated self.node_mapping and snapshot_node_indices.
        # Let's create a Polars DF for the mapping to enable joins.
        
        # Flatten the current mapping for resources present in 'df'
        # To do this efficiently, we can use the 'df' and map using map_elements (slow) or join.
        # Better: Create a small DF of (Type, ID, Index) from the loop we just did?
        # Step 1 logic was: iterate per type, get_index.
        # We can optimize Step 1 to be:
        # unique_ids = df.select("resourceType", "id").unique()
        # Then map them.
        
        # For simplicity in this "Refactor Phase", we will stick to the fact that we have 'df'.
        # We'll rely on the Reference string extracting (Type, ID).
        
        # 3a. Define Relations to extract
        relation_configs = [
            # (payload_field, edge_relation)
            ("subject", "refers_to"),
            ("encounter", "occurs_in"),
            ("performer", "performed_by"),
        ]

        # 3b. Process each relation
        for ref_field, edge_rel in relation_configs:
            # Filter rows that have this field
            # We assume 'payload' is struct. If not, this might fail, so we wrap in try/except or check schema.
            # Safe approach using map_elements roughly for now if schema is unknown, 
            # OR assuming the user provided standard Polars Structs.
            # Given the previous code iterated row["payload"], it implies Mixed or Struct.
            
            # Extract references: source_type, source_id, target_ref_str
            # We use `pl.col("payload").struct.field(ref_field).struct.field("reference")`
            # This is efficient.
            try:
                # Select only relevant columns
                edge_candidates = df.select([
                    pl.col("resourceType").alias("src_type"),
                    pl.col("id").alias("src_id"),
                    pl.col("payload").struct.field(ref_field).struct.field("reference").alias("ref_str")
                ]).filter(pl.col("ref_str").is_not_null())
                
                if edge_candidates.height == 0:
                    continue

                # Parse ref_str "Type/ID" -> "tgt_type", "tgt_id"
                # split by /
                edge_candidates = edge_candidates.with_columns([
                    pl.col("ref_str").str.split("/").arr.get(-2).alias("tgt_type"),
                    pl.col("ref_str").str.split("/").arr.get(-1).alias("tgt_id")
                ])
                
                # Now we have src and tgt identifiers. We need to convert them to indices.
                # Since 'node_mapping' is a Dict of Dicts, we can't easily join on it unless we convert it to a DF.
                # HOWEVER, for a PRODUCTION builder, the mapping should be a DB or a growing DF.
                # Here, we can do a quick lookup using map_dict if the mapping isn't massive, 
                # or loop over the edge candidates df (which is much smaller than full df usually).
                
                # Compromise: Vectorized extraction, iterating for index lookup (still usually faster because python loop is smaller).
                # OR, strictly vectorized:
                # We have to handle that 'tgt' might be a NEW node not in 'df' (e.g. referencing a Patient not in this snapshot).
                # If so, we must register it.
                
                # Let's iterate the edge_candidates to register nodes and build indices. 
                # This is faster than iterating *every* row, just edges.
                
                src_indices = []
                tgt_indices = []
                
                # Group by types to use specific mapping dicts
                for row in edge_candidates.iter_rows(named=True):
                    src_t, src_i = row["src_type"], row["src_id"]
                    tgt_t, tgt_i = row["tgt_type"], row["tgt_id"]
                    
                    # Src index (should already be known from step 1, but safe to get)
                    s_idx = self._get_node_index(src_t, src_i)
                    
                    # Tgt index (might be new/external)
                    t_idx = self._get_node_index(tgt_t, tgt_i)
                    
                    src_indices.append(s_idx)
                    tgt_indices.append(t_idx)
                
                if src_indices:
                    # We might have mixed target types in one ref field (unlikely in FHIR strict, but possible "Subject" is Group or Patient)
                    # We need to segregate by (src_type, tgt_type).
                    # Re-iterate is sad. 
                    # Optimization: The previous loop could group into dicts.
                    
                    # Revised Loop:
                    for row in edge_candidates.iter_rows(named=True):
                         src_t, src_i = row["src_type"], row["src_id"]
                         tgt_t, tgt_i = row["tgt_type"], row["tgt_id"]
                         
                         key = (src_t, edge_rel, tgt_t)
                         if key not in edges_dict:
                             edges_dict[key] = ([], [])
                         
                         edges_dict[key][0].append(self._get_node_index(src_t, src_i))
                         edges_dict[key][1].append(self._get_node_index(tgt_t, tgt_i))

            except Exception as e:
                # Likely field doesn't exist in schema
                # logger.debug(f"Skipping edge extraction for {ref_field}: {e}")
                pass
        
        # Rename for compatibility
        edge_lists = edges_dict

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
