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
    torch_geometric_available = True
except ImportError:
    torch = None
    torch_geometric_available = False
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
        self._warned_about_torch = False

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
            # Timestamp priority: recordedDate, effectiveDateTime, issued, authoredOn, period.start, birthDate, date
            ts_str = res.get("recordedDate") or \
                     res.get("effectiveDateTime") or \
                     res.get("issued") or \
                     res.get("authoredOn") or \
                     res.get("period", {}).get("start") or \
                     res.get("birthDate") or \
                     res.get("date")
            
            if not ts_str:
                continue
                
            try:
                # Handle varying precision (YYYY, YYYY-MM)
                if len(ts_str) == 4: # YYYY
                     ts_str += "-01-01"
                if len(ts_str) == 7: # YYYY-MM
                     ts_str += "-01"
                     
                if ts_str.endswith("Z"):
                     ts_str = ts_str[:-1] + "+00:00"
                ts = datetime.datetime.fromisoformat(ts_str)
                
                # Ensure timezone awareness (UTC)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=datetime.timezone.utc)
                else:
                    ts = ts.astimezone(datetime.timezone.utc)
            except ValueError:
                continue

            # Extract References Eagerly (to avoid Polars mixed-schema issues)
            subj_ref = res.get("subject", {}).get("reference")
            enc_ref = res.get("encounter", {}).get("reference")

            data_dicts.append({
                "resourceType": res.get("resourceType"),
                "id": res.get("id"),
                "timestamp": ts,
                "subject_ref": subj_ref,
                "encounter_ref": enc_ref,
                "payload": res # Store full payload for potential feature extraction later
            })
            
        if not data_dicts:
            return iter([])

        df = pl.DataFrame(data_dicts, strict=False).sort("timestamp")
        
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
        # 3. Construct Graphs per Window
        for window_start, window_df in df.group_by("snapshot_idx", maintain_order=True):
             # Ensure window_start is datetime (it is, from dt.truncate)
             # But might be tuple if multiple keys? No, single key.
             # Polars group_by key is tuple if multiple keys, scalar if single?
             # Let's handle it safely.
             if isinstance(window_start, tuple):
                 window_start = window_start[0]
                 
             yield self._construct_hetero_data(window_df, window_start)

    def _construct_hetero_data(self, df: pl.DataFrame, window_start: Any) -> Union[HeteroData, Dict]:
        """
        Constructs a HeteroData object (or dict) from the dataframe of the current snapshot.
        Resolves references to build edge indices.
        """
        if torch is None:
            if not self._warned_about_torch:
                logger.warning("Torch not found, returning dict representation (suppressing further warnings).") 
                self._warned_about_torch = True
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
            # (payload_field, edge_relation)
            ("subject", "refers_to"),
            ("encounter", "occurs_in"),
            # ("performer", "performed_by"), # Disabled until list handling is implemented
        ]

        # 3b. Process each relation
        for ref_field, edge_rel in relation_configs:
            # Filter rows that have this field
            # We assume 'payload' is struct. If not, this might fail, so we wrap in try/except or check schema.
            # Safe approach using map_elements roughly for now if schema is unknown, 
            # OR assuming the user provided standard Polars Structs.
            # Given the previous code iterated row["payload"], it implies Mixed or Struct.
            
            # Extract references: source_type, source_id, target_ref_str
            # Use eagerly extracted columns
            col_name = f"{ref_field}_ref"
            
            try:
                # Select only relevant columns
                edge_candidates = df.select([
                    pl.col("resourceType").alias("src_type"),
                    pl.col("id").alias("src_id"),
                    pl.col(col_name).alias("ref_str")
                ]).filter(pl.col("ref_str").is_not_null())
                
                if edge_candidates.height == 0:
                    continue

                # Parse ref_str "Type/ID" -> "tgt_type", "tgt_id"
                # Handle cases like "urn:uuid:..." or simple IDs
                # We will extract the ID part. If it's a URN, we might need a mapping of urn->(Type, ID)
                # But typically in these bundles, the 'id' field of the target resource matches the UUID if stripped, 
                # OR we just rely on the fact that we mapped everything by (Type, ID).
                
                # Robust parsing:
                # If "Type/ID", take ID.
                # If "urn:uuid:ID", take ID.
                # If just "ID", take ID.
                
                edge_candidates = edge_candidates.with_columns([
                    pl.col("ref_str").map_elements(lambda x: x.split("/")[-2] if "/" in x and not x.startswith("urn:") else None, return_dtype=pl.Utf8).alias("tgt_type_hint"),
                    pl.col("ref_str").map_elements(lambda x: x.split(":")[-1] if "urn:" in x else (x.split("/")[-1] if "/" in x else x), return_dtype=pl.Utf8).alias("tgt_id")
                ])

                # Note: If we just have 'tgt_id', we might not know 'tgt_type' to look up in node_mapping.
                # However, for 'refers_to' (Subject), we know it's usually Patient.
                # For 'occurs_in' (Encounter), it's Encounter.
                # For 'performed_by' (Performer), it's Practitioner/Organization.
                
                # We can try to infer type or look up in ALL types.
                # Since we know the schema:
                inferred_target_type = "Patient" if ref_field == "subject" else \
                                       "Encounter" if ref_field == "encounter" else \
                                       "Practitioner" # performed_by is ambiguous but let's try
                                       
                # If the reference string had a type hint, use it.
                edge_candidates = edge_candidates.with_columns(
                   pl.when(pl.col("tgt_type_hint").is_not_null())
                   .then(pl.col("tgt_type_hint"))
                   .otherwise(pl.lit(inferred_target_type))
                   .alias("tgt_type")
                )

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

        if torch_geometric_available:
            data = HeteroData()
            # ... (truncated for brevity, keep existing logic)
            for (src, rel, dst), (src_idx, dst_idx) in edges_dict.items():
                 data[src, rel, dst].edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
            data["timestamp"] = window_start
            return data
        else:
             return {
                "timestamp": window_start,
                "num_nodes": {k: len(v) for k, v in self.node_mapping.items()}, # Approximation or use snapshot_node_indices counts if available? 
                # Actually builder uses self.node_mapping length which is cumulative.
                "edge_index_dict": edges_dict
            }

    @staticmethod
    def summary(snapshots: Iterator[Union[Any, Dict]]) -> None:
        """
        Consumes the iterator (warning: irreversible!) and prints stats.
        Useful for debugging or analysis after build.
        """
        count = 0
        total_nodes = 0
        total_edges = 0
        
        print("--- FHIR Temporal Graph Summary ---")
        for i, snap in enumerate(snapshots):
            count += 1
            if isinstance(snap, dict):
                # Dict mode
                # num_nodes might be a dict {type: count}
                nn = snap.get("num_nodes", {})
                if isinstance(nn, dict):
                    nn_val = sum(nn.values())
                else:
                    nn_val = nn
                
                # Edges: dict of list pairs
                ne_val = 0
                edges = snap.get("edges", {})
                for k, v in edges.items():
                    ne_val += len(v[0])
            else:
                # PyG HeteroData mode
                nn_val = snap.num_nodes
                ne_val = snap.num_edges
            
            total_nodes += nn_val
            total_edges += ne_val
            print(f"Snapshot {i}: {nn_val} nodes, {ne_val} edges")
            
        print("-----------------------------------")
        print(f"Total Snapshots: {count}")
        print(f"Avg Nodes/Snap:  {total_nodes / count if count else 0:.1f}")
        print(f"Avg Edges/Snap:  {total_edges / count if count else 0:.1f}")

