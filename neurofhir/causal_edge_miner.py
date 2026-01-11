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
from typing import List, Dict, Any, Tuple
import datetime

try:
    import polars as pl
except ImportError:
    pl = None

logger = logging.getLogger(__name__)

class CausalEdgeMiner:
    """
    The Doctor's Logic: Mines causal relationships from FHIR events.
    Uses DAG optimization to link interventions to outcomes.
    """

    def __init__(self):
        if pl is None:
            raise ImportError("NeuroFHIR requires 'polars' for CausalEdgeMiner.")

    def mine_relationships(self, fhir_resources: List[Dict[str, Any]]) -> pl.DataFrame:
        """
        Identify causal links like CAUSES_IMPROVEMENT.
        Returns a DataFrame of edges: [source_id, target_id, relation, weight, timestamp].
        """
        # 1. Preprocess into Events
        # We need specific types: Condition, MedicationRequest, Observation
        events = []
        for res in fhir_resources:
            rt = res.get("resourceType")
            if rt not in ["Condition", "MedicationRequest", "Observation"]:
                continue
            
            # Simple timestamp extraction (reuse logic or simplify)
            ts_str = res.get("effectiveDateTime") or res.get("recordedDate") or res.get("issued")
            # For medications, period.start
            if not ts_str and rt == "MedicationRequest":
                # Meds might not have period in some simplified FHIR, check other fields
                ts_str = res.get("authoredOn")

            if not ts_str:
                continue

            try:
                # Handle 'Z' for UTC and ensure timezone awareness
                if ts_str.endswith("Z"):
                    ts_str = ts_str[:-1] + "+00:00"
                ts = datetime.datetime.fromisoformat(ts_str)
            except ValueError:
                logger.debug(f"Skipping resource {res.get('id')} due to invalid date: {ts_str}")
                continue

            # Extract Codes (simplified)
            # In real FHIR, walk code.coding
            code_text = ""
            cc = res.get("code", {})
            if cc.get("text"):
                code_text = cc.get("text")
            elif cc.get("coding"):
                code_text = cc["coding"][0].get("display", "") or cc["coding"][0].get("code", "")
            
            # Value for Observations
            val = None
            if rt == "Observation":
                # valueQuantity.value
                val = res.get("valueQuantity", {}).get("value")

            events.append({
                "id": res.get("id"),
                "type": rt,
                "code": code_text.lower(),
                "timestamp": ts,
                "value": val,
                "patient": res.get("subject", {}).get("reference", "")
            })

        if not events:
            return pl.DataFrame()

        df = pl.DataFrame(events)

        # 2. Rule: Infection -> Antibiotic -> Fever Drop
        
        # A. Find Infections
        # A. Find Infections
        conditions = df.filter(
            (pl.col("type") == "Condition") & 
            (pl.col("code").str.contains("infection|sepsis|pneumonia"))
        )

        # B. Find Antibiotics started AFTER infection
        meds = df.filter(
            (pl.col("type") == "MedicationRequest") &
            (pl.col("code").str.contains("antibiotic|penicillin|fluoroquinolone"))
        )

        # C. Find Fever Observations
        obs = df.filter(
            (pl.col("type") == "Observation") &
            (pl.col("code").str.contains("temp|fever")) &
            (pl.col("value").is_not_null())
        )
        
        # Sort observations by time to check for drops?
        # For simplicity of the "Fever Drop" rule, we verify that the observation 
        # timestamp is after med, and (optionally) value is < threshold or dropping.
        # Let's assume finding ANY fever observation after med implies check.
        # A robust "Drop" check requires comparing Obs at t_after vs Obs at t_before.
        
        # Join Condition + Meds (Med after Cond)
        # Using cross join on patient if not too big, or join on patient
        # Polars join
        
        # cond join med on patient
        cond_med = conditions.join(meds, on="patient", how="inner", suffix="_med")
        
        # Filter: Med.ts > Cond.ts
        cond_med = cond_med.filter(pl.col("timestamp_med") > pl.col("timestamp"))
        
        # Join with Obs (Obs after Med)
        triplets = cond_med.join(obs, on="patient", how="inner", suffix="_obs")
        
        # Filter: Obs.ts > Med.ts
        triplets = triplets.filter(pl.col("timestamp_obs") > pl.col("timestamp_med"))
        
        # Filter: "Fever Drop" logic (simplified: value < 37.5 or 99.5 depending on unit)
        # Assuming Celsius for demo
        triplets = triplets.filter(pl.col("value") < 38.0) 

        # Construct Edges
        # Relation: Med -> Condition (TREATS)
        # Relation: Med -> Observation (CAUSES_IMPROVEMENT)
        
        edges = []
        if triplets.height > 0:
            # We have matches
            # Create edges for the graph
            # Med -> Obs
            med_ids = triplets["id_med"]
            obs_ids = triplets["id_obs"]
            
            new_edges = pl.DataFrame({
                "source": med_ids,
                "target": obs_ids,
                "relation": "CAUSES_IMPROVEMENT",
                "weight": 1.0, 
                "timestamp": triplets["timestamp_obs"] # link valid at obs time
            })
            edges.append(new_edges)

        if edges:
            return pl.concat(edges).unique()
        
        return pl.DataFrame(schema=["source", "target", "relation", "weight", "timestamp"])

    def create_dag(self, edges_df: pl.DataFrame) -> Any:
        """
        Convert edge dataframe to NetworkX DAG for analysis if needed.
        """
        try:
            import networkx as nx
        except ImportError:
            logger.warning("NetworkX not found. Returning adjacency dict.")
            # Fallback: Adjacency Dict
            adj = {}
            if edges_df.height > 0:
                for row in edges_df.iter_rows(named=True):
                    src, dst = row["source"], row["target"]
                    if src not in adj: adj[src] = []
                    adj[src].append(dst)
            return adj

        G = nx.DiGraph()
        if edges_df.height > 0:
            for row in edges_df.iter_rows(named=True):
                # Add edge with attributes
                G.add_edge(row["source"], row["target"], 
                           relation=row["relation"], 
                           weight=row["weight"], 
                           timestamp=row["timestamp"])
        return G
