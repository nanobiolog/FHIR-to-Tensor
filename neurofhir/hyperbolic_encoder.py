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
import math
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    # Dummy nn.Module for type checking/runtime safety if torch missing
    class MockModule:
        pass
    class MockNN:
        pass
    nn = MockNN
    nn.Module = MockModule

logger = logging.getLogger(__name__)

# Lightweight ontology of common ICD-10 roots for initialization demo
ONTOLOGY_ROOTS = {
    "ICD-10": {
        "A00-B99": ["A00", "A01", "B20"], # Certain infectious and parasitic diseases
        "C00-D48": ["C34", "C50"], # Neoplasms
        "J00-J99": ["J18", "J45"], # Respiratory
    }
}

class PoincareEmbedding(nn.Module):
    """
    The Ontology Brain: Embeds medical codes into a Poincaré ball.
    Values are initialized based on a hierarchical tree structure to preserve
    parent-child relationships in hyperbolic space.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None, 
                 ontology_map: Optional[Dict[str, List[str]]] = None, idx_to_code: Optional[Dict[int, str]] = None):
        """
        Args:
            num_embeddings: Size of the dictionary of embeddings.
            embedding_dim: The size of each embedding vector.
            padding_idx: If given, pads the output with the embedding vector at padding_idx.
            padding_idx: If given, pads the output with the embedding vector at padding_idx.
            ontology_map: A dictionary mapping parent codes to children for initialization.
            idx_to_code: Optional dictionary mapping embedding indices to code strings.
                         Required for semantic initialization using ontology_map.
        """
        if torch is None:
            raise ImportError("NeuroFHIR requires 'torch' for PoincareEmbedding.")
            
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Standard lookup table, but we interpret weights as coordinates in B^d
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        
        # Initialize
        self.reset_parameters(ontology_map, idx_to_code)

    def reset_parameters(self, ontology_map: Optional[Dict[str, List[str]]] = None, 
                         idx_to_code: Optional[Dict[int, str]] = None):
        """
        Initialize weights using the ontology hierarchy.
        Nodes closer to the root (generic codes) are placed near the origin.
        Children are placed in sectors around the parent, further out.
        """
        # Standard small normal init for everything first
        nn.init.normal_(self.weight, 0, 0.01)
        
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)
        
        if ontology_map is None or idx_to_code is None:
            return

        # 1. Invert mapping to find parents for each code
        code_to_parent = {}
        for parent, children in ontology_map.items():
            for child in children:
                code_to_parent[child] = parent

        # 2. Iterate all indices and place them
        # Heuristic: Roots at radius 0.05, Children at radius +0.1 from parent angle
        with torch.no_grad():
            for idx, code in idx_to_code.items():
                if idx == self.padding_idx: continue
                
                # Check depth
                depth = 0
                curr = code
                while curr in code_to_parent:
                    depth += 1
                    curr = code_to_parent[curr]
                    if depth > 10: break # Safety
                
                # Simple radial placement based on depth
                # r = 0.1 * (depth + 1)
                # theta = hash(code) # Deterministic random angle
                
                r = 0.1 * (depth + 1)
                # Use simple math for angle to distribute
                angle = (hash(code) % 360) * (math.pi / 180.0)
                
                # 2D projection for first 2 dims, others noise
                self.weight[idx, 0] = r * math.cos(angle)
                self.weight[idx, 1] = r * math.sin(angle)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings. 
        Note: The returned vectors are in the Poincaré ball.
        """
        return torch.nn.functional.embedding(
            input, self.weight, self.padding_idx, max_norm=None, 
            norm_type=2.0, scale_grad_by_freq=False, sparse=False
        )

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Performs Möbius addition for B^d with curvature c=1.
        x +_m y = ( (1 + 2<x,y> + |y|^2)x + (1 - |x|^2)y ) / (1 + 2<x,y> + |x|^2|y|^2)
        """
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        
        denom = 1 + 2*xy + x2 * y2
        num = (1 + 2*xy + y2) * x + (1 - x2) * y
        
        # Avoid div by zero
        return num / (denom.clamp_min(1e-15))

    @classmethod
    def from_pretrained_ontology(cls, codes: List[str]) -> 'PoincareEmbedding':
        """
        Factory method to create an encoder populated with known codes.
        """
        # Simple factory
        vocab_size = len(codes)
        return cls(vocab_size, 16) # default dim 16
