import sys
import os
import torch
# Add parent directory to path to allow importing neurofhir from source
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from neurofhir.hyperbolic_encoder import PoincareEmbedding
except ImportError:
    print("Error: Could not import PoincareEmbedding. Ensure dependency structure is correct.")
    sys.exit(1)

def main():
    print("=== Example 2: Hierarchy-Aware Concept Embedding ===")
    
    # 1. Initialize Ontology Brain
    # NeuroFHIR automatically places generic roots near the origin (0,0,0)
    # and specific leaves near the boundary of the ball.
    # Note: We need a dummy ontology map for this example.
    ontology_map = {
        "A00": ["A01", "A02"],     # A00 is parent of A01, A02
        "A01": ["A01.1", "A01.2"], # A01 is parent of ...
    }
    
    # Map codes to indices
    idx_to_code = {
        0: "A00", 
        1: "A01", 
        2: "A02", 
        3: "A01.1", 
        4: "A01.2"
    }
    
    print("Initializing PoincareEmbedding...")
    embedding_layer = PoincareEmbedding(
        num_embeddings=100, 
        embedding_dim=16, # Low dimension for hyperbolic space!
        ontology_map=ontology_map,
        idx_to_code=idx_to_code
    )

    # 2. Embed Codes
    # Index 0: "A00" (Root)
    # Index 1: "A01" (Child)
    ids = torch.tensor([0, 1])
    vectors = embedding_layer(ids)
    
    print(f"Embedded vectors shape: {vectors.shape}")

    # 3. Calculate Hyperbolic Distance
    # In hyperbolic space, distance grows exponentially as you move to edge
    dist = embedding_layer.dist(vectors[0], vectors[1])
    print(f"Hyperbolic Distance between Root (A00) and Child (A01): {dist.item():.4f}")
    
    # Compare with a distant node if we had one...

if __name__ == "__main__":
    main()
