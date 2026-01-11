import pytest
import torch
from neurofhir.hyperbolic_encoder import PoincareEmbedding

def test_embedding_init():
    emb = PoincareEmbedding(num_embeddings=10, embedding_dim=4)
    assert emb.weight.shape == (10, 4)

def test_mobius_add():
    emb = PoincareEmbedding(num_embeddings=10, embedding_dim=4)
    x = torch.zeros(1, 4)
    y = torch.zeros(1, 4)
    res = emb.mobius_add(x, y)
    assert torch.allclose(res, x) # 0+0=0
