# NeuroFHIR: Geometric Deep Learning for Clinical Informatics

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPi](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/neurofhir/)

**NeuroFHIR** is a state-of-the-art library for transforming longitudinal FHIR (Fast Healthcare Interoperability Resources) data into **Temporal, Hierarchical, and Explainable Graph Tensors**. It enables the application of **Geometric Deep Learning (GDL)** to clinical informatics, moving beyond simple feature vectors to capture the true topological and temporal nature of patient health.

---

## 📐 Theoretical Foundations

NeuroFHIR is built upon three pillars of modern geometric learning:

### 1. Dynamic Temporal Graphs
Traditional models flatten patient history into static vectors, losing the sequential structure of disease progression. NeuroFHIR models a patient trajectory as a sequence of graph snapshots $\mathcal{G} = \{G_1, G_2, ..., G_T\}$.

- **Input**: Longitudinal FHIR Resources $R = \{r_1, ..., r_N\}$ with timestamps $t_i$.
- **Process**: Vectorized time-windowing (e.g., daily $\Delta t = 24h$) partitions $R$ into subsets $R_t$.
- **Output**: A **Dynamic Heterogeneous Graph** where nodes $V_t$ and edges $E_t$ evolve over time, compatible with **Temporal GNNs** (e.g., EvolveGCN, TGN).

### 2. Hyperbolic Geometry (The Poincaré Ball)
Medical ontologies (ICD-10, ATC, SNOMED) are inherently hierarchical trees. Embedding them into Euclidean space ($\mathbb{R}^n$) causes massive distortion. NeuroFHIR utilizes **Hyperbolic Space** ($\mathbb{H}^n$), which grows exponentially, naturally accommodating trees.

We model embeddings in the **Poincaré Ball** $(\mathbb{D}^n, g_x)$ with curvature $c=1$.
To perform updates (gradient descent) or semantic additions, we implement **Möbius Addition**:

$$
\mathbf{x} \oplus_c \mathbf{y} = \frac{(1 + 2c\langle\mathbf{x}, \mathbf{y}\rangle + c\|\mathbf{y}\|^2)\mathbf{x} + (1 - c\|\mathbf{x}\|^2)\mathbf{y}}{1 + 2c\langle\mathbf{x}, \mathbf{y}\rangle + c^2\|\mathbf{x}\|^2\|\mathbf{y}\|^2}
$$

This ensures that parent nodes (generic diseases) remain continuously closer to the origin, while child nodes (specific diagnoses) expand toward the boundary, preserving semantic hierarchy.

### 3. Causal Topology
Correlation is not causation. NeuroFHIR mines explicit causal links using clinical priors (logical implication).
- **Rule**: $A \xrightarrow{causes} B$ if $t_A < t_B$ and $Logic(A, B) = True$.
- **Structure**: Resulting structures form a **Directed Acyclic Graph (DAG)** layered on top of the associative graph, allowing models to learn counterfactual reasoning.

---

## 📦 Installation

```bash
pip install neurofhir
```

*Requirements: `polars` (fast data processing), `torch` (tensors), `networkx` (graph algorithms).*

---

## 📖 Detailed Usage Manual

### Use Case 1: Temporal Patient Modeling for Sepsis Prediction
*Goal: Convert a patient's raw FHIR history into a tensor stream for a Temporal GNN to predict septic shock 24h in advance.*

```python
from neurofhir import FHIRTemporalGraphBuilder
import torch

# 1. Initialize the Time Machine
# We want daily snapshots to capture the progression of vitals.
builder = FHIRTemporalGraphBuilder(time_window="1d")

# 2. Load Raw FHIR Data (simulated)
patient_history = [
    {"resourceType": "Patient", "id": "p1", "recordedDate": "2025-01-01T08:00:00Z"},
    # Day 1: Infection suspected
    {"resourceType": "Condition", "id": "c1", "code": {"text": "Sepsis"}, "recordedDate": "2025-01-01T09:00:00Z"},
    # Day 2: Antibiotics administered
    {"resourceType": "MedicationRequest", "id": "m1", "authoredOn": "2025-01-02T09:00:00Z"},
]

# 3. Build Dynamic Graph
# Returns an iterator of PyG HeteroData objects
snapshots = builder.build_snapshots(patient_history)

# 4. Integrate with PyTorch Geometric Temporal
temporal_signal = []
for snapshot in snapshots:
    # Snapshot contains:
    # - snapshot.x: Node features
    # - snapshot.edge_index: Adjacency matrices for that day
    temporal_signal.append(snapshot)

print(f"Generated {len(temporal_signal)} distinct time steps.")
# > Generated 2 distinct time steps.
```

### Use Case 2: Hierarchy-Aware Concept Embedding
*Goal: Embed a rare disease code such that it remains close to its broad category, enabling zero-shot generalization.*

```python
from neurofhir import PoincareEmbedding
import torch

# 1. Initialize Ontology Brain
# NeuroFHIR automatically places generic roots near the origin (0,0,0)
# and specific leaves near the boundary of the ball.
embedding_layer = PoincareEmbedding(num_embeddings=5000, embedding_dim=128)

# 2. Embed Codes (Indices mapped from your vocab)
# Index 0: "Infection" (Root)
# Index 1: "Viral Infection" (Child)
# Index 2: "COVID-19" (Leaf)
ids = torch.tensor([0, 1, 2])
vectors = embedding_layer(ids)

# 3. Calculate Hyperbolic Distance
# In hyperbolic space, distance grows exponentially as you move to edge
def hyperbolic_dist(u, v):
    sqdist = torch.sum((u - v) ** 2, dim=-1)
    return torch.acosh(1 + 2 * sqdist / ((1 - torch.norm(u)**2) * (1 - torch.norm(v)**2)))

dist_root_leaf = hyperbolic_dist(vectors[0], vectors[2])
print(f"Semantic Distance: {dist_root_leaf.item()}")
```

### Use Case 3: Explaining Outcomes with Causal Graphs
*Goal: Understand **why** a patient's fever dropped. was it natural recovery or the medication?*

```python
from neurofhir import CausalEdgeMiner

miner = CausalEdgeMiner()

# 1. Mine Relationships
# Auto-detects patterns like "Medication X given before Symptom Y improved"
edges_df = miner.mine_relationships(patient_history)

# 2. Inspect the "Why"
# Output: Source(Med_1) -> Relation(CAUSES_IMPROVEMENT) -> Target(Obs_Fever)
print(edges_df.select(["source", "relation", "target", "weight"]))

# 3. Export to DAG
G_causal = miner.create_dag(edges_df)
# Now you can filter your GNN's message passing to only propagate along causal paths!
```

---

## 🏥 Real-World Applications

1.  **Medication Repurposing**: Use **Hyperbolic Embeddings** to find drugs that are geometrically close to a disease target in the side-effect interaction space.
2.  **Patient Trajectory Forecasting**: Use **Temporal Graphs** to predict the next clinical event (e.g., readmission) by learning the vector field of the patient's state over time.
3.  **Counterfactual Treatment Analysis**: Use **Causal Graphs** to answer "What would have happened if we *didn't* give this drug?" by severing the incoming edges to the outcome node in the graph.

---

## 🤝 Contributing

We welcome contributions from researchers and clinicians.
1. Fork the repo.
2. Install dev dependencies: `pip install -e .[full]`
3. Run tests: `python tests/verify_modules.py`
4. Submit a PR.

## 📄 License

**AGPL-3.0**
Copyright (C) 2026 ATIL İHSAN YALI.
Commercial dual-licensing available upon request.
