import json
import os
import networkx as nx
import matplotlib.pyplot as plt

def visualize_snapshot(json_path, snapshot_index=-1):
    """
    Visualizes a specific snapshot from the JSON output.
    """
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        snapshots = json.load(f)

    if not snapshots:
        print("No snapshots found in file.")
        return

    # Select snapshot (default to finding one with max edges to be interesting)
    if snapshot_index == -1:
        # Helper to count edges
        def get_edge_count(s):
            edges = s.get("edges", {})
            return sum(len(v["src"]) for v in edges.values())
            
        # Find snapshot with max edges
        max_snap = max(snapshots, key=get_edge_count)
        snapshot_index = snapshots.index(max_snap)
    
    snap = snapshots[snapshot_index]
    print(f"Visualizing Snapshot {snapshot_index} (Timestamp: {snap.get('timestamp')})")

    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes (from num_nodes counts)
    # The JSON stores counts like {"Condition": 5}, so we'll create placeholder nodes
    # e.g. "Condition_1", "Condition_2", etc. or just one abstract node per type?
    # Better: The edges contain actual references. Let's build from edges.
    
    edges_data = snap.get("edges", {})
    if not edges_data:
        print("No edges in this snapshot.")
        return

    edge_labels = {}
    
    for rel_key, edge_info in edges_data.items():
        # rel_key is like "Patient__refers_to__Organization"
        # src_list, dst_list are lists of indices.
        # This is a bit abstract because we mapped IDs to Integers in the builder.
        # We don't have the original strings here easily unless we reconstruct them.
        # But for visualization, we can just show the structure of connections.
        
        try:
            src_type, rel_name, dst_type = rel_key.split("__")
        except ValueError:
             src_type, rel_name, dst_type = ("?", rel_key, "?")

        src_indices = edge_info.get("src", [])
        dst_indices = edge_info.get("dst", [])

        for s, d in zip(src_indices, dst_indices):
            u = f"{src_type}_{s}"
            v = f"{dst_type}_{d}"
            
            G.add_node(u, color=get_node_color(src_type))
            G.add_node(v, color=get_node_color(dst_type))
            G.add_edge(u, v)
            # Only label a few to avoid clutter?
            # edge_labels[(u, v)] = rel_name

    # Draw
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw nodes
    node_colors = [G.nodes[n].get('color', 'gray') for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, alpha=0.9)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrowstyle='->', arrowsize=10)
    
    plt.title(f"Patient Graph Snapshot at {snap.get('timestamp')}")
    plt.axis('off')
    
    output_img = "graph_visualization.png"
    plt.savefig(output_img, dpi=150, bbox_inches='tight')
    print(f"Graph image saved to {output_img}")
    plt.show()

def get_node_color(node_type):
    colors = {
        "Patient": "blue",
        "Practitioner": "green",
        "Organization": "red",
        "Encounter": "orange",
        "Condition": "purple",
        "Observation": "cyan",
        "MedicationRequest": "magenta",
        "Procedure": "pink"
    }
    return colors.get(node_type, "gray")

if __name__ == "__main__":
    # Look in the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(script_dir, "output_graph_sample.json")
    
    if not os.path.exists(json_file):
        # Fallback to local if running from that dir
        json_file = "output_graph_sample.json"
        
    visualize_snapshot(json_file)
