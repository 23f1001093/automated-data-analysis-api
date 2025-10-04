import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Load the dataframe.
df = pd.read_csv('edges.csv')

# Create the graph.
G = nx.from_pandas_edgelist(df, 'source', 'target')

# 1. Edge count
edge_count = G.number_of_edges()

# 2. Highest degree node
degrees = dict(G.degree())
highest_degree_node = max(degrees, key=degrees.get)

# 3. Average degree
average_degree = sum(degrees.values()) / len(degrees)

# 4. Density
density = nx.density(G)

# 5. Shortest path between Alice and Eve
try:
    shortest_path_alice_eve = nx.shortest_path_length(G, source='Alice', target='Eve')
except nx.NetworkXNoPath:
    shortest_path_alice_eve = -1

# 6. Network graph
plt.figure(figsize=(8, 6))
layout = nx.spring_layout(G)
labels = {node: node for node in G.nodes()} 

# Check if the graph is very large, and sample if so to control image size and plotting time.
if len(G.nodes) > 300:   # Sample at most 300 nodes
    sampled_nodes = list(G.nodes)[:300]
    subgraph = G.subgraph(sampled_nodes)
    nx.draw(subgraph, layout, labels=labels, with_labels=True, node_size=500, node_color="skyblue", font_size=10)
else:
    nx.draw(G, layout, labels=labels, with_labels=True, node_size=500, node_color="skyblue", font_size=10)

buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
network_graph_base64 = base64.b64encode(buffer.read()).decode('utf-8')
plt.close()

# 7. Degree histogram
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
plt.figure(figsize=(8, 6))
plt.hist(degree_sequence, bins=range(min(degree_sequence), max(degree_sequence) + 2), align='left', rwidth=0.8, color='green')
plt.xlabel("Degree")
plt.ylabel("Number of Nodes")
plt.title("Degree Distribution")

buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
degree_histogram_base64 = base64.b64encode(buffer.read()).decode('utf-8')
plt.close()

# Save the results to result.json
results = {
    'edge_count': int(edge_count),
    'highest_degree_node': highest_degree_node,
    'average_degree': float(average_degree),
    'density': float(density),
    'shortest_path_alice_eve': int(shortest_path_alice_eve) if shortest_path_alice_eve != -1 else None,
    'network_graph': network_graph_base64,
    'degree_histogram': degree_histogram_base64
}

import json
with open('result.json', 'w') as f:
    json.dump(results, f)