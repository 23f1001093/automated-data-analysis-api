import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import base64
import json

# Load the edge list
edges = pd.read_csv('edges.csv')

# Create the graph
G = nx.from_pandas_edgelist(edges, source='source', target='target')

# 1. Number of edges
edge_count = G.number_of_edges()

# 2. Node with highest degree
degrees = dict(G.degree())
highest_degree_node = max(degrees, key=degrees.get)

# 3. Average degree
average_degree = sum(degrees.values()) / len(degrees)

# 4. Network density
density = nx.density(G)

# 5. Shortest path between Alice and Eve
shortest_path_alice_eve = nx.shortest_path_length(G, source='Alice', target='Eve')

# 6. Network graph
plt.figure(figsize=(8, 6))
layout = nx.spring_layout(G, seed=42) # Seed for consistent layout
labels = {n: str(n) for n in G.nodes()}
nx.draw(G, layout, with_labels=True, labels=labels, node_color="lightblue", node_size=500)
plt.savefig('network_graph.png')
with open('network_graph.png', "rb") as img_file:
    network_graph = base64.b64encode(img_file.read()).decode('utf-8')
plt.clf()

# 7. Degree histogram
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
plt.figure(figsize=(8, 6))
plt.bar(range(len(degree_sequence)), degree_sequence, color='green')
plt.xlabel("Nodes")
plt.ylabel("Degree")
plt.title("Degree Distribution")
node_labels = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
plt.xticks(range(len(degree_sequence)), node_labels, rotation=45, ha='right') # Rotate x-axis labels
plt.tight_layout() # Adjust layout to prevent labels from overlapping

# Ensure the plot is saved before encoding to base64
plt.savefig('degree_histogram.png')

with open('degree_histogram.png', "rb") as img_file:
    degree_histogram = base64.b64encode(img_file.read()).decode('utf-8')

plt.clf() # Clear the current plot to avoid overlap with future plots

# Save the results to result.json
result = {
    "edge_count": edge_count,
    "highest_degree_node": highest_degree_node,
    "average_degree": average_degree,
    "density": density,
    "shortest_path_alice_eve": shortest_path_alice_eve,
    "network_graph": network_graph,
    "degree_histogram": degree_histogram
}
with open('result.json', 'w') as f:
    json.dump(result, f)