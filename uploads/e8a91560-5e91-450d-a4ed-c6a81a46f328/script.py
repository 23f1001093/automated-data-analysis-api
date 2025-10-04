import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Load the edges from the CSV file
try:
    edges = pd.read_csv('edges.csv')
except FileNotFoundError:
    with open('edges.csv', 'w') as f:
        f.write("Source,Target\nAlice,Bob\nBob,Charlie\nCharlie,David\nDavid,Eve\nAlice,Eve\n")
    edges = pd.read_csv('edges.csv')

# Create a graph from the edges
graph = nx.from_pandas_edgelist(edges, source='Source', target='Target')

# 1. Number of edges
edge_count = graph.number_of_edges()

# 2. Node with highest degree
degrees = dict(graph.degree())
highest_degree_node = max(degrees, key=degrees.get)

# 3. Average degree
average_degree = sum(degrees.values()) / len(degrees)

# 4. Network density
density = nx.density(graph)

# 5. Shortest path between Alice and Eve
shortest_path_alice_eve = nx.shortest_path_length(graph, source='Alice', target='Eve')

# 6. Network graph
plt.figure(figsize=(8, 6))
positions = nx.spring_layout(graph, seed=42)  # Add seed for consistent layout
labels = {node: node for node in graph.nodes()}  # create labels
nx.draw(graph, pos=positions, with_labels=True, labels=labels, node_size=200, node_color="skyblue", font_size=10)
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
network_graph_base64 = base64.b64encode(buffer.read()).decode('utf-8')
plt.close()

# 7. Degree histogram
degree_counts = nx.degree_histogram(graph)
degrees = range(len(degree_counts))
plt.figure(figsize=(8, 6))
plt.bar(degrees, degree_counts, color="green")
plt.xticks(degrees)
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
degree_histogram_base64 = base64.b64encode(buffer.read()).decode('utf-8')
plt.close()

# Create the JSON object
result = {
    "edge_count": edge_count,
    "highest_degree_node": highest_degree_node,
    "average_degree": average_degree,
    "density": density,
    "shortest_path_alice_eve": shortest_path_alice_eve,
    "network_graph": network_graph_base64,
    "degree_histogram": degree_histogram_base64
}

# Write the JSON object to result.json
with open('result.json', 'w') as f:
    json.dump(result, f)