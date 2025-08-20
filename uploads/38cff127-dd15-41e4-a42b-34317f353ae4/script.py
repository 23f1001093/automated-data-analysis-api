import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
import json

df = pd.read_csv('edges.csv')

# Create the graph
graph = nx.from_pandas_edgelist(df, 'source', 'target', create_using=nx.Graph())

# 1. Edge Count
edge_count = graph.number_of_edges()

# 2. Highest Degree Node
degrees = dict(graph.degree())
highest_degree_node = max(degrees, key=degrees.get)

# 3. Average Degree
average_degree = sum(degrees.values()) / len(degrees)

# 4. Network Density
density = nx.density(graph)

# 5. Shortest Path between Alice and Eve
try:
    shortest_path_alice_eve = nx.shortest_path_length(graph, source='Alice', target='Eve')
except nx.NetworkXNoPath:
    shortest_path_alice_eve = -1  # Or handle the case where no path exists

# 6. Network Graph
plt.figure(figsize=(8, 6))
layout = nx.spring_layout(graph)
labels = {node: node for node in graph.nodes()}  # Use node names as labels
nx.draw(graph, layout, with_labels=True, labels=labels, node_color='skyblue', edge_color='gray', width=1, node_size=500)
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
network_graph = base64.b64encode(buf.read()).decode('utf-8')
buf.close()
plt.clf()

# 7. Degree Histogram
degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
degree_counts = {}
for degree in degree_sequence:
    if degree not in degree_counts:
        degree_counts[degree] = 0
    degree_counts[degree] += 1

degrees = list(degree_counts.keys())
counts = list(degree_counts.values())

plt.figure(figsize=(8, 6))
plt.bar(degrees, counts, color='green')
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree Distribution')
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
degree_histogram = base64.b64encode(buf.read()).decode('utf-8')
buf.close()

# Save to result.json
results = {
    "edge_count": int(edge_count),
    "highest_degree_node": highest_degree_node,
    "average_degree": float(average_degree),
    "density": float(density),
    "shortest_path_alice_eve": int(shortest_path_alice_eve),
    "network_graph": network_graph,
    "degree_histogram": degree_histogram,
}

with open('result.json', 'w') as f:
    json.dump(results, f)