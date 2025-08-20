import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
import base64
from io import BytesIO

# Step 1: Data Exploration

df_edges = pd.read_csv('edges.csv')

with open('metadata.txt', 'w') as f:
    f.write('Columns in edges.csv:\n')
    f.write(str(df_edges.columns.tolist()))
    f.write('\nFirst few rows of edges.csv:\n')
    f.write(str(df_edges.head()))

# Step 2 & 3: Full Analysis

G = nx.from_pandas_edgelist(df_edges, 'source', 'target')

edge_count = G.number_of_edges()
degrees = dict(G.degree())
highest_degree_node = max(degrees, key=degrees.get)
average_degree = sum(degrees.values()) / len(degrees)
density = nx.density(G)
shortest_path_alice_eve = nx.shortest_path_length(G, source='Alice', target='Eve')

# Network Graph
plt.figure(figsize=(8, 6))
positions = nx.spring_layout(G)
labels = {node: node for node in G.nodes()}  # Use node names as labels
nx.draw(G, pos=positions, with_labels=True, labels=labels, node_size=500, node_color='skyblue', font_size=12, font_weight='bold', edge_color='gray')
plt.title('Network Graph')

# Convert to base64 PNG
buffer = BytesIO()
plt.savefig(buffer, format="png")
buffer.seek(0)
network_graph_base64 = base64.b64encode(buffer.read()).decode('utf-8')
plt.close()

# Degree Histogram
plt.figure(figsize=(8, 6))
plt.hist(degrees.values(), color='green', bins=15)
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree Distribution')

# Convert to base64 PNG
buffer = BytesIO()
plt.savefig(buffer, format="png")
buffer.seek(0)
degree_histogram_base64 = base64.b64encode(buffer.read()).decode('utf-8')
plt.close()

result = {
    "edge_count": edge_count,
    "highest_degree_node": highest_degree_node,
    "average_degree": average_degree,
    "density": density,
    "shortest_path_alice_eve": shortest_path_alice_eve,
    "network_graph": network_graph_base64,
    "degree_histogram": degree_histogram_base64
}

with open('result.json', 'w') as f:
    json.dump(result, f)
