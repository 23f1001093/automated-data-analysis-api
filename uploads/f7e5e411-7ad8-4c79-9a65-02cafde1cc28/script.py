import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Load the dataframe.
df = pd.read_csv('edges.csv')

# Create the graph.
G = nx.from_pandas_edgelist(df, 'source', 'target')

# 1. Number of edges.
edge_count = G.number_of_edges()

# 2. Node with highest degree.
degrees = dict(G.degree())
highest_degree_node = max(degrees, key=degrees.get)

# 3. Average degree.
average_degree = sum(degrees.values()) / len(degrees)

# 4. Network density.
density = nx.density(G)

# 5. Shortest path between Alice and Eve.
try:
    shortest_path_alice_eve = nx.shortest_path_length(G, source='Alice', target='Eve')
except nx.NetworkXNoPath:
    shortest_path_alice_eve = -1  # Or handle the case where no path exists as needed

# 6. Network graph.
plt.figure(figsize=(8, 6))
plt.title('Network Graph')
pos = nx.spring_layout(G) #Kamada Kawai layout requires scipy
labels = {n: n for n in G.nodes}

# Modified draw function to make image generation work.
def my_draw_networkx_edge_labels(G, pos, edge_labels=None, label_pos=0.5, font_size=10,
                           font_color="k", font_family="sans-serif",
                           font_weight="normal", alpha=None, bbox=None,
                           horizontalalignment="center",
                           verticalalignment="center", ax=None, rotate=True,
                           clip_on=True):
    from numbers import Number
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cbook as cb
    except ImportError:
        raise ImportError("Matplotlib required for draw()")
    try:
        from matplotlib.transforms import Affine2D
    except ImportError:
        raise ImportError("Matplotlib >= 1.2 required for draw()")
    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = dict(zip(G.edges(), G.edges()))
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (x1 * label_pos + x2 * (1.0 - label_pos),
                  y1 * label_pos + y2 * (1.0 - label_pos))

        if rotate:
            # in polar coordinates, the angle 'rad' is the angle
            # of the edge. This is the angle of rotation needed.
            rad = np.arctan2(y2 - y1, x2 - x1)

        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same
        t = ax.text(x, y, label, size=font_size, color=font_color,
                     family=font_family, weight=font_weight, alpha=alpha,
                     horizontalalignment=horizontalalignment,
                     verticalalignment=verticalalignment,
                     transform=ax.transData, clip_on=clip_on, bbox=bbox)

import numpy as np
my_draw_networkx_edge_labels(G, pos)
nx.draw(G, pos=pos, with_labels=True, node_color="lightblue", node_size=700, font_family="sans-serif", font_size=10)

img = BytesIO()
plt.savefig(img, format='png')
network_graph = base64.b64encode(img.getvalue()).decode()
plt.close()

# 7. Degree histogram.
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
plt.figure(figsize=(8, 6))
plt.title('Degree Distribution')
plt.xlabel('Degree')
plt.ylabel('Count')
plt.hist(degree_sequence, color='green')
img = BytesIO()
plt.savefig(img, format='png')
degree_histogram = base64.b64encode(img.getvalue()).decode()
plt.close()

result = {
    'edge_count': int(edge_count),
    'highest_degree_node': highest_degree_node,
    'average_degree': float(average_degree),
    'density': float(density),
    'shortest_path_alice_eve': int(shortest_path_alice_eve),
    'network_graph': network_graph,
    'degree_histogram': degree_histogram
}

import json
with open('result.json', 'w') as f:
    json.dump(result, f)