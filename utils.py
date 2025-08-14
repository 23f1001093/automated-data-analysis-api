import pandas as pd
import requests
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import re
import networkx as nx
from sklearn.linear_model import LinearRegression

def _get_and_process_wiki_table(url):
    try:
        response = requests.get(url, timeout=20)
        tables = pd.read_html(response.text)
        if not tables:
            return "No table found"
        df = tables[0]
        df.columns = [col.replace('[a]', '').strip() for col in df.columns]
        df['Worldwide gross'] = pd.to_numeric(df['Worldwide gross'].str.replace('$', '', regex=False).str.replace(',', '', regex=False), errors='coerce')
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
        df['Peak'] = pd.to_numeric(df['Peak'], errors='coerce')
        return df
    except Exception as e:
        return f"Error during data processing: {str(e)}"

def _generate_scatterplot(df):
    df_cleaned = df.dropna(subset=['Rank', 'Peak'])
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Rank', y='Peak', data=df_cleaned)
    X = df_cleaned[['Rank']].values
    y = df_cleaned['Peak'].values
    reg = LinearRegression().fit(X, y)
    plt.plot(X, reg.predict(X), color='red', linestyle=':')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

def _handle_wiki_task(questions_text, df):
    answers = []
    questions = re.findall(r"\d+\.\s*(.*?)(?=\n\d+\.|$)", questions_text, flags=re.DOTALL)
    for q in questions:
        q_lower = q.lower()
        if "2 bn movies" in q_lower:
            q1_answer = len(df[(df['Worldwide gross'] >= 2000000000) & (df['Year'] < 2000)])
            answers.append(str(q1_answer))
        elif "earliest film" in q_lower:
            q2_df = df[df['Worldwide gross'] >= 1500000000].sort_values(by='Year')
            q2_answer = q2_df.iloc[0]['Title'] if not q2_df.empty else "No such film"
            answers.append(q2_answer)
        elif "correlation" in q_lower:
            df_cleaned = df.dropna(subset=['Rank', 'Peak'])
            correlation = df_cleaned['Rank'].corr(df_cleaned['Peak'])
            answers.append(float(np.round(correlation, 6)))
        elif "scatterplot" in q_lower:
            answers.append(_generate_scatterplot(df))
        else:
            answers.append(None)
    return answers

def _handle_network_task(questions_text, file_paths):
    answers = {}
    
    # Load the edges.csv file
    edges_file = file_paths.get('edges.csv')
    if not edges_file:
        return {"error": "edges.csv is required"}

    try:
        df_edges = pd.read_csv(edges_file)
        G = nx.from_pandas_edgelist(df_edges, 'Source', 'Target')
    except Exception as e:
        return {"error": f"Failed to load or parse edges.csv: {str(e)}"}
    
    # Calculate metrics
    answers['edge_count'] = G.number_of_edges()
    degrees = dict(G.degree())
    if degrees:
        answers['highest_degree_node'] = max(degrees, key=degrees.get)
    else:
        answers['highest_degree_node'] = None
    
    answers['average_degree'] = float(np.round(np.mean(list(degrees.values())), 6))
    
    n = G.number_of_nodes()
    answers['density'] = float(np.round(nx.density(G), 6))
    
    try:
        answers['shortest_path_alice_eve'] = nx.shortest_path_length(G, source='Alice', target='Eve')
    except nx.NetworkXNoPath:
        answers['shortest_path_alice_eve'] = -1
    except nx.NodeNotFound:
        answers['shortest_path_alice_eve'] = -1
    
    # Generate visualizations
    answers['network_graph'] = _generate_network_graph(G)
    answers['degree_histogram'] = _generate_degree_histogram(G)
    
    return answers

def _generate_network_graph(G):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10, font_weight='bold')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

def _generate_degree_histogram(G):
    degrees = [G.degree(n) for n in G.nodes()]
    degree_counts = nx.degree_histogram(G)
    
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(degree_counts)), degree_counts, width=0.8, color='green')
    plt.title("Degree Histogram")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.xticks(range(len(degree_counts)))
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

def answer_all_questions(questions_text, file_paths):
    # Detect the task type
    if "wikipedia" in questions_text.lower() or "http" in questions_text.lower():
        # Handle the Wikipedia task
        url_match = re.search(r"http[s]?://\S+", questions_text)
        if url_match:
            page_url = url_match.group(0)
            df = _get_and_process_wiki_table(page_url)
            if isinstance(df, str):
                return [df] * 4
            return _handle_wiki_task(questions_text, df)
        else:
            return ["No URL found"] * 4
    elif "network" in questions_text.lower() or "edges.csv" in questions_text.lower():
        # Handle the network analysis task
        return _handle_network_task(questions_text, file_paths)
    else:
        # Default response for unknown tasks
        return ["Not implemented"] * 4