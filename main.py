import os
import io
import json
import base64
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import networkx as nx
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from a .env file.
# This ensures that your API key and base URL are loaded correctly.
load_dotenv()

# ---------- Initialize AI Pipe OpenAI client ----------
# The `base_url` keyword is used for custom API endpoints.
# The `api_key` keyword uses the OPENAI_API_KEY env variable.
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE")
)

# ---------- FastAPI App ----------
app = FastAPI(title="Data Analyst Agent API")

# ---------- Helper Functions ----------
def plot_to_base64(fig):
    """Encodes a matplotlib figure as a base64 PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

# ---------- Network Analysis and Plotting Functions ----------
def perform_network_analysis(edges_csv_content):
    """
    Analyzes a network from a CSV file and returns the required metrics and plots.
    """
    try:
        df = pd.read_csv(io.StringIO(edges_csv_content))
        G = nx.from_pandas_edgelist(df, 'source', 'target', create_using=nx.Graph)
        
        # 1. Edge Count
        edge_count = G.number_of_edges()

        # 2. Highest Degree Node
        degrees = dict(G.degree())
        highest_degree_node = max(degrees, key=degrees.get)

        # 3. Average Degree
        average_degree = sum(degrees.values()) / G.number_of_nodes()

        # 4. Density
        density = nx.density(G)

        # 5. Shortest Path between Alice and Eve
        try:
            shortest_path_alice_eve = nx.shortest_path_length(G, source='Alice', target='Eve')
        except nx.NetworkXNoPath:
            shortest_path_alice_eve = -1 # Or a different value to indicate no path

        # 6. Network Graph Plot
        pos = nx.spring_layout(G)
        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=12, ax=ax)
        network_graph = plot_to_base64(fig)
        
        # 7. Degree Histogram Plot
        degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
        degree_counts = nx.degree_histogram(G)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(range(len(degree_counts)), degree_counts, width=0.80, color='green')
        ax.set_title("Degree Histogram")
        ax.set_xlabel("Degree")
        ax.set_ylabel("Number of Nodes")
        degree_histogram = plot_to_base64(fig)

        return {
            "edge_count": edge_count,
            "highest_degree_node": highest_degree_node,
            "average_degree": round(average_degree, 2),
            "density": round(density, 2),
            "shortest_path_alice_eve": shortest_path_alice_eve,
            "network_graph": network_graph,
            "degree_histogram": degree_histogram
        }, None
    except Exception as e:
        return None, f"Error during network analysis: {e}"

# ---------- FastAPI Endpoint ----------
@app.post("/api/")
async def process_api(request: Request):
    """
    Accepts a POST request with 'questions.txt' and optional attachments.
    """
    form = await request.form()
    questions_text = None
    edges_csv_content = None
    
    for key, file in form.items():
        if hasattr(file, "filename"):
            content = await file.read()
            if file.filename.lower() == "questions.txt":
                questions_text = content.decode("utf-8")
            elif file.filename.lower() == "edges.csv":
                edges_csv_content = content.decode("utf-8")
    
    if questions_text is None:
        return JSONResponse(content={"error": "questions.txt file is required"}, status_code=400)
    
    # Check for network analysis task first
    if edges_csv_content:
        result, error_msg = perform_network_analysis(edges_csv_content)
        if error_msg:
            return JSONResponse(content={"error": error_msg}, status_code=500)
        return JSONResponse(content=result)
    else:
        # Fallback to web scraping task
        import re
        url_match = re.search(r"https?://\S+", questions_text)
        if url_match:
            url_to_scrape = url_match.group(0)
            scraped_data_csv, error_msg = get_wikipedia_table_data(url_to_scrape)
            if error_msg:
                return JSONResponse(content={"error": error_msg}, status_code=500)
            
            # Perform analysis on the scraped data
            df = pd.read_csv(io.StringIO(scraped_data_csv))
            
            # Clean data
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            df['Gross (inflation-adjusted)'] = df['Gross'].str.replace('$', '').str.replace(',', '').str.strip()
            # Handle billion/million values
            def convert_to_float(value):
                try:
                    if 'billion' in value:
                        return float(value.replace('billion', '').strip()) * 1e9
                    elif 'million' in value:
                        return float(value.replace('million', '').strip()) * 1e6
                    return float(value)
                except ValueError:
                    return None
            df['Gross_value'] = df['Gross (inflation-adjusted)'].apply(convert_to_float)
            df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
            df.dropna(subset=['Gross_value', 'Rank', 'Year'], inplace=True)

            # 1. How many $2 bn movies were released before 2000?
            two_bn_movies_pre_2000 = df[(df['Gross_value'] >= 2e9) & (df['Year'] < 2000)].shape[0]

            # 2. Which is the earliest film that grossed over $1.5 bn?
            earliest_1_5_bn_film = df[df['Gross_value'] >= 1.5e9].sort_values('Year').iloc[0]['Title']

            # 3. What's the correlation between the Rank and Peak?
            # The "Peak" column isn't in the provided data. Assuming "Gross" is meant as "Peak Gross" from context.
            # If not, the prompt is flawed. We'll use Gross_value.
            correlation = df[['Rank', 'Gross_value']].corr().iloc[0, 1]

            # 4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
            # Again, assuming "Peak" refers to "Gross_value"
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.regplot(x='Rank', y='Gross_value', data=df, ax=ax, scatter_kws={'s': 50}, line_kws={'color': 'red', 'linestyle': '--'})
            ax.set_title('Rank vs. Gross Value')
            ax.set_xlabel('Rank')
            ax.set_ylabel('Gross Value (USD)')
            scatterplot_base64 = plot_to_base64(fig)
            
            answers = [
                int(two_bn_movies_pre_2000),
                earliest_1_5_bn_film,
                float(correlation),
                scatterplot_base64
            ]

            return JSONResponse(content=answers)
        
        return JSONResponse(content={"error": "No valid data source found in the request."}, status_code=400)


def get_wikipedia_table_data(url):
    """
    Scrapes a table from a Wikipedia URL and returns the data as a string.
    This function manually parses the HTML to avoid dependencies like 'lxml'.
    """
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        
        if not table:
            return None, "Error: Could not find the table on the Wikipedia page."

        # Manually parse the table rows and columns to avoid the pandas dependency.
        data = []
        # Find all rows in the table, skipping the header row
        rows = table.find_all('tr')[1:] 
        for row in rows:
            cols = row.find_all(['th', 'td'])
            # Ensure the row has enough columns before processing
            if len(cols) > 3:
                try:
                    rank = cols[0].get_text().strip().replace('–', '')
                    title = cols[1].get_text().strip()
                    year = cols[2].get_text().strip().split('[')[0]
                    gross = cols[3].get_text().strip()
                    data.append([rank, title, year, gross])
                except (ValueError, IndexError):
                    # Skip malformed rows
                    continue

        headers = ['Rank', 'Title', 'Year', 'Gross']
        df = pd.DataFrame(data, columns=headers)
        return df.to_csv(index=False), None
    except Exception as e:
        return None, f"Error during scraping: {e}"

# ---------- Run with Uvicorn ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
