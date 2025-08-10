import pandas as pd
import requests
import io
import json
import base64
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

async def run_analysis(questions: str, files: dict):
    """
    Parses questions and performs the data analysis task.
    """
    # Simple logic to determine the task based on keywords in the questions
    if "wikipedia" in questions.lower() and "highest-grossing" in questions.lower():
        return analyze_wikipedia_films()
    elif "indian high court" in questions.lower() and "duckdb" in questions.lower():
        # This example is not implemented, as it requires a separate logic.
        return {"error": "DuckDB analysis not implemented in this example."}
    else:
        if "data.csv" in files:
            # Placeholder for handling generic CSV file analysis
            df = pd.read_csv(io.BytesIO(files["data.csv"]))
            return {"questions_received": questions, "data_columns": list(df.columns)}
        else:
            return {"error": "Could not determine the task or find data."}

def analyze_wikipedia_films():
    """
    Handles the Wikipedia highest-grossing films analysis.
    """
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    
    try:
        # Scrape the table from Wikipedia
        tables = pd.read_html(url)
        df = tables[0]

        # Clean the data
        df.rename(columns={'Worldwide gross': 'Worldwide gross ($)', 'Ref.': 'Ref', 'Year': 'Year', 'Film': 'Film'}, inplace=True)
        df['Worldwide gross ($)'] = df['Worldwide gross ($)'].astype(str).str.replace(r'[\$,]', '', regex=True).astype(float)
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
        
        # Q1: How many $2 bn movies were released before 2000?
        q1_answer = len(df[(df['Worldwide gross ($)'] >= 2e9) & (df['Year'] < 2000)])

        # Q2: Which is the earliest film that grossed over $1.5 bn?
        over_1_5b = df[df['Worldwide gross ($)'] >= 1.5e9].sort_values(by='Year')
        q2_answer = over_1_5b.iloc[0]['Film']

        # Clean 'Rank' and 'Peak' for correlation
        df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
        df['Peak'] = pd.to_numeric(df['Peak'], errors='coerce')
        df.dropna(subset=['Rank', 'Peak'], inplace=True)
        
        # Q3: What's the correlation between the Rank and Peak?
        q3_answer = df['Rank'].corr(df['Peak'])

        # Q4: Generate scatterplot
        q4_answer = generate_scatterplot(df['Rank'], df['Peak'])
        
        return [
            q1_answer,
            q2_answer,
            round(q3_answer, 6),
            q4_answer
        ]

    except Exception as e:
        return {"error": f"An error occurred during Wikipedia analysis: {e}"}

def generate_scatterplot(x_data, y_data):
    """
    Creates a scatterplot with a regression line and returns it as a Base64 data URI.
    """
    plt.figure(figsize=(6, 4))
    plt.scatter(x_data, y_data)
    
    # Add a regression line
    x_data_reshaped = x_data.values.reshape(-1, 1)
    model = LinearRegression().fit(x_data_reshaped, y_data)
    regression_line = model.predict(x_data_reshaped)
    plt.plot(x_data, regression_line, color='red', linestyle='dotted')
    
    plt.xlabel('Rank')
    plt.ylabel('Peak')
    plt.title('Rank vs Peak')
    
    # Save the plot to a BytesIO object
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close()
    img_buffer.seek(0)
    
    # Encode to Base64
    base64_encoded_img = base64.b64encode(img_buffer.read()).decode('utf-8')
    
    return f"data:image/png;base64,{base64_encoded_img}"