import os
import io
import json
import base64
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
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

        # Manually parse the table rows and columns
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

# ---------- LLM-based Analyzer ----------
async def analyze_questions(questions_text: str, attachments: dict):
    """
    Uses an LLM to dynamically interpret and answer questions based on a prompt.
    """
    # Convert attachments to a small sample text for the LLM prompt
    attachment_texts = {}
    for filename, content in attachments.items():
        if filename.endswith(".csv"):
            # Provide the LLM with the full CSV content for analysis
            attachment_texts[filename] = content.decode("utf-8")
        elif filename.endswith(".json"):
            attachment_texts[filename] = content.decode("utf-8")
        else:
            attachment_texts[filename] = f"<{len(content)} bytes>"

    # Compose the LLM prompt with the questions and file content
    prompt = f"""
You are a data analyst agent. Your task is to analyze the data and answer the questions provided.

**Crucially, your final output must be a single, raw JSON array of strings, with no other text, comments, or explanations.**

Questions:
{questions_text}

Files (sample content if too large):
{attachment_texts}
"""
    # Call the LLM via AI Pipe using a valid model name
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer_text = response.choices[0].message.content

    # Check if the output is a markdown code block and extract the content
    if answer_text.startswith("```json") and answer_text.endswith("```"):
        # Strip the markdown tags
        answer_text = answer_text.strip()[7:-3].strip()

    # Parse the JSON response safely
    try:
        answers = json.loads(answer_text)
        # Ensure the output is a list, as expected
        if not isinstance(answers, list):
            raise ValueError("LLM response is not a JSON array.")
    except (json.JSONDecodeError, ValueError) as e:
        # Fallback to a failure message if the LLM doesn't produce valid JSON
        answers = [f"Error: LLM failed to produce a valid JSON array. Details: {e}", answer_text]

    return answers

# ---------- FastAPI Endpoint ----------
@app.post("/api/")
async def process_api(request: Request):
    """
    Accepts a POST request with 'questions.txt' and optional attachments.
    """
    form = await request.form()
    questions_text = None
    attachments = {}

    for key, file in form.items():
        if hasattr(file, "filename"):
            content = await file.read()
            if file.filename.lower() == "questions.txt":
                questions_text = content.decode("utf-8")
            else:
                attachments[file.filename] = content

    if questions_text is None:
        return JSONResponse(content={"error": "questions.txt file is required"}, status_code=400)
    
    # Check questions_text for URL and perform scraping
    import re
    url_match = re.search(r"https?://\S+", questions_text)
    if url_match:
        url_to_scrape = url_match.group(0)
        scraped_data_csv, error_msg = get_wikipedia_table_data(url_to_scrape)
        if scraped_data_csv:
            attachments["scraped_data.csv"] = scraped_data_csv.encode("utf-8")
        else:
            return JSONResponse(content={"error": error_msg}, status_code=500)

    result = await analyze_questions(questions_text, attachments)
    return JSONResponse(content=result)

# ---------- Run with Uvicorn ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
