import os
import io
import json
import base64
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def scatterplot_with_regression(x, y, xlabel="", ylabel=""):
    """Creates a scatterplot with a regression line and returns it as a base64 string."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots()
    sns.scatterplot(x=x, y=y, ax=ax)
    sns.regplot(x=x, y=y, ax=ax, scatter=False, color="red", line_kws={"linestyle":"--"})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return plot_to_base64(fig)

# ---------- LLM-based Analyzer ----------
async def analyze_questions(questions_text: str, attachments: dict):
    """
    Uses an LLM to dynamically interpret and answer questions based on a prompt.
    """
    # Convert attachments to a small sample text for the LLM prompt
    attachment_texts = {}
    for filename, content in attachments.items():
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
            attachment_texts[filename] = df.head(50).to_csv(index=False)
        elif filename.endswith(".json"):
            attachment_texts[filename] = content.decode("utf-8")
        else:
            attachment_texts[filename] = f"<{len(content)} bytes>"

    # Compose the LLM prompt with the questions and file content
    prompt = f"""
You are a data analyst agent. Answer the questions below using Python, pandas, matplotlib/seaborn.
Do not hardcode anything; interpret the data dynamically.
Questions:
{questions_text}

Files (sample content if too large):
{attachment_texts}

Instructions:
- Return a JSON array of answers in order.
- If a plot is required, return it as a base64 PNG under 100,000 bytes.
- Only return the JSON array as output.
"""
    # Call the LLM via AI Pipe using a valid model name
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Changed to a valid, supported model
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer_text = response.choices[0].message.content

    # Parse the LLM's JSON response safely
    try:
        answers = json.loads(answer_text)
    except Exception:
        answers = [answer_text]

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

    result = await analyze_questions(questions_text, attachments)
    return JSONResponse(content=result)

# ---------- Run with Uvicorn ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)