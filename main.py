import uvicorn
import pandas as pd
import requests
import re
import io
import base64
import logging
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional

# ==============================================================================
# 1. TOOLS - Functions for scraping, analysis, plotting
# ==============================================================================

def scrape_highest_grossing_films(url: str) -> Optional[pd.DataFrame]:
    """
    Scrapes the list of highest-grossing films from the given Wikipedia URL.
    
    Args:
        url: The Wikipedia URL.
        
    Returns:
        A cleaned pandas DataFrame of the film data.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    
    # pandas.read_html() returns a list of DataFrames found on the page
    # FIX: Use io.StringIO to address the FutureWarning
    tables = pd.read_html(io.StringIO(response.text), attrs={"class": "wikitable"})
    
    if not tables:
        return None
        
    df = tables[0] # The main table is usually the first one
    
    # --- Data Cleaning ---
    # Rename columns for easier access
    df.columns = ['Rank', 'Peak', 'Title', 'Worldwide gross', 'Year', 'Reference']
    
    # Clean 'Worldwide gross' column - remove $, commas, and citations like [1]
    def clean_gross(value):
        value = re.sub(r'\[\d+\]', '', str(value)) # Remove citations like [1], [2]
        value = value.replace('$', '').replace(',', '')
        return pd.to_numeric(value)

    df['Worldwide gross'] = df['Worldwide gross'].apply(clean_gross)
    
    # Clean 'Year' and 'Rank' and 'Peak' columns and convert to numeric
    df['Year'] = pd.to_numeric(df['Year'].astype(str).str.extract(r'(\d{4})', expand=False))
    df['Rank'] = pd.to_numeric(df['Rank'])
    df['Peak'] = pd.to_numeric(df['Peak'])
    
    # Drop the unnecessary Reference column
    df = df.drop(columns=['Reference'])
    
    return df

def calculate_correlation(df: pd.DataFrame, col1: str, col2: str) -> float:
    """Calculates the Pearson correlation between two columns in a DataFrame."""
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"Columns {col1} or {col2} not found in DataFrame.")
    return df[col1].corr(df[col2])

def generate_scatterplot(df: pd.DataFrame, x_col: str, y_col: str, title: str, x_label: str, y_label: str) -> str:
    """
    Generates a scatterplot with a regression line and returns it as a base64 encoded data URI.
    
    Returns:
        A string in the format "data:image/png;base64,..."
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create the scatterplot with a regression line using seaborn
    sns.regplot(
        x=df[x_col],
        y=df[y_col],
        ax=ax,
        scatter_kws={'alpha': 0.6, 's': 50},
        line_kws={'color': 'red', 'linestyle': '--', 'linewidth': 2}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    plt.tight_layout()
    
    # --- Save plot to an in-memory buffer ---
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80) # Use lower dpi to keep size down
    buf.seek(0)
    
    # --- Encode buffer to base64 ---
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig) # Close the figure to free up memory
    
    data_uri = f"data:image/png;base64,{image_base64}"

    # Check if the image size is under the limit (100,000 bytes)
    if len(data_uri) > 100000:
        print(f"Warning: Image size is {len(data_uri)} bytes, which is over the 100,000 byte limit.")

    return data_uri

# ==============================================================================
# 2. AGENT - Core logic to process questions and delegate tasks
# ==============================================================================

class DataAnalystAgent:
    """
    The agent responsible for orchestrating the data analysis tasks.
    It holds state (like a DataFrame) and uses tools to answer questions.
    """
    def __init__(self):
        self.dataframe = None
        self.results: List[Any] = []

    def _reset_state(self):
        """Resets the agent's state for a new request."""
        self.dataframe = None
        self.results = []

    def run(self, questions_content: str, files: Dict[str, UploadFile]) -> List[Any]:
        """
        Main execution method for the agent.

        Args:
            questions_content: The string content of questions.txt.
            files: A dictionary of other uploaded files.

        Returns:
            A list of answers.
        """
        self._reset_state()
        
        # Split questions by lines, ignoring empty ones
        questions = [q.strip() for q in questions_content.strip().split('\n') if q.strip()]

        for question in questions:
            self._process_question(question)
        
        return self.results

    def _process_question(self, question: str):
        """Processes a single question and calls the appropriate tool."""
        # Clean up question numbering like "1. " or "1. "
        cleaned_question = re.sub(r'^\d+\.\s*', '', question).lower()

        # --- Task Routing based on keywords ---
        
        # Scrape data task
        if "scrape" in cleaned_question and "highest grossing films" in cleaned_question:
            url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
            self.dataframe = scrape_highest_grossing_films(url)
            # This task doesn't produce a direct answer, it just loads data.
            return
            
        if self.dataframe is None:
            # If no dataframe is loaded, we can't answer analytical questions.
            return

        # Q1: How many $2 bn movies were released before 2000?
        if "how many" in cleaned_question and "2 bn" in cleaned_question and "before 2000" in cleaned_question:
            answer = self.dataframe[
                (self.dataframe['Worldwide gross'] >= 2_000_000_000) & 
                (self.dataframe['Year'] < 2000)
            ].shape[0]
            self.results.append(answer)

        # Q2: Which is the earliest film that grossed over $1.5 bn?
        elif "earliest film" in cleaned_question and "1.5 bn" in cleaned_question:
            filtered_df = self.dataframe[self.dataframe['Worldwide gross'] >= 1_500_000_000]
            
            # FIX: Check if filtered_df is empty before trying to access iloc[0]
            if not filtered_df.empty:
                answer = filtered_df.sort_values(by='Year').iloc[0]['Title']
            else:
                answer = "No films found that grossed over $1.5 billion."
            
            self.results.append(answer)

        # Q3: What's the correlation between the Rank and Peak?
        elif "correlation between rank and peak" in cleaned_question:
            correlation = calculate_correlation(self.dataframe, 'Rank', 'Peak')
            self.results.append(correlation)

        # Q4: Draw a scatterplot of Rank and Peak
        elif "draw a scatterplot of rank and peak" in cleaned_question:
            base64_image = generate_scatterplot(
                self.dataframe, 
                x_col='Rank', 
                y_col='Peak',
                title='Rank vs. Peak of Highest-Grossing Films',
                x_label='Rank',
                y_label='Peak Position'
            )
            self.results.append(base64_image)

# ==============================================================================
# 3. FastAPI APP - API endpoint setup
# ==============================================================================

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Data Analyst Agent API",
    description="An API that uses tools to source, prepare, analyze, and visualize data.",
    version="1.0.0"
)

# Initialize the agent
agent = DataAnalystAgent()

@app.post("/api/")
async def analyze_data(
    files: List[UploadFile] = File(...)
):
    """
    Accepts a data analysis task via a POST request with files.

    - **questions.txt**: (Required) Contains the questions to answer.
    - **Other files**: (Optional) Data files like .csv, .png, etc.

    Returns the analysis results in the format requested by the questions.
    """
    questions_file = None
    other_files: Dict[str, UploadFile] = {}

    for uploaded_file in files:
        if uploaded_file.filename == "questions.txt":
            questions_file = uploaded_file
        else:
            other_files[uploaded_file.filename] = uploaded_file

    if not questions_file:
        raise HTTPException(status_code=400, detail="questions.txt file is missing.")

    try:
        # Read the questions from the uploaded file
        questions_content = (await questions_file.read()).decode("utf-8")
        
        logger.info(f"Received questions:\n{questions_content}")
        
        # Process the request using the agent
        result = agent.run(questions_content, other_files)
        
        logger.info("Successfully processed request, sending response.")
        
        # Return the result as a JSON response
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the application using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)