# automated-data-analysis-api

An intelligent API service that leverages LLMs to source, prepare, analyze, and visualize data in response to plain-English analytical requests. Designed for participation in the TDS Project 2 challenge.

## Overview

The Data Analyst Agent exposes a single REST API endpoint that accepts POST requests containing:
- `questions.txt`: a text file with data analysis instructions (required)
- Zero or more data files (.csv, .json, .parquet, images, etc.)

It will:1
- Parse the questions,
- Load and process datasets,
- Clean, analyze, and compute results as requested,
- Return answers as a JSON array or object (including charts as base64 Data URIs).

## Example Request

curl "<api-endpoint>"
-F "questions.txt=@questions.txt"
-F "data.csv=@data.csv"

## Requirements

- All answers must be returned within **3 minutes**
- Output **must** be valid JSON in the requested format (array or object)
- Plots/images are returned as base64-encoded data URIs (max size: 100KB per image)

## Tech Stack

- **Backend:** Python (FastAPI/Flask)
- **Data Processing:** pandas, numpy, duckdb, pyarrow, requests, BeautifulSoup
- **Visualization:** matplotlib, seaborn, base64
- **Optional AI:** OpenAI API 
- **Deployment:** ngrok

