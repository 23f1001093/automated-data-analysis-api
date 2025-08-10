# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List, Dict, Any
import uvicorn
import time
import os
from analysis_agent import run_analysis_async

app = FastAPI()

# configuration via env
MAX_SECONDS = int(os.getenv("AGENT_MAX_SECONDS", "170"))  # leave margin for 3-minute limit

@app.post("/api/", response_model=Any)
async def data_analyst_agent(files: List[UploadFile] = File(...)):
    """
    Accepts multipart/form-data:
      - questions.txt (required)
      - 0..N other files (data.csv, data.parquet, images...)
    Returns whatever the analysis returns (JSON array/object/string).
    """
    start = time.time()

    # read uploads into memory (small-to-medium payloads)
    uploaded = {}
    questions_text = None
    for f in files:
        name = f.filename or ""
        content = await f.read()
        uploaded[name] = content
        if name.lower() == "questions.txt" or name.lower().endswith("questions.txt"):
            try:
                questions_text = content.decode("utf-8", errors="ignore")
            except:
                questions_text = content.decode("latin-1", errors="ignore")

    if not questions_text:
        raise HTTPException(status_code=400, detail="questions.txt is required in the multipart upload.")

    # offload to analysis agent (this function is async and has its own time check)
    try:
        result = await run_analysis_async(questions_text, uploaded, start_time=start, max_seconds=MAX_SECONDS)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level="info")