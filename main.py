import shutil
import os
import json
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from utils import answer_all_questions # Assuming utils.py is in the same directory

app = FastAPI()

@app.post("/api/")
@app.post("/")
async def analyze_data(files: List[UploadFile] = File(None)):
    """
    Analyzes data from uploaded files based on questions.txt.
    """
    if not files:
        return JSONResponse(status_code=400, content={"error": "No files provided."})

    questions_content = None
    file_paths = {}
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            if file.filename == "questions.txt":
                with open(file_path, "r") as f:
                    questions_content = f.read()
            else:
                file_paths[file.filename] = file_path
        
        if not questions_content:
            return JSONResponse(status_code=400, content={"error": "questions.txt is required"})

        answers = answer_all_questions(questions_content, file_paths)
        return JSONResponse(content=answers)
    
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)