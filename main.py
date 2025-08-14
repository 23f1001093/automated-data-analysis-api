import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import os
import shutil
import json

# Import the new, more robust utils file
from utils import answer_all_questions

app = FastAPI()

def parse_questions_from_file(file_content: bytes) -> str:
    return file_content.decode('utf-8')

@app.post("/api/")
async def analyze_data(files: List[UploadFile] = File(...)):
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
        shutil.rmtree(temp_dir)