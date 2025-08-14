import json
import tempfile
import shutil
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from utils import (
    parse_questions,
    answer_all_questions,
    ensure_eval_array
)

app = FastAPI()

@app.post("/api/")
async def analyze_api(files: list[UploadFile] = File(...)):
    temp_dir = None
    try:
        # Create a temporary directory to save uploaded files
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        questions_file = None

        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append(file_path)

            if file.filename == "questions.txt":
                questions_file = file_path
        
        if not questions_file:
            return JSONResponse(content=ensure_eval_array(["questions.txt required"] * 4), status_code=400)

        with open(questions_file, 'r', encoding='utf-8') as f:
            q_text = f.read()

        questions = parse_questions(q_text)
        answers = answer_all_questions(questions, file_paths)
        return JSONResponse(content=ensure_eval_array(answers))

    except Exception as ex:
        # Handle exceptions and return a consistent error format
        return JSONResponse(content=ensure_eval_array([str(ex)] * 4), status_code=500)
    finally:
        # Clean up the temporary directory
        if temp_dir:
            shutil.rmtree(temp_dir)