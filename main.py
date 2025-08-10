from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import List
import uvicorn
from analysis_agent import run_analysis

app = FastAPI()

@app.post("/api/")
async def data_analyst_agent(
    questions_txt: UploadFile = File(..., alias="questions.txt"),
    files: List[UploadFile] = File(default=[])
):
    """
    API endpoint for the data analyst agent.
    Accepts a 'questions.txt' file and optional data files.
    """
    try:
        # Read the questions from the text file
        questions_content = await questions_txt.read()
        questions = questions_content.decode('utf-8')

        # Prepare a dictionary of uploaded files
        uploaded_files = {file.filename: await file.read() for file in files}

        # Run the analysis asynchronously
        result = await run_analysis(questions, uploaded_files)
        
        return result
    
    except Exception as e:
        # Return a 500 Internal Server Error for unexpected issues
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)