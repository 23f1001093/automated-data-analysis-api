from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from analysis_agent import run_analysis_async

app = FastAPI()

@app.post("/api/")
async def analyze(request: Request):
    form = await request.form()
    files_dict = {}

    # Read all uploaded files into memory
    for name, file in form.items():
        if hasattr(file, "filename"):
            content = await file.read()
            files_dict[file.filename] = content

    # questions.txt is mandatory
    if "questions.txt" not in files_dict:
        return JSONResponse(
            status_code=400,
            content={"error": "questions.txt is required"}
        )

    # Read questions as string
    questions_str = files_dict["questions.txt"].decode("utf-8", errors="ignore")

    try:
        output = await run_analysis_async(
            questions=questions_str,
            files=files_dict
        )
        return JSONResponse(content=output)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})