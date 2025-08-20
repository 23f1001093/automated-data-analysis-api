from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import aiofiles
import json
import logging
import difflib
import time
import itertools
import re
import subprocess
import sys

from gemini import parse_question_with_llm

async def run_python_code(code, libraries, folder, python_exec):
    """
    Executes Python code safely in a separate subprocess.
    """
    script_path = os.path.join(folder, "script.py")
    
    with open(script_path, "w") as f:
        f.write(code)
    
    # The command uses the simple filename because `cwd` is set to the script's directory.
    command = [python_exec, "script.py"] 
    
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=folder
        )
        
        if process.returncode == 0:
            return {"code": 1, "output": process.stdout}
        else:
            return {"code": 0, "output": process.stderr}
            
    except subprocess.TimeoutExpired:
        return {"code": 0, "output": "Execution timed out after 60 seconds."}
    except Exception as e:
        return {"code": 0, "output": f"Failed to execute script: {e}"}


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if os.path.exists("frontend.html"):
        with open("frontend.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    return HTMLResponse(content="<h1>Frontend file not found</h1>")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def last_n_words(s, n=100):
    s = str(s)
    words = s.split()
    return ' '.join(words[-n:])

def is_base64_image(s: str) -> bool:
    if s.startswith("data:image"):
        return True
    if len(s) > 100 and re.fullmatch(r'[A-Za-z0-9+/=]+', s):
        return True
    return False

def strip_base64_from_json(data: dict) -> dict:
    def _process_value(value):
        if isinstance(value, str) and is_base64_image(value):
            return "[IMAGE_BASE64_STRIPPED]"
        elif isinstance(value, list):
            return [_process_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: _process_value(v) for k, v in value.items()}
        return value
    return _process_value(data)

PYTHON_EXECUTABLE = sys.executable

@app.post("/api")
async def analyze(request: Request):
    main_loop = 0
    while main_loop < 3:
        try:
            request_id = str(uuid.uuid4())
            
            # --- FIX ---
            # Create an absolute path for the request folder to prevent any ambiguity.
            request_folder = os.path.abspath(os.path.join(UPLOAD_DIR, request_id))
            os.makedirs(request_folder, exist_ok=True)
            
            log_path = os.path.join(request_folder, "app.log")
            
            logger = logging.getLogger(request_id)
            logger.setLevel(logging.INFO)
            if logger.hasHandlers():
                logger.handlers.clear()
            file_handler = logging.FileHandler(log_path)
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
            
            logger.info("Step-1: Folder created: %s", request_folder)
            
            form = await request.form()
            question_text = None
            saved_files = {}
            
            for field_name, value in form.items():
                if hasattr(value, "filename") and value.filename:
                    file_path = os.path.join(request_folder, value.filename)
                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(await value.read())
                    saved_files[field_name] = file_path
                    if "question" in field_name.lower():
                        async with aiofiles.open(file_path, "r") as f:
                            question_text = await f.read()
                else:
                    saved_files[field_name] = value

            if question_text is None and saved_files:
                target_name = "question.txt"
                file_names = [name for name in saved_files.keys() if isinstance(saved_files[name], str) and os.path.exists(saved_files[name])]
                closest_matches = difflib.get_close_matches(target_name, file_names, n=1, cutoff=0.6)
                if closest_matches:
                    selected_file_path = saved_files[closest_matches[0]]
                    async with aiofiles.open(selected_file_path, "r") as f:
                        question_text = await f.read()

            if not question_text:
                return JSONResponse(status_code=400, content={"message": "Could not find a question to process."})

            logger.info("Using Python executable: %s", PYTHON_EXECUTABLE)
            
            question_to_llm = f"<question>{question_text}</question>"
            
            session_id = request_id
            retry_message = None
            runner = 1
            max_attempts = 3
            
            response = None
            for attempt in range(max_attempts):
                logger.info("🤖 Step-1: Getting initial code from LLM. Attempt %d", attempt + 1)
                try:
                    prompt_to_send = retry_message if retry_message else question_to_llm
                    response = await parse_question_with_llm(
                        question_text=prompt_to_send,
                        folder=request_folder,
                        session_id=session_id,
                    )
                    if isinstance(response, dict) and "error" not in response:
                        logger.info("🤖 Step-1: Successfully parsed response from LLM.")
                        break
                    else:
                        retry_message = f"Invalid JSON response received: {response}. Please provide a valid JSON object."
                except Exception as e:
                    retry_message = f"⚠️ Error during initial LLM call: <error>{last_n_words(str(e))}</error>"
                    logger.error("❌🤖 Step-1: Error parsing LLM response: %s", retry_message)
                response = None

            if not response:
                return JSONResponse(status_code=500, content={"message": "Error: Could not get a valid initial response from the AI."})
            
            code_to_run = response.get("code", "")
            required_libraries = response.get("libraries", [])
            runner = response.get("run_this", 1)
            
            loop_counter = 0
            while runner == 1 and loop_counter < 5:
                loop_counter += 1
                logger.info(f"💻 Loop-{loop_counter}: Running LLM-generated code.")

                execution_result = await run_python_code(
                    code=code_to_run,
                    libraries=required_libraries,
                    folder=request_folder,
                    python_exec=PYTHON_EXECUTABLE
                )
                
                if execution_result["code"] == 0:
                    logger.error(f"❌💻 Loop-{loop_counter}: Code execution failed: %s", last_n_words(execution_result["output"]))
                    retry_message = f"<error_snippet>{last_n_words(execution_result['output'])}</error_snippet>\nSolve this error."
                else:
                    logger.info(f"✅💻 Loop-{loop_counter}: Code executed successfully.")
                    metadata_file = os.path.join(request_folder, "metadata.txt")
                    if os.path.exists(metadata_file):
                        with open(metadata_file, "r") as f:
                            metadata = f.read()
                        retry_message = f"<metadata>{metadata}</metadata>"
                    else:
                        retry_message = "Metadata file not found. Please generate the next step."
                
                result_path = os.path.join(request_folder, "result.json")
                
                if os.path.exists(result_path):
                    logger.info(f"✅📁 Loop-{loop_counter}: Found result.json. Verifying...")
                    with open(result_path, "r") as f:
                        result_content = f.read()
                    
                    try:
                        stripped_result = strip_base64_from_json(json.loads(result_content))
                    except json.JSONDecodeError:
                        stripped_result = result_content

                    verification_prompt = f"""
                    Check this result based on the original question: <result>{json.dumps(stripped_result)}</result>
                    - If correct, return `"run_this": 0`.
                    - If incorrect, generate new code to fix it and set `"run_this": 1`.
                    """
                    retry_message = verification_prompt

                response = None
                for attempt in range(max_attempts):
                    logger.info(f"🤖 Loop-{loop_counter}: Getting next step from LLM. Attempt %d", attempt + 1)
                    try:
                        response = await parse_question_with_llm(
                            retry_message=retry_message,
                            folder=request_folder,
                            session_id=session_id
                        )
                        if isinstance(response, dict) and "error" not in response:
                            break
                    except Exception as e:
                        logger.error(f"❌🤖 Loop-{loop_counter}: Error parsing LLM response: %s", e)
                    response = None

                if not response:
                    return JSONResponse(status_code=500, content={"message": "Error: Could not get a valid response from the AI in the execution loop."})
                
                code_to_run = response.get("code", "")
                required_libraries = response.get("libraries", [])
                runner = response.get("run_this", 0)

            final_result_path = os.path.join(request_folder, "result.json")
            if os.path.exists(final_result_path):
                with open(final_result_path, "r") as f:
                    try:
                        final_data = json.load(f)
                        logger.info("✅ Final result found. Returning to user.")
                        return JSONResponse(content=final_data)
                    except Exception as e:
                        logger.error(f"❌ Error reading final result.json: {e}")
                        f.seek(0)
                        return JSONResponse(status_code=500, content={"message": f"Error reading result file: {e}", "raw_content": f.read()})
            else:
                 return JSONResponse(status_code=404, content={"message": "Analysis finished, but no result.json file was produced."})

        except Exception as e:
            logger.error(f"❌ An unexpected error occurred in the main function: {e}", exc_info=True)
            main_loop += 1
    
    return JSONResponse(status_code=500, content={"message": "Maximum main loop iterations reached. Could not complete the request."})
