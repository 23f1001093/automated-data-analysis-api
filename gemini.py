import os
import json
import google.generativeai as genai
from api_key_rotator import get_api_key
import re


MODEL_NAME = "gemini-1.5-pro-latest"

# Give response in JSON format
generation_config = genai.types.GenerationConfig(
    response_mime_type="application/json"
)


async def send_with_rotation(prompt, session_id, system_prompt):
    """Sends a prompt to the Gemini API, handling API key rotation and retries."""
    while True:
        try:
            api_key = get_api_key(auto_wait=True)
            genai.configure(api_key=api_key)

            chat = await get_chat_session(parse_chat_sessions, session_id, system_prompt)
            
            response = chat.send_message(prompt)
          
            return response

        except Exception as e:
            print(f"⚠️ API key {api_key} failed: {e}. Retrying with another key...")
            continue


# In-memory dictionary to store active chat sessions
parse_chat_sessions = {}


async def get_chat_session(sessions_dict, session_id, system_prompt, model_name=MODEL_NAME):
    """
    Retrieves an existing chat session or creates a new one if it doesn't exist.
    """
    if session_id not in sessions_dict:
        print(f"Creating new chat session for ID: {session_id}")
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            system_instruction=system_prompt
        )
        chat = model.start_chat(history=[])
        sessions_dict[session_id] = chat
    return sessions_dict[session_id]


async def parse_question_with_llm(question_text=None, uploaded_files=None, session_id="default_parse", retry_message=None, folder="uploads"):
    """
    Parses a user's question using the LLM with a persistent chat session.
    """

    # --- FIX ---
    # Added a strict instruction to check for column existence before using them.
    SYSTEM_PROMPT = f"""
You are a highly intelligent AI assistant that specializes in generating Python code for data analysis. Your primary goal is to answer user questions by creating and executing a multi-step analysis plan.

### CORE WORKFLOW

1.  **Analyze the Request**: Understand the user's question and the data sources provided.
2.  **Step 1: Initial Data Exploration**: Your first step is ALWAYS to generate code that inspects the dataset. For CSVs, this means loading it into a pandas DataFrame and printing the column names (`df.columns`) and the first few rows (`df.head()`). Append these findings to `metadata.txt`.
3.  **Step 2: Generate Analysis Code**: Using the verified column names from `metadata.txt`, generate the complete Python code to perform the full analysis.
4.  **Step 3: Validation and Correction**: If I provide you with an error message, you must debug it and provide the corrected code.

### OUTPUT FORMAT

You MUST ALWAYS respond with a valid JSON object in the following structure. Do NOT include any explanations or text outside of the JSON structure.

{{
    "code": "<python_code_to_execute>",
    "libraries": ["list", "of", "pip-installable", "libraries"],
    "run_this": 1
}}

### RULES & CONSTRAINTS

-   **COLUMN NAMES (CRITICAL RULE)**: Before using any column name in your code, you MUST verify that it exists by inspecting the data first (e.g., with `df.columns`). Do not assume column names like 'precipitation_mm' exist.
-   **FILE PATHS**: Your code will be executed inside the working directory `{folder}`. YOU MUST refer to all files using their FILENAME ONLY (e.g., `pd.read_csv('sample-sales.csv')`).
-   **Final Answer**: The final output of your analysis must always be written to `result.json`.
-   **Imports**: Always include all necessary imports (like `json`, `pandas`) in your generated code.
"""

    chat = await get_chat_session(parse_chat_sessions, session_id, SYSTEM_PROMPT)

    if retry_message:
        prompt = retry_message
    else:
        prompt = question_text

    file_path = os.path.join(folder, "metadata.txt")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("")

    history_data = []
    for msg in chat.history:
        history_data.append({
            "role": msg.role,
            "parts": [str(p) for p in msg.parts]
        })
    chat_history_path = os.path.join(folder, "chat_history.json")
    with open(chat_history_path, "w") as f:
        json.dump(history_data, f, indent=2)

    response = await send_with_rotation(prompt, session_id, SYSTEM_PROMPT)

    if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
        print("⚠️(gemini.py) LLM returned a response with no content part.")
        return {"error": "LLM returned no content."}
        
    try:
        return json.loads(response.text)
    except Exception as e:
        print(f"⚠️(gemini.py) Failed to parse response as JSON: {e}")
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        
        return {"error": "Failed to parse JSON from LLM response."}
