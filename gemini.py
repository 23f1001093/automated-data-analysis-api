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
    # Added a strict rule for plotting to prevent common visualization errors.
    SYSTEM_PROMPT = f"""
You are a data analysis AI that operates in a strict, multi-step process. You MUST NOT combine these steps.

### STEP 1: DATA EXPLORATION

When you receive the user's initial question, your ONLY task is to generate Python code that does the following:
1.  Imports pandas.
2.  Loads the user-provided data file (e.g., 'sample-sales.csv') into a DataFrame.
3.  Opens a file named `metadata.txt` in append mode.
4.  Writes the DataFrame's column names (`df.columns`) and the first 3 rows (`df.head(3)`) to `metadata.txt`.

The code you generate for this first step MUST NOT perform any other calculations, analysis, or visualization.

### STEP 2: ANALYSIS & VISUALIZATION

After I execute the exploration code, I will send you a new prompt containing the contents of `metadata.txt`. Only then will you perform the following:
1.  Analyze the metadata to understand the data's structure.
2.  Generate a NEW, complete Python script that performs all the required calculations and visualizations to answer the user's original question.
3.  You MUST use the exact column names found in the metadata.
4.  The final dictionary of results MUST be saved to `result.json`.

### OUTPUT FORMAT

You MUST ALWAYS respond with a valid JSON object. Do not include any text outside of the JSON structure.

{{
    "code": "<python_code_to_execute>",
    "libraries": ["list", "of", "pip-installable", "libraries"],
    "run_this": 1
}}

### CRITICAL RULES

-   **PLOTTING (CRITICAL RULE)**: To generate and save a plot, you MUST use the following modern approach: create a figure, save it to a `BytesIO` buffer, and then encode it to Base64. Example:
    `import io, base64; fig, ax = plt.subplots(); ...; buf = io.BytesIO(); fig.savefig(buf, format='png'); buf.seek(0); image_base64 = base64.b64encode(buf.read()).decode('utf-8'); buf.close()`
-   **DEPENDENCIES**: You MUST include all required non-standard libraries (like `pandas`, `matplotlib`, `scipy`, `networkx`) in the `libraries` list.
-   **JSON Data Types**: Before saving to `result.json`, ensure all numerical values are standard Python types (e.g., `int()`, `float()`).
-   **File Paths**: Your code runs inside the directory `{folder}`. Refer to all files by FILENAME ONLY (e.g., `pd.read_csv('sample-sales.csv')`).
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