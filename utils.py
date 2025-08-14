import pandas as pd
import requests
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.linear_model import LinearRegression

def parse_questions(q_text: str):
    questions_list = []
    
    preamble_match = re.match(r"(.*?)\n\d+\.", q_text, flags=re.DOTALL)
    if preamble_match:
        preamble = preamble_match.group(1).strip()
        if preamble:
            questions_list.append(preamble)

    pattern = r"\d+\.\s*(.*?)(?=\n\d+\.|$)"
    questions_list.extend([q.strip() for q in re.findall(pattern, q_text, flags=re.DOTALL)])
    
    return questions_list

def ensure_eval_array(answers):
    if not isinstance(answers, list):
        answers = [answers]
    while len(answers) < 4:
        answers.append(None)
    return answers[:4]

def answer_all_questions(questions, file_paths):
    answers = []
    
    full_text = " ".join(questions)
    match = re.search(r"http[s]?://\S+", full_text)
    if match:
        page_url = match.group(0)
    else:
        return ["No URL found"] * 4
    
    df = _get_and_process_wiki_table(page_url)
    if isinstance(df, str):
        return [df] * 4

    for q in questions[1:]:
        q_lower = q.lower()
        try:
            if "2 bn movies" in q_lower:
                q1_answer = len(df[(df['Worldwide gross'] >= 2000000000) & (df['Year'] < 2000)])
                answers.append(str(q1_answer))
            elif "earliest film" in q_lower:
                q2_df = df[df['Worldwide gross'] >= 1500000000].sort_values(by='Year')
                q2_answer = q2_df.iloc[0]['Title'] if not q2_df.empty else "No such film"
                answers.append(q2_answer)
            elif "correlation" in q_lower:
                df_cleaned = df.dropna(subset=['Rank', 'Peak'])
                correlation = df_cleaned['Rank'].corr(df_cleaned['Peak'])
                answers.append(float(np.round(correlation, 6)))
            elif "scatterplot" in q_lower:
                answers.append(_generate_scatterplot(df))
        except Exception as ex:
            answers.append(str(ex))
    
    return ensure_eval_array(answers)

def _get_and_process_wiki_table(url):
    try:
        response = requests.get(url, timeout=20)
        tables = pd.read_html(response.text)
        
        if not tables:
            return "No table found"
        
        df = tables[0]
        
        df.columns = [col.replace('[a]', '').strip() for col in df.columns]
        
        df['Worldwide gross'] = df['Worldwide gross'].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
        df['Worldwide gross'] = pd.to_numeric(df['Worldwide gross'], errors='coerce')
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        
        # New code to handle non-numeric values in Rank and Peak
        df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
        df['Peak'] = pd.to_numeric(df['Peak'], errors='coerce')
        
        return df
    except Exception as e:
        return f"Error during data processing: {str(e)}"
        
def _generate_scatterplot(df):
    df_cleaned = df.dropna(subset=['Rank', 'Peak'])
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Rank', y='Peak', data=df_cleaned)
    
    X = df_cleaned[['Rank']].values
    y = df_cleaned['Peak'].values
    reg = LinearRegression().fit(X, y)
    plt.plot(X, reg.predict(X), color='red', linestyle=':')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"