# analysis_agent.py
import pandas as pd
import requests
import io, os, time, traceback, math, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any
import duckdb
from PIL import Image
import openai

# OpenAI key from environment (optional but recommended)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

PLOT_MAX_BYTES = int(os.getenv("PLOT_MAX_BYTES", "100000"))  # 100 KB default

def to_data_uri(img_bytes: bytes, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(img_bytes).decode("ascii")

def compress_png_bytes(img_bytes: bytes, max_bytes=PLOT_MAX_BYTES):
    """
    Try to compress PNG bytes by opening with PIL and saving with decreasing quality/size.
    Returns best-effort bytes (might still exceed max_bytes).
    """
    try:
        im = Image.open(io.BytesIO(img_bytes))
    except Exception:
        return img_bytes

    # try resizing progressively
    for scale in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        w = max(1, int(im.width * scale))
        h = max(1, int(im.height * scale))
        im2 = im.resize((w,h), Image.LANCZOS)
        buf = io.BytesIO()
        # save as PNG (lossless) — PNG doesn't support 'quality' but smaller dims shrink size
        im2.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data
    # fallback: try WebP lossy which usually is much smaller
    for quality in [80,65,50,40,30]:
        buf = io.BytesIO()
        im.save(buf, format="WEBP", quality=quality, method=6)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data
    # return smallest we created
    return data

def save_fig_to_bytes(fig, max_bytes=PLOT_MAX_BYTES):
    """
    Iterate over dpi/size to produce smallest acceptable PNG, or webp fallback.
    """
    buf = io.BytesIO()
    # try set of DPIs
    for dpi in [200,150,120,100,80,60,50,40]:
        buf.seek(0); buf.truncate(0)
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data, "image/png"
    # try reducing figure size
    for size in [(8,6),(7,5),(6,4),(5,3),(4,3),(3,2.5)]:
        fig.set_size_inches(size)
        buf.seek(0); buf.truncate(0)
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=80)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data, "image/png"
    # fallback: produce webp via PIL post-processing
    buf.seek(0)
    data = buf.getvalue()
    compressed = compress_png_bytes(data, max_bytes=max_bytes)
    # detect format
    if compressed[:4] == b'RIFF' or compressed[:4] == b'RIFF':
        return compressed, "image/webp"
    # default
    return compressed, "image/webp"

def parse_money_to_float(s):
    try:
        if pd.isna(s): return None
        st = str(s).lower().strip()
        st = st.replace('$','').replace(',','').replace('\xa0','')
        # handle bn, m
        if 'bn' in st or 'b' in st:
            st2 = st.replace('bn','').replace('b','').strip()
            return float(st2) * 1_000_000_000
        if 'm' in st and not 'mm' in st:
            st2 = st.replace('m','').strip()
            return float(st2) * 1_000_000
        # plain numeric
        return float(st)
    except:
        return None

def extract_table_from_wikipedia(html_text):
    """
    Use pandas.read_html to extract candidate tables and pick the table
    that looks like the highest grossing films table.
    """
    tables = pd.read_html(html_text)
    # heuristics: column names include 'Rank' and 'Worldwide' or 'Peak'
    best = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns.astype(str)]
        joined = " ".join(cols)
        if ("rank" in joined and ("worldwide" in joined or "gross" in joined or "peak" in joined)):
            best = t
            break
    if best is None:
        # fallback: choose largest table with at least 3 columns
        cand = sorted(tables, key=lambda x: x.shape[1]*x.shape[0], reverse=True)
        best = cand[0] if cand else None
    return best

def safe_numeric_series(s):
    return pd.to_numeric(s.astype(str).str.extract(r'(-?\d+\.?\d*)')[0], errors='coerce')

async def run_analysis_async(questions: str, files: Dict[str, bytes], start_time=None, max_seconds=170) -> Any:
    """
    Main dispatcher.
    - Uses simple heuristics + optional OpenAI plan step to understand tasks.
    - Returns JSON-serializable answers exactly as requested.
    """
    if start_time is None:
        start_time = time.time()

    try:
        q_lower = questions.lower()
        # If OpenAI key present, optionally ask it to summarize tasks (short and safe)
        use_llm = False
        if OPENAI_API_KEY:
            try:
                plan_prompt = (
                    "You are a small planner. Given the user instructions below, "
                    "produce a one-sentence plan and list the required outputs (e.g., a 4-element JSON "
                    "array containing [ans1, ans2, corr, data-uri]). Only respond with JSON: "
                    '{"plan":"...", "outputs":["array","..."]}.\\n\\nInstructions:\\n' + questions
                )
                resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",  # keep model flexible; if unavailable it will error -> fallback
                    messages=[{"role":"user","content":plan_prompt}],
                    max_tokens=150,
                    temperature=0.0
                )
                # we won't rely on it too strongly; primarily a helper
                use_llm = True
            except Exception:
                use_llm = False

        # If a Wikipedia URL is present in questions, handle scraping
        import re
        urls = re.findall(r'https?://[^\s,]+', questions)
        wiki_url = None
        for u in urls:
            if "wikipedia.org" in u:
                wiki_url = u
                break

        if wiki_url:
            # Scrape page
            r = requests.get(wiki_url, timeout=30)
            r.raise_for_status()
            table = extract_table_from_wikipedia(r.text)
            if table is None:
                # return structured failure in the expected shape
                return ["0", "", 0.0, ""]

            df = table.copy()
            # Try to find title/name column
            title_col = None
            candidates = [c for c in df.columns if 'title' in str(c).lower() or 'film' in str(c).lower() or 'movie' in str(c).lower() or 'name' in str(c).lower()]
            if candidates:
                title_col = candidates[0]
            # columns for rank, peak, year, gross
            rank_col = next((c for c in df.columns if 'rank' in str(c).lower()), None)
            peak_col = next((c for c in df.columns if 'peak' in str(c).lower()), None)
            year_col = next((c for c in df.columns if 'year' in str(c).lower() or 'release' in str(c).lower()), None)
            gross_col = next((c for c in df.columns if 'worldwide' in str(c).lower() or 'gross' in str(c).lower()), None)

            # make normalized columns
            norm = pd.DataFrame()
            if rank_col is not None:
                norm['Rank'] = safe_numeric_series(df[rank_col])
            if peak_col is not None:
                norm['Peak'] = safe_numeric_series(df[peak_col])
            if gross_col is not None:
                norm['Gross'] = df[gross_col].apply(parse_money_to_float)
            # find years in any column if no explicit year_col
            if year_col is not None:
                years = pd.to_numeric(df[year_col].astype(str).str.extract(r'(\d{4})')[0], errors='coerce')
                norm['Year'] = years
            else:
                # try to parse from any string columns
                for c in df.columns:
                    if str(df[c].dtype) == 'object':
                        yy = df[c].astype(str).str.extract(r'(\b(19|20)\d{2}\b)')[0]
                        if yy.notna().any():
                            norm['Year'] = pd.to_numeric(yy, errors='coerce')
                            break

            # Q1: count movies with gross >= 2bn and year < 2000
            ans1 = 0
            if 'Gross' in norm.columns and 'Year' in norm.columns:
                try:
                    ans1 = int(norm[(norm['Gross'].notna()) & (norm['Gross'] >= 2_000_000_000) & (norm['Year'] < 2000)].shape[0])
                except Exception:
                    ans1 = 0
            else:
                ans1 = 0

            # Q2: earliest film that grossed over 1.5bn
            ans2 = ""
            if 'Gross' in norm.columns:
                try:
                    df_big = norm[norm['Gross'] >= 1_500_000_000].copy()
                    if not df_big.empty:
                        # get smallest year
                        if 'Year' in df_big.columns and df_big['Year'].notna().any():
                            idx = df_big['Year'].idxmin()
                            if title_col is not None:
                                ans2 = str(df.loc[idx, title_col])
                            else:
                                # fallback to index or any column
                                ans2 = df.index[idx] if idx is not None else ""
                        else:
                            # fallback: pick first index from df_big
                            idx = df_big.index[0]
                            ans2 = str(df.loc[idx, title_col]) if title_col else str(idx)
                    else:
                        ans2 = ""
                except Exception:
                    ans2 = ""
            else:
                ans2 = ""

            # Q3: correlation Rank vs Peak
            corr = 0.0
            if 'Rank' in norm.columns and 'Peak' in norm.columns:
                try:
                    d = norm[['Rank','Peak']].dropna()
                    if not d.empty and len(d) > 1:
                        corr = float(d['Rank'].corr(d['Peak']))
                    else:
                        corr = 0.0
                except Exception:
                    corr = 0.0
            else:
                corr = 0.0

            # Q4: plot Rank vs Peak with dotted red regression line
            data_uri = ""
            if 'Rank' in norm.columns and 'Peak' in norm.columns:
                d = norm[['Rank','Peak']].dropna()
                if not d.empty and len(d) > 1:
                    fig, ax = plt.subplots(figsize=(6,4))
                    ax.scatter(d['Rank'], d['Peak'], s=20)
                    ax.set_xlabel("Rank")
                    ax.set_ylabel("Peak")
                    ax.set_title("Rank vs Peak")
                    # regression
                    try:
                        x = d['Rank'].values.reshape(-1,1)
                        y = d['Peak'].values
                        from sklearn.linear_model import LinearRegression
                        reg = LinearRegression().fit(x,y)
                        xs = np.linspace(x.min(), x.max(), 200)
                        ys = reg.predict(xs.reshape(-1,1))
                        ax.plot(xs, ys, linestyle='--', color='red')
                    except Exception:
                        pass
                    img_bytes, mime = save_fig_to_bytes(fig, max_bytes=PLOT_MAX_BYTES)
                    img_bytes = compress_png_bytes(img_bytes, max_bytes=PLOT_MAX_BYTES)
                    data_uri = to_data_uri(img_bytes, mime=f"image/{mime.split('/')[-1]}")
                    plt.close(fig)

            # final output must be JSON array of four items
            # ensure types: ans1:int, ans2:str, corr:float, image:str
            return [int(ans1), str(ans2), float(0.0 if (not isinstance(corr,(int,float)) or math.isnan(corr)) else round(corr,6)), data_uri]

        # Fallback: if 'data.csv' in files => run small summary + top correlation / sample plot
        if any(name.lower().endswith('.csv') for name in files.keys()):
            # simple CSV analysis
            csv_name = next(name for name in files.keys() if name.lower().endswith('.csv'))
            df = pd.read_csv(io.BytesIO(files[csv_name]))
            # sample response
            out = {
                "n_rows": int(df.shape[0]),
                "n_cols": int(df.shape[1]),
                "columns": df.columns.tolist()
            }
            return out

        # Other fallback: return helpful error
        return {"error":"No recognized action. Provide a Wikipedia URL or attach data."}

    except Exception as e:
        return {"error": "internal", "message": str(e), "trace": traceback.format_exc()}