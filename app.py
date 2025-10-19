import os
import json
import faiss
import numpy as np
import pandas as pd
import requests
import tempfile
import streamlit as st
from openai import OpenAI

# ---------------------------
# CONFIGURATION
# ---------------------------
INDEX_URL = "https://huggingface.co/datasets/SriSumanth/mumbai-zomato-cafe-index/resolve/main/faiss.index"
META_URL  = "https://huggingface.co/datasets/SriSumanth/mumbai-zomato-cafe-index/resolve/main/metas.jsonl"
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o"
TOP_K = 50   # Number of candidates retrieved before GPT filtering

# ---------------------------
# Load API Key (from Streamlit Secrets)
# ---------------------------
api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI(api_key=api_key)

# ---------------------------
# Load FAISS index and metadata (from Hugging Face)
# ---------------------------
@st.cache_resource
def load_index_and_meta():
    st.info("📥 Downloading FAISS index from Hugging Face…")

    try:
        # Download index with progress
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
            with requests.get(INDEX_URL, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                downloaded = 0
                progress = st.progress(0)
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        tmp.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            progress.progress(min(downloaded / total, 1.0))
        
        # Read index after file is closed but still exists
        index = faiss.read_index(tmp_path)
        os.unlink(tmp_path)

        # Download metadata
        st.info("📥 Downloading metadata…")
        meta_resp = requests.get(META_URL, timeout=60)
        meta_resp.raise_for_status()
        metas = []
        for i, line in enumerate(meta_resp.text.strip().splitlines()):
            try:
                metas.append(json.loads(line))
            except json.JSONDecodeError as e:
                st.warning(f"⚠️ Skipping malformed JSON at line {i+1}: {str(e)}")
                continue
        
        st.success("✅ Index loaded successfully!")
        return index, metas
    
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTP Error: {e.response.status_code} - {e.response.reason}")
        st.error(f"URL: {e.response.url}")
        st.stop()
    except requests.exceptions.Timeout:
        st.error("❌ Download timed out. The files may be too large or the connection is slow.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading index: {str(e)}")
        st.stop()

index, metas = load_index_and_meta()

# ---------------------------
# Embed query
# ---------------------------
def embed_query(text: str):
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text]
    )
    return np.array(resp.data[0].embedding, dtype="float32")

# ---------------------------
# Retrieve from FAISS
# ---------------------------
def retrieve(query: str, k: int = TOP_K):
    qvec = embed_query(query)
    qvec = qvec / np.linalg.norm(qvec)
    scores, ids = index.search(qvec.reshape(1, -1), k)
    results = []
    for idx, score in zip(ids[0], scores[0]):
        if idx == -1:
            continue
        meta = metas[idx].copy()
        meta["score"] = float(score)
        results.append(meta)
    return results

# ---------------------------
# GPT-4o Reasoning
# ---------------------------
def generate_llm_recommendations(user_query: str, retrieved):
    df = pd.DataFrame(retrieved)
    df = df[['Name', 'Cuisine', 'Location', 'Rating', 'Cost_for_two', 'Review']]
    context_text = df.to_csv(index=False)

    system_prompt = """
You are an intelligent food place recommendation assistant specializing in cafés of Mumbai, India.
For this task, you are given:
1) A user's natural-language request describing preferences (cuisines, vibe, location, budget, group size, etc.).
2) A list of Mumbai cafés scraped from Zomato with metadata and real user reviews.

Your responsibilities

Intent parsing
- Extract cuisines (e.g., "South Indian", "Italian"), vibe/ambience keywords (e.g., cozy, outdoor, romantic, peaceful, work-friendly),
  location cues (e.g., "in Dadar"), and any budget / group size.
- Treat vibe primarily as a signal found in the Review text. Reviews may be informal, ungrammatical, use slang/emojis—interpret intent and sentiment robustly.

Budget interpretation
- The Cost field is cost-for-two.
- If the user gives a total budget and a group size (e.g., "₹2400 for six people"), convert to a per-two budget: (total ÷ people) × 2 = per-two.
- Apply all cost filtering only against this per-two figure.
- If no budget is given, do not filter or exclude high-cost cafés.
- If nothing fits strictly within the per-two budget, say so upfront and then present the best close matches by rating, explicitly noting the relaxed cost constraint.
- In justifications, always compare each café's cost-for-two to the computed per-two budget (not the total budget). If it exceeds, state that clearly.

Matching rules
- Cuisine: match case-insensitively if the target cuisine appears anywhere in the café's Cuisine list.
- Location: match user location (e.g., "Dadar") against the Location field (case-insensitive substring match is acceptable).
- Vibe: match vibe / ambience terms from the user request and infer from review text (and, secondarily, any hints in metadata). When inferring vibe, weight review evidence by:
  - Recency (see recency parsing below),
  - Helpful_Count (more helpful votes = more weight),
  - Followers of the reviewer (more followers = more weight).

Recency parsing & weighting
- Consider today's date as reference.
- The dataset's Posted field may be relative (e.g., "4 days ago", "one month ago", "9 months ago") or absolute (e.g., "Apr 01, 2023").
- As per that, get the date and check the recency when compared to today.
- The crux is that - if user requested vibe or ambience hints are present in more recent reviews, then they are to be more valued.
- Use this whole to strengthen or weaken vibe signals and for tie-breaking.

Apply only the filters the user mentions. If multiple filters are provided, apply all of them.

Deduplication & entity handling
- If a café appears multiple times due to multiple reviews, treat it as one entity for ranking and output. For vibe inference, consider the essence of all its reviews, with added weight per the recency/helpful/followers rule above.
- Use the café's main metadata plus the most relevant (and weighted) reviews as evidence in the justification.
- If the same café name appears in different locations (branches), treat each branch as a distinct café.

Ranking
- After filtering, rank cafés strictly by Rating (descending).
- If ratings tie or are very close, break ties using:
  1) strength of cuisine/location/vibe match (with vibe evidence weighted by recency, helpful votes, and reviewer followers),
  2) closeness to the per-two budget (when a budget applies),
  3) overall recency and credibility of relevant reviews.

Output Policy:
- Begin with a concise intro explaining which filters were applied and how ranking was done; explicitly note any relaxed/partially met criteria (e.g., budget or vibe).
- Provide up to 5 recommendations.
- For each café, include exactly:
  • Name
  • Cuisine
  • Location
  • Cost for two
  • Rating
  • Justification — brief, specific, and grounded in metadata + review evidence (no generic claims). If Cost for two exceeds the computed per-two budget, say so.
- Ordering: sort by Rating (highest → lowest) by default **after all the filterings based on user's request**. If the user explicitly requests a different order or strongly emphasizes a feature (e.g., ambience, price, cuisine), prioritize that feature instead. When deviating, state the reason in the intro.
- If no cafés fully match the user's request and you are suggesting alternatives, maintain a strict rating order of presenting the options (highest → lowest).
- Data integrity: do not invent data; use only the provided dataset and reviews.
"""

    user_prompt = f"""
User query:
{user_query}

Relevant cafés data:
{context_text}
"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Mumbai Café AI Recommender", layout="wide")

st.title("☕ Mumbai Café Recommender")
st.write("Ask in natural language, e.g.:")
st.code("Cozy South Indian café with Italian food in Dadar under ₹800 for four people")

query = st.text_input("Enter your requirement:")

if st.button("Get Recommendations") and query.strip():
    with st.spinner("🔍 Retrieving and analyzing..."):
        retrieved = retrieve(query)
        if not retrieved:
            st.warning("No matching cafés found.")
        else:
            response_text = generate_llm_recommendations(query, retrieved)
            st.markdown("### 📝 Recommendations")
            st.markdown(response_text)
