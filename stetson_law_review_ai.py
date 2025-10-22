"""
⚖️ Stetson Law Review AI Assistant — Data & Insights Edition (Windows / Streamlit)
Elegant, semantic search + 8-line summaries + insights + downloads tracking.
Fully local (no API keys). Designed for Stetson Law students.

HOW TO RUN (PowerShell):
  cd "C:\Users\spattnaik\Downloads\stetson-law-review-ai"
  python -m streamlit run stetson_law_review_ai.py
"""

# ================== Auto-Install Dependencies ==================
import sys, subprocess, importlib

def ensure(pkg):
    try:
        importlib.import_module(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for p in [
    "streamlit", "torch", "sentence-transformers", "faiss-cpu",
    "langchain-community", "langchain-text-splitters", "pypdf",
    "pandas", "plotly", "scikit-learn"
]:
    ensure(p)

# ================== Imports ==================
import os, re, io, time, itertools, zipfile, csv
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Optional

import streamlit as st
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# ================== Theme ==================
APP_TITLE = "⚖️ Stetson Law Review AI Assistant"
PRIMARY, GOLD, IVORY, FONT = "#0A3D62", "#C49E56", "#FAFAF8", "Georgia, serif"
CONTACT_EMAIL = "lreview@law.stetson.edu"
DISCLAIMER = "Unofficial academic project by Sarthak Pattnaik using public Stetson Law Review archives."

st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="wide")
st.markdown(f"""
<style>
html, body, [class*="css"]  {{ background-color: {IVORY}; }}
h1, h2, h3, h4 {{ font-family:{FONT}; color:{PRIMARY}; }}
.card {{
  background: linear-gradient(180deg, #ffffff 0%, #fbfbfb 100%);
  border: 1px solid {GOLD};
  border-radius: 18px;
  padding: 18px;
  margin-bottom: 14px;
  box-shadow: 0 2px 12px rgba(10,61,98,.10);
  transition: transform .15s ease-in-out, box-shadow .15s ease-in-out;
}}
.card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 22px rgba(10,61,98,.18); }}
.score-badge {{
  float:right; background:{GOLD}; color:{PRIMARY}; padding:3px 8px;
  border-radius:12px; font-size:12px; font-weight:700;
}}
section[data-testid="stSidebar"] {{ background-color: {PRIMARY}; color: #fff; }}
</style>
""", unsafe_allow_html=True)

# ================== PATH CONFIG ==================
VOLUMES_ROOT = Path(r"C:\Users\spattnaik\Downloads\stetson-law-review-ai\volumes")  # ✅ FIXED PATH
BASE_DIR = VOLUMES_ROOT.parent
INDEX_DIR = BASE_DIR / "stetson_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_LOG = BASE_DIR / "downloads.csv"

# ================== UTILITIES ==================
def prettify_filename(name: str) -> str:
    return re.sub(r"[_\-]+", " ", Path(name).stem).strip().title()

def extract_volume(name: str) -> str:
    m = re.search(r"vol(?:ume)?\s*(\d+)", name.lower())
    return f"Volume {m.group(1)}" if m else "Unknown Volume"

def get_all_pdfs() -> List[Path]:
    if not VOLUMES_ROOT.exists():
        return []
    return list(VOLUMES_ROOT.rglob("*.pdf"))

def format_bytes(n: int) -> str:
    try: return f"{n/1024/1024:.1f} MB"
    except: return "—"

# ================== DOWNLOAD LOG ==================
def log_download(path: str, title: str, vol: str):
    exists = DOWNLOAD_LOG.exists()
    with open(DOWNLOAD_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp", "date", "volume", "title", "path"])
        now = datetime.now()
        w.writerow([now.isoformat(timespec="seconds"), str(date.today()), vol, title, path])

def read_downloads_df() -> pd.DataFrame:
    if not DOWNLOAD_LOG.exists():
        return pd.DataFrame(columns=["timestamp","date","volume","title","path"])
    try: return pd.read_csv(DOWNLOAD_LOG)
    except: return pd.DataFrame(columns=["timestamp","date","volume","title","path"])

# ================== SUMMARIES ==================
def summarize_multi_chunk(chunks: List[str], model: SentenceTransformer, n_sentences: int = 8) -> str:
    text = " ".join(chunks)
    sents = re.split(r'(?<=[.!?]) +', text)
    sents = [s.strip() for s in sents if len(s.strip()) > 30]
    if not sents:
        return "Summary unavailable."
    sents = sents[:200]
    emb = model.encode(sents, convert_to_tensor=True)
    sim = emb @ emb.T
    imp = sim.sum(dim=1).tolist()
    top_idx = sorted(sorted(range(len(sents)), key=lambda i: imp[i], reverse=True)[:n_sentences])
    return " ".join(sents[i] for i in top_idx)

# ================== FAISS INDEX ==================
@st.cache_resource(show_spinner=True)
def build_or_load_index() -> Optional[FAISS]:
    emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_path = INDEX_DIR / "index.faiss"
    if faiss_path.exists():
        try:
            return FAISS.load_local(str(INDEX_DIR), emb, allow_dangerous_deserialization=True)
        except:
            pass
    pdfs = get_all_pdfs()
    if not pdfs:
        st.info(f"No PDFs found under {VOLUMES_ROOT}")
        return None
    st.info(f"📚 Building semantic index from {len(pdfs)} PDFs… (first run)")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = []
    progress = st.progress(0.0)
    for i, pdf in enumerate(pdfs, 1):
        try:
            loader = PyPDFLoader(str(pdf))
            pages = loader.load()
            chunks = splitter.split_documents(pages)
            vol = extract_volume(pdf.name)
            for c in chunks:
                c.metadata.update({"source_file": pdf.name, "volume": vol, "path": str(pdf)})
            docs.extend(chunks)
        except Exception as e:
            st.warning(f"Skipping {pdf.name}: {e}")
        progress.progress(i/len(pdfs))
    db = FAISS.from_documents(docs, emb)
    db.save_local(str(INDEX_DIR))
    st.success("✅ Index built successfully.")
    return db

@st.cache_resource
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")

# ================== HEADER ==================
st.markdown(f"<h1 style='text-align:center'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#555;'>Discover, summarize, and download Stetson Law Review articles.</p>", unsafe_allow_html=True)
st.markdown('<hr class="gold"/>', unsafe_allow_html=True)

# ================== SIDEBAR ==================
st.session_state.setdefault("reading_list", [])
with st.sidebar:
    st.markdown("### 📬 Contact")
    st.markdown(f"- **Email:** {CONTACT_EMAIL}")
    st.markdown("- **Institution:** Stetson University College of Law")
    st.markdown("---")
    st.markdown("### 📚 Reading List")
    if st.session_state.reading_list:
        for item in st.session_state.reading_list:
            st.markdown(f"- {item}")
        df = pd.DataFrame({"Articles": st.session_state.reading_list})
        st.download_button("⬇️ Download List (CSV)", df.to_csv(index=False).encode(), "reading_list.csv")
        if st.button("🧹 Clear List"): 
            st.session_state.reading_list = []
            st.experimental_rerun()
    else:
        st.caption("No items yet.")
    st.markdown("---")
    dl_df = read_downloads_df()
    st.markdown("### 📈 Insights (Live)")
    st.markdown(f"- Total Downloads: **{len(dl_df)}**")
    top_vol = dl_df["volume"].value_counts().idxmax() if not dl_df.empty else "—"
    st.markdown(f"- Most Active Volume: **{top_vol}**")
    st.caption(f"⚖️ {DISCLAIMER}")

# ================== TABS ==================
tab_search, tab_recent, tab_insights, tab_about = st.tabs(["🧠 Ask / Search", "📰 Recent", "📊 Insights", "📬 About"])
db, embedder = build_or_load_index(), load_embedder()

# ================== SEARCH ==================
with tab_search:
    st.markdown("### 🔎 Ask or Search the Archive")
    query = st.text_input("Enter topic, author, or question", placeholder="e.g., Privacy law, AI in courts, First Amendment")
    if query:
        if not db:
            st.warning(f"No PDFs found under {VOLUMES_ROOT}")
        else:
            with st.spinner("Searching and summarizing..."):
                raw = db.similarity_search_with_score(query, k=40)
                results = []
                for d, dist in raw:
                    path = d.metadata.get("path", "")
                    if not os.path.exists(path): continue
                    results.append((d, dist, path))
                if not results:
                    st.info("No results found.")
                else:
                    st.markdown(f"### {len(results)} Articles Found")
                    for i, (doc, dist, path) in enumerate(results[:15]):
                        title = prettify_filename(Path(path).stem)
                        vol = extract_volume(title)
                        summary = summarize_multi_chunk([doc.page_content], embedder, n_sentences=8)
                        rel = round(100 - dist*100, 1)
                        st.markdown(f"""
                        <div class="card">
                            <h4>{title} <span class="score-badge">{rel}%</span></h4>
                            <p class="meta">{vol}</p>
                            <p>{summary}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        c1, c2 = st.columns([1,1])
                        with c1:
                            with open(path, "rb") as f:
                                if st.download_button("📥 Download", f.read(), file_name=os.path.basename(path),
                                                      mime="application/pdf", key=f"dl-{i}-{time.time()}"):
                                    log_download(path, title, vol)
                        with c2:
                            if st.button("➕ Add to List", key=f"add-{i}"):
                                if title not in st.session_state.reading_list:
                                    st.session_state.reading_list.append(title)
                                    st.success("✅ Added to Reading List")

# ================== RECENT ==================
with tab_recent:
    st.markdown("### 📰 Recently Added Articles")
    pdfs = sorted(get_all_pdfs(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    if not pdfs:
        st.info("No PDFs found.")
    for pdf in pdfs[:15]:
        title = prettify_filename(pdf.name)
        vol = extract_volume(pdf.name)
        mtime = datetime.fromtimestamp(pdf.stat().st_mtime).strftime("%b %d, %Y")
        st.markdown(f"<div class='card'><h4>{title}</h4><p class='meta'>{vol} • {mtime} • {format_bytes(pdf.stat().st_size)}</p></div>", unsafe_allow_html=True)
        with open(pdf, "rb") as f:
            if st.download_button("📥 Download PDF", f.read(), file_name=pdf.name, mime="application/pdf", key=f"recent-{pdf}-{time.time()}"):
                log_download(str(pdf), title, vol)

# ================== INSIGHTS ==================
with tab_insights:
    st.markdown("### 📊 Library Insights & Trends")
    dl_df = read_downloads_df()
    if dl_df.empty:
        st.info("No downloads yet — charts appear after first download.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            top_dl = dl_df.groupby(["title","volume"]).size().reset_index(name="downloads").sort_values("downloads",ascending=False).head(10)
            fig = px.bar(top_dl, x="downloads", y="title", color="volume", orientation="h", title="Most Downloaded")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            daily = dl_df.groupby("date").size().reset_index(name="downloads")
            fig2 = px.line(daily, x="date", y="downloads", markers=True, title="Downloads Over Time")
            st.plotly_chart(fig2, use_container_width=True)

# ================== ABOUT ==================
with tab_about:
    st.markdown("### 📬 About")
    st.markdown(f"""
    <div class='card'>
        <p><b>{APP_TITLE}</b> helps students discover and summarize Stetson Law Review articles using local AI search.</p>
        <p><b>Features:</b> Semantic search (FAISS), 8-line summaries, downloads, reading list, and insights dashboard.</p>
        <p><b>Storage:</b> Place PDFs in <code>{VOLUMES_ROOT}</code>.</p>
        <p><b>Contact:</b> {CONTACT_EMAIL}</p>
        <p class='meta'>Local-first; no cloud API required.</p>
    </div>
    """, unsafe_allow_html=True)

# ================== FOOTER ==================
st.markdown(f"<div style='text-align:center;margin-top:40px;color:#666;'>⚖️ {DISCLAIMER}</div>", unsafe_allow_html=True)
