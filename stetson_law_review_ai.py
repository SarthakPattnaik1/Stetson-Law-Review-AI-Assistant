"""
⚖️ Stetson Law Review AI Assistant
Elegant, academic, and intuitive — built for Stetson Law students.
Supports semantic search, summaries, relevance ranking, downloads, and insights.
"""

# ================== Auto Install Dependencies ==================
import importlib, subprocess, sys

def ensure(pkg):
    """Auto-installs missing dependencies."""
    try:
        importlib.import_module(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

packages = [
    "streamlit", "torch", "sentence-transformers", "faiss-cpu",
    "langchain", "langchain-community", "langchain-text-splitters",
    "pypdf", "pandas", "plotly"
]
for p in packages:
    ensure(p)

# ================== Imports ==================
import os, re, io, csv, time, zipfile
from pathlib import Path
from datetime import datetime, date
import streamlit as st
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer

# --- Fix for LangChain version differences ---
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.docstore.document import Document
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.vectorstores import FAISS
    from langchain.docstore.document import Document
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# ================== Paths ==================
BASE_DIR = Path(r"C:\Users\spattnaik\Downloads\stetson-law-review-ai")
VOLUMES_ROOT = BASE_DIR / "volumes"
VOLUME_FOLDERS = [VOLUMES_ROOT / f"Volume {i}" for i in range(30, 56)]
PDF_DIRS = [p for p in VOLUME_FOLDERS if p.exists()]
INDEX_DIR = BASE_DIR / "stetson_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_LOG = BASE_DIR / "downloads.csv"

# ================== App Constants ==================
APP_TITLE = "⚖️ Stetson Law Review AI Assistant"
PRIMARY, GOLD, IVORY = "#0A3D62", "#C49E56", "#F5F3EE"
CONTACT_EMAIL = "lreview@law.stetson.edu"
DISCLAIMER = "Unofficial academic project by Sarthak Pattnaik."

# ================== Streamlit Config ==================
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="⚖️")

st.markdown(
    f"""
    <style>
    html, body, [class*="css"]  {{ background-color: {IVORY}; color:{PRIMARY}; font-family: Georgia; }}
    h1, h2, h3, h4 {{ color: {PRIMARY}; font-family: Georgia; }}
    .card {{ background: white; border-radius: 14px; padding: 16px; margin: 8px 0; border: 1px solid {GOLD}; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
    .score-badge {{ float:right; background:{GOLD}; color:{PRIMARY}; padding:3px 8px; border-radius:8px; font-size:12px; }}
    .meta {{ color:#444; font-size:13px; }}
    </style>
    """,
    unsafe_allow_html=True
)

# ================== Utility Functions ==================
def get_all_pdfs():
    pdfs = []
    for vol_dir in PDF_DIRS:
        pdfs.extend(vol_dir.rglob("*.pdf"))
    return pdfs

def prettify_filename(name: str) -> str:
    name = os.path.splitext(name)[0]
    name = re.sub(r"[_\-]+", " ", name)
    return name.strip().title()

def extract_volume(name: str) -> str:
    m = re.search(r"Volume\s*(\d+)", name, re.IGNORECASE)
    return f"Volume {m.group(1)}" if m else "Unknown Volume"

def format_bytes(n: int) -> str:
    try:
        return f"{n/1024/1024:.1f} MB"
    except:
        return "—"

def log_download(path: str, title: str, vol: str):
    DOWNLOAD_LOG.parent.mkdir(parents=True, exist_ok=True)
    exists = DOWNLOAD_LOG.exists()
    with open(DOWNLOAD_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp", "date", "volume", "title", "path"])
        now = datetime.now()
        w.writerow([now.isoformat(timespec="seconds"), str(date.today()), vol, title, path])

def read_downloads_df() -> pd.DataFrame:
    if not DOWNLOAD_LOG.exists():
        return pd.DataFrame(columns=["timestamp", "date", "volume", "title", "path"])
    try:
        return pd.read_csv(DOWNLOAD_LOG)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "date", "volume", "title", "path"])

# ================== Embedding & Index ==================
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=True)
def build_or_load_index():
    pdfs = get_all_pdfs()
    if not pdfs:
        st.warning("⚠️ No PDFs found under volumes. Please add files and refresh.")
        return None
    index_path = INDEX_DIR / "faiss.index"
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    for pdf in pdfs:
        try:
            loader = PyPDFLoader(str(pdf))
            pages = loader.load()
            for chunk in splitter.split_documents(pages):
                docs.append(Document(page_content=chunk.page_content, metadata={"path": str(pdf)}))
        except Exception as e:
            print(f"Error reading {pdf}: {e}")
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embedder)
    db.save_local(str(index_path))
    return db

# ================== Summarization ==================
def summarize_text(text: str, model: SentenceTransformer, max_sentences=8) -> str:
    sents = re.split(r'(?<=[.!?]) +', text)
    sents = [s.strip() for s in sents if len(s.strip()) > 40]
    if not sents:
        return "Summary unavailable."
    scores = model.encode(sents, convert_to_tensor=True)
    importance = scores @ scores.T
    ranked = sorted(zip(sents, importance.sum(1).tolist()), key=lambda x: x[1], reverse=True)
    return " ".join(s for s, _ in ranked[:max_sentences])

# ================== Main Layout ==================
st.title(APP_TITLE)
st.caption("Elegant, academic, and intuitive — built for Stetson Law students.")
st.markdown("---")

db = build_or_load_index()
embedder = load_embedder()

tab_search, tab_recent, tab_insights, tab_about = st.tabs(
    ["🔍 Ask / Search", "📰 Recent Articles", "📊 Insights", "📬 About"]
)

# ================== SEARCH TAB ==================
with tab_search:
    st.subheader("🔎 Search the Archive")
    query = st.text_input("Enter topic, author, or question:", placeholder="e.g., Privacy law, AI in courts, First Amendment")

    if query and db:
        with st.spinner("Retrieving, ranking, and summarizing…"):
            results = db.similarity_search_with_score(query, k=15)
            if not results:
                st.info("No matching results.")
            else:
                st.success(f"Found {len(results)} relevant articles.")
                for i, (doc, score) in enumerate(results):
                    path = doc.metadata["path"]
                    if not os.path.exists(path):
                        continue
                    fname = Path(path).name
                    vol = extract_volume(str(path))
                    summary = summarize_text(doc.page_content, embedder)
                    relevance = max(0, round(100 - score * 100, 1))

                    st.markdown(f"""
                    <div class="card">
                        <h4>{prettify_filename(fname)} 
                        <span class="score-badge">{relevance}%</span></h4>
                        <p class="meta">{vol} • {format_bytes(Path(path).stat().st_size)}</p>
                        <p style="text-align:justify">{summary}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    with open(path, "rb") as f:
                        st.download_button("📥 Download PDF", f.read(), file_name=fname, mime="application/pdf", key=f"dl-{i}")
                    log_download(path, prettify_filename(fname), vol)
    else:
        st.caption("Tip: Try ‘privacy law’, ‘death penalty’, or ‘AI in courts’.")

# ================== RECENT TAB ==================
with tab_recent:
    st.subheader("📰 Recently Added Articles")
    pdfs = sorted(get_all_pdfs(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pdfs:
        st.info("No PDFs found.")
    else:
        for pdf in pdfs[:12]:
            title = prettify_filename(pdf.name)
            vol = extract_volume(pdf.name)
            st.markdown(f"<div class='card'><b>{title}</b><br>{vol} • {format_bytes(pdf.stat().st_size)}</div>", unsafe_allow_html=True)
            with open(pdf, "rb") as f:
                st.download_button("📥 Download PDF", f.read(), file_name=pdf.name, mime="application/pdf")

# ================== INSIGHTS TAB ==================
with tab_insights:
    st.subheader("📊 Download Trends")
    df = read_downloads_df()
    if df.empty:
        st.info("No downloads logged yet.")
    else:
        top_dl = df.groupby(["title", "volume"]).size().reset_index(name="downloads").sort_values("downloads", ascending=False).head(10)
        fig = px.bar(top_dl, x="downloads", y="title", color="volume", orientation="h", title="Most Downloaded Articles")
        st.plotly_chart(fig, use_container_width=True)

        daily = df.groupby("date").size().reset_index(name="downloads")
        fig2 = px.line(daily, x="date", y="downloads", title="Downloads Over Time", markers=True)
        st.plotly_chart(fig2, use_container_width=True)

# ================== ABOUT TAB ==================
with tab_about:
    st.subheader("📬 Contact & Info")
    st.markdown(f"""
    <div class="card">
        <p><b>Email:</b> {CONTACT_EMAIL}</p>
        <p><b>Institution:</b> Stetson University College of Law</p>
        <p><b>About:</b> This assistant helps students explore Stetson Law Review archives via semantic search and summaries.</p>
        <p><b>Disclaimer:</b> {DISCLAIMER}</p>
    </div>
    """, unsafe_allow_html=True)
