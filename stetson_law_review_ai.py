"""
⚖️ Stetson Law Review AI Assistant
Elegant, academic, and intuitive — built for Stetson Law students.
Semantic search, summaries, ranking, downloads, and trends.
"""

# ================== Imports ==================
import os, re, io, csv, time
from pathlib import Path
from datetime import datetime, date
import streamlit as st
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer

# --- LangChain version compatibility ---
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

# ================== Streamlit UI ==================
st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="⚖️")
st.markdown(f"""
<style>
html, body, [class*="css"]  {{ background-color: {IVORY}; color:{PRIMARY}; font-family: Georgia; }}
.card {{ background: white; border-radius: 14px; padding: 16px; margin: 8px 0; border: 1px solid {GOLD}; box-shadow: 0 2px 5px rgba(0,0,0,0.08); }}
.score-badge {{ float:right; background:{GOLD}; color:{PRIMARY}; padding:3px 8px; border-radius:8px; font-size:12px; }}
.meta {{ color:#555; font-size:13px; }}
</style>
""", unsafe_allow_html=True)

# ================== Utility Functions ==================
def get_all_pdfs():
    pdfs = []
    for vol_dir in PDF_DIRS:
        pdfs.extend(vol_dir.rglob("*.pdf"))
    return pdfs

def prettify_filename(name):
    return re.sub(r"[_\-]+", " ", os.path.splitext(name)[0]).title()

def extract_volume(name):
    m = re.search(r"Volume\s*(\d+)", name, re.IGNORECASE)
    return f"Volume {m.group(1)}" if m else "Unknown Volume"

def format_bytes(n):
    try:
        return f"{n/1024/1024:.1f} MB"
    except:
        return "—"

def log_download(path, title, vol):
    DOWNLOAD_LOG.parent.mkdir(parents=True, exist_ok=True)
    exists = DOWNLOAD_LOG.exists()
    with open(DOWNLOAD_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp", "date", "volume", "title", "path"])
        now = datetime.now()
        w.writerow([now.isoformat(timespec="seconds"), str(date.today()), vol, title, path])

def read_downloads_df():
    if not DOWNLOAD_LOG.exists():
        return pd.DataFrame(columns=["timestamp","date","volume","title","path"])
    try:
        return pd.read_csv(DOWNLOAD_LOG)
    except Exception:
        return pd.DataFrame(columns=["timestamp","date","volume","title","path"])

# ================== Models ==================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def build_or_load_index():
    pdfs = get_all_pdfs()
    if not pdfs:
        st.warning("⚠️ No PDFs found under 'volumes'. Please add PDF files in Volume folders.")
        return None
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    for pdf in pdfs:
        try:
            loader = PyPDFLoader(str(pdf))
            for chunk in splitter.split_documents(loader.load()):
                docs.append(Document(page_content=chunk.page_content, metadata={"path": str(pdf)}))
        except Exception:
            continue
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, emb)
    db.save_local(str(INDEX_DIR))
    return db

def summarize_text(text, model, n=8):
    sents = re.split(r'(?<=[.!?]) +', text)
    sents = [s.strip() for s in sents if len(s.strip()) > 40]
    if not sents:
        return "Summary unavailable."
    emb = model.encode(sents, convert_to_tensor=True)
    score = emb @ emb.T
    ranked = sorted(zip(sents, score.sum(1).tolist()), key=lambda x: x[1], reverse=True)
    return " ".join([s for s, _ in ranked[:n]])

# ================== Layout ==================
st.title(APP_TITLE)
st.caption("Elegant, academic, and intuitive — built for Stetson Law students.")
st.markdown("---")

db = build_or_load_index()
embedder = load_embedder()

tabs = st.tabs(["🔍 Search", "📰 Recent", "📊 Insights", "📬 About"])

# === SEARCH ===
with tabs[0]:
    st.subheader("🔎 Search the Archive")
    query = st.text_input("Enter topic, author, or question:", placeholder="e.g., Privacy law, AI in courts, First Amendment")
    if query and db:
        with st.spinner("Searching and summarizing..."):
            results = db.similarity_search_with_score(query, k=15)
            if not results:
                st.info("No matches found.")
            else:
                st.success(f"Found {len(results)} relevant articles.")
                for i, (doc, score) in enumerate(results):
                    path = doc.metadata["path"]
                    if not os.path.exists(path): continue
                    title = prettify_filename(Path(path).name)
                    vol = extract_volume(str(path))
                    rel = round(100 - score*100, 1)
                    summary = summarize_text(doc.page_content, embedder)

                    st.markdown(f"""
                    <div class='card'>
                        <h4>{title}<span class='score-badge'>{rel}%</span></h4>
                        <p class='meta'>{vol} • {format_bytes(Path(path).stat().st_size)}</p>
                        <p style='text-align:justify'>{summary}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    with open(path, "rb") as f:
                        st.download_button("📥 Download PDF", f.read(), file_name=Path(path).name, mime="application/pdf", key=f"d-{i}")
                    log_download(path, title, vol)
    else:
        st.caption("Tip: Try ‘privacy law’, ‘death penalty’, or ‘AI in courts’.")

# === RECENT ===
with tabs[1]:
    st.subheader("📰 Recently Added Articles")
    pdfs = sorted(get_all_pdfs(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pdfs:
        st.info("No PDFs found.")
    else:
        for pdf in pdfs[:10]:
            t = prettify_filename(pdf.name)
            v = extract_volume(pdf.name)
            st.markdown(f"<div class='card'><b>{t}</b><br>{v} • {format_bytes(pdf.stat().st_size)}</div>", unsafe_allow_html=True)
            with open(pdf, "rb") as f:
                st.download_button("📥 Download PDF", f.read(), file_name=pdf.name, mime="application/pdf")

# === INSIGHTS ===
with tabs[2]:
    st.subheader("📊 Download Trends")
    df = read_downloads_df()
    if df.empty:
        st.info("No downloads yet.")
    else:
        top_dl = df.groupby(["title","volume"]).size().reset_index(name="downloads").sort_values("downloads", ascending=False).head(10)
        st.plotly_chart(px.bar(top_dl, x="downloads", y="title", color="volume", orientation="h", title="Most Downloaded"), use_container_width=True)

        daily = df.groupby("date").size().reset_index(name="downloads")
        st.plotly_chart(px.line(daily, x="date", y="downloads", title="Downloads Over Time", markers=True), use_container_width=True)

# === ABOUT ===
with tabs[3]:
    st.subheader("📬 Contact & Info")
    st.markdown(f"""
    <div class='card'>
        <p><b>Email:</b> {CONTACT_EMAIL}</p>
        <p><b>Institution:</b> Stetson University College of Law</p>
        <p>This tool helps students search, summarize, and download Stetson Law Review articles.</p>
        <p><i>{DISCLAIMER}</i></p>
    </div>
    """, unsafe_allow_html=True)
