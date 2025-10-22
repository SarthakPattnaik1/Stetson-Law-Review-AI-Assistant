"""
⚖️ Stetson Law Review AI Assistant
Elegant, academic, and intuitive — built for Stetson Law students.
"""

# ================== IMPORTS ==================
import os, re, io, csv
from pathlib import Path
from datetime import datetime, date
import streamlit as st
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer

# --- LangChain imports (with fallback) ---
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

# ================== PATHS ==================
BASE_DIR = Path(__file__).resolve().parent        # dynamic base folder
VOLUMES_DIR = BASE_DIR / "volumes"
VOLUME_PATHS = [VOLUMES_DIR / f"Volume {i}" for i in range(30, 56)]
INDEX_DIR = BASE_DIR / "stetson_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_LOG = BASE_DIR / "downloads.csv"

# ================== UI CONFIG ==================
APP_TITLE = "⚖️ Stetson Law Review AI Assistant"
PRIMARY, GOLD, IVORY = "#0A3D62", "#C49E56", "#F5F3EE"
CONTACT_EMAIL = "lreview@law.stetson.edu"

st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="wide")
st.markdown(f"""
<style>
html, body, [class*="css"]  {{ background-color: {IVORY}; color:{PRIMARY}; font-family: Georgia; }}
.card {{
 background: #fff; border-radius: 14px; padding: 18px 22px; 
 border: 1px solid {GOLD}; box-shadow: 0 2px 10px rgba(0,0,0,0.08);
}}
.score-badge {{
 float:right; background:{GOLD}; color:{PRIMARY};
 padding:3px 8px; border-radius:8px; font-size:12px; font-weight:700;
}}
.meta {{ color:#666; font-size:13px; margin-bottom:10px; }}
</style>
""", unsafe_allow_html=True)

# ================== HELPERS ==================
def get_all_pdfs():
    """Recursively collect all PDFs under Volume 30–55."""
    pdfs = []
    for vol in VOLUME_PATHS:
        if vol.exists():
            pdfs.extend(list(vol.rglob("*.pdf")))
    return pdfs

def prettify_filename(name):
    return re.sub(r"[_\-]+", " ", Path(name).stem).title()

def extract_volume(path):
    m = re.search(r"Volume\s*(\d+)", str(path), re.IGNORECASE)
    return f"Volume {m.group(1)}" if m else "Unknown Volume"

def format_bytes(n):
    try:
        return f"{n/1024/1024:.1f} MB"
    except:
        return "—"

def log_download(path, title, vol):
    exists = DOWNLOAD_LOG.exists()
    with open(DOWNLOAD_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp","date","volume","title","path"])
        now = datetime.now()
        w.writerow([now.isoformat(timespec="seconds"), str(date.today()), vol, title, path])

def read_downloads_df():
    if not DOWNLOAD_LOG.exists():
        return pd.DataFrame(columns=["timestamp","date","volume","title","path"])
    try:
        return pd.read_csv(DOWNLOAD_LOG)
    except Exception:
        return pd.DataFrame(columns=["timestamp","date","volume","title","path"])

# ================== MODEL ==================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def build_or_load_index():
    pdfs = get_all_pdfs()
    if not pdfs:
        st.warning("⚠️ No PDFs found. Check that Volume 30–55 folders exist and contain PDFs.")
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    all_docs = []
    for pdf in pdfs:
        try:
            loader = PyPDFLoader(str(pdf))
            pages = loader.load()
            chunks = splitter.split_documents(pages)
            for c in chunks:
                c.metadata["path"] = str(pdf)
            all_docs.extend(chunks)
        except Exception:
            continue
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(all_docs, emb)
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

# ================== MAIN UI ==================
st.title(APP_TITLE)
st.caption("Elegant, academic, and intuitive — built for Stetson Law students.")
st.markdown("---")

embedder = load_embedder()
db = build_or_load_index()

pdf_count = len(get_all_pdfs())
st.success(f"📂 Detected {pdf_count} PDF files across Volumes 30 – 55.")

tabs = st.tabs(["🔍 Search", "📰 Recent", "📊 Insights", "📬 About"])

# === SEARCH TAB ===
with tabs[0]:
    st.subheader("🔎 Search the Archive")
    query = st.text_input("Enter topic, author, or question:",
                          placeholder="e.g., Privacy law, Death penalty, AI in courts")

    if query and db:
        with st.spinner("Searching and summarizing..."):
            results = db.similarity_search_with_score(query, k=15)
            if not results:
                st.info("No matches found.")
            else:
                for i, (doc, score) in enumerate(results):
                    path = doc.metadata.get("path", "")
                    if not os.path.exists(path):
                        continue
                    title = prettify_filename(Path(path).name)
                    vol = extract_volume(path)
                    rel = round(max(0, 100 - score*100), 1)
                    summary = summarize_text(doc.page_content, embedder)
                    st.markdown(f"""
                    <div class='card'>
                        <h4>{title}<span class='score-badge'>{rel}%</span></h4>
                        <p class='meta'>{vol} • {format_bytes(Path(path).stat().st_size)}</p>
                        <p style='text-align:justify'>{summary}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    with open(path, "rb") as f:
                        st.download_button("📥 Download PDF", f.read(),
                                           file_name=Path(path).name,
                                           mime="application/pdf",
                                           key=f"d-{i}")
                    log_download(path, title, vol)
    else:
        st.caption("Tip → Try 'privacy law', 'AI ethics', or 'death penalty'.")

# === RECENT TAB ===
with tabs[1]:
    st.subheader("📰 Recently Added Articles")
    pdfs = sorted(get_all_pdfs(),
                  key=lambda p: p.stat().st_mtime if p.exists() else 0,
                  reverse=True)
    if not pdfs:
        st.info("No PDFs found yet.")
    else:
        for pdf in pdfs[:10]:
            t = prettify_filename(pdf.name)
            v = extract_volume(pdf.name)
            st.markdown(f"<div class='card'><b>{t}</b><br>{v} • {format_bytes(pdf.stat().st_size)}</div>",
                        unsafe_allow_html=True)
            with open(pdf, "rb") as f:
                st.download_button("📥 Download PDF", f.read(),
                                   file_name=pdf.name, mime="application/pdf")

# === INSIGHTS TAB ===
with tabs[2]:
    st.subheader("📊 Download Insights")
    df = read_downloads_df()
    if df.empty:
        st.info("No downloads yet.")
    else:
        top_dl = (df.groupby(["title","volume"])
                    .size()
                    .reset_index(name="downloads")
                    .sort_values("downloads", ascending=False)
                    .head(10))
        st.plotly_chart(px.bar(top_dl, x="downloads", y="title", color="volume",
                               orientation="h", title="Most Downloaded Articles"),
                        use_container_width=True)

# === ABOUT TAB ===
with tabs[3]:
    st.subheader("📬 Contact & About")
    st.markdown(f"""
    <div class='card'>
        <p><b>Email:</b> {CONTACT_EMAIL}</p>
        <p><b>Institution:</b> Stetson University College of Law</p>
        <p>This assistant helps students discover, summarize, and download Stetson Law Review articles quickly.</p>
        <p><i>Unofficial academic project by Sarthak Pattnaik.</i></p>
    </div>
    """, unsafe_allow_html=True)
