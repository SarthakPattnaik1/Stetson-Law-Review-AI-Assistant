"""
⚖️ Stetson Law Review AI Assistant — Cloud Edition
Elegant, semantic search + summaries + insights + download tracking.
Fully cloud-deployable (no local paths or installs).
Designed for Stetson Law students.
"""

# ================== Imports ==================
import os, re, io, time, itertools, zipfile, csv
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Optional

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

# ================== Constants / Paths ==================
APP_TITLE = "⚖️ Stetson Law Review AI Assistant"
PRIMARY, GOLD, IVORY, FONT = "#0A3D62", "#C49E56", "#FAFAF8", "Georgia, serif"

BASE_DIR = Path(".")
VOLUME_PATHS = [BASE_DIR / f"volumes/Volume {i}" for i in range(30, 56)]
PDF_DIRS: List[Path] = [p for p in VOLUME_PATHS if p.exists()]
INDEX_DIR = BASE_DIR / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_LOG = BASE_DIR / "downloads.csv"

CONTACT_EMAIL = "lreview@law.stetson.edu"
DISCLAIMER = "Unofficial academic project by Sarthak Pattnaik using public Stetson Law Review archives."

# ================== Page Config & CSS ==================
st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="wide")
st.markdown(f"""
<style>
html, body, [class*="css"]  {{ background-color: {IVORY}; }}
h1, h2, h3, h4 {{ font-family:{FONT}; color:{PRIMARY}; letter-spacing:.2px; }}
hr.gold {{ border:0; border-top:1px solid {GOLD}; margin: 8px 0 24px; }}
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
.meta {{ color:#666; font-size:13px; margin: 6px 0 10px; }}
.score-badge {{
  float:right; background:{GOLD}; color:{PRIMARY}; padding:3px 8px;
  border-radius:12px; font-size:12px; font-weight:700;
}}
.result-grid {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
@media (min-width: 1100px) {{ .result-grid {{ grid-template-columns: 1fr 1fr; }} }}
div[data-testid="stTextInput"] input {{
  border-radius: 28px !important; border: 1.5px solid {GOLD} !important;
  text-align: center !important; padding: 12px !important; font-size: 16px !important; color: {PRIMARY} !important;
}}
section[data-testid="stSidebar"] {{ background-color: {PRIMARY}; color: #fff; }}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
  color: {GOLD} !important; font-family: {FONT};
}}
.stButton>button, .stDownloadButton>button {{ border-radius: 22px; }}
.footer {{ text-align:center; font-size:13px; color:#555; border-top:1px solid {GOLD}; margin-top:36px; padding-top:12px; }}
.small {{ font-size:12px; color:#666; }}
</style>
""", unsafe_allow_html=True)

# ================== Utilities ==================
def prettify_filename(name: str) -> str:
    return re.sub(r"[_\-]+", " ", Path(name).stem).strip().title()

def extract_volume(name: str) -> str:
    m = re.search(r"vol(?:ume)?\s*(\d+)", name.lower())
    return f"Volume {m.group(1)}" if m else "Unknown Volume"

def get_all_pdfs() -> List[Path]:
    return list(itertools.chain.from_iterable([list(p.glob("*.pdf")) for p in PDF_DIRS]))

def format_bytes(n: int) -> str:
    try: return f"{n/1024/1024:.1f} MB"
    except: return "—"

def read_downloads_df() -> pd.DataFrame:
    if not DOWNLOAD_LOG.exists():
        return pd.DataFrame(columns=["timestamp","date","volume","title","path"])
    try:
        return pd.read_csv(DOWNLOAD_LOG)
    except Exception:
        return pd.DataFrame(columns=["timestamp","date","volume","title","path"])

def log_download(path: str, title: str, vol: str):
    DOWNLOAD_LOG.parent.mkdir(parents=True, exist_ok=True)
    exists = DOWNLOAD_LOG.exists()
    with open(DOWNLOAD_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp","date","volume","title","path"])
        now = datetime.now()
        w.writerow([now.isoformat(timespec="seconds"), str(date.today()), vol, title, path])

# ================== Summaries ==================
def summarize_multi_chunk(chunks: List[str], model: SentenceTransformer, n_sentences: int = 8) -> str:
    text = " ".join(chunks)
    sents = re.split(r'(?<=[.!?]) +', text)
    sents = [s.strip() for s in sents if len(s.strip()) > 30]
    if not sents: return "Summary unavailable."
    sents = sents[:200]
    emb = model.encode(sents, convert_to_tensor=True)
    sim = emb @ emb.T
    imp = sim.sum(dim=1).tolist()
    top_idx = sorted(sorted(range(len(sents)), key=lambda i: imp[i], reverse=True)[:n_sentences])
    return " ".join(sents[i] for i in top_idx)

# ================== FAISS Index ==================
@st.cache_resource(show_spinner=True)
def build_or_load_index() -> Optional[FAISS]:
    emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    if (INDEX_DIR / "index.faiss").exists():
        try:
            return FAISS.load_local(str(INDEX_DIR), emb, allow_dangerous_deserialization=True)
        except Exception:
            pass
    pdfs = get_all_pdfs()
    if not pdfs:
        st.info("No PDFs found. Upload PDFs under volumes/Volume XX folders to enable search.")
        return None
    st.info(f"📚 Building semantic index from {len(pdfs)} PDFs… (first run)")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = []
    progress = st.progress(0.0, text="Indexing PDFs…")
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
    st.success("✅ Index ready.")
    return db

@st.cache_resource
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")

# ================== Header ==================
st.markdown(f"<h1 style='text-align:center'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#555;'>Discover, analyze, and download Stetson Law Review articles.</p>", unsafe_allow_html=True)
st.markdown('<hr class="gold"/>', unsafe_allow_html=True)

# ================== Sidebar ==================
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
        st.download_button("⬇️ Download List (CSV)", df.to_csv(index=False).encode(), "reading_list.csv", mime="text/csv")
        if st.button("🧹 Clear Reading List"):
            st.session_state.reading_list = []
            st.experimental_rerun()
    else:
        st.caption("No items yet.")
    st.markdown("---")
    dl_df = read_downloads_df()
    total_dl = int(dl_df.shape[0])
    top_vol = dl_df["volume"].value_counts().idxmax() if not dl_df.empty else "—"
    st.markdown("### 📈 Insights (Live)")
    st.markdown(f"- Total Downloads: **{total_dl}**")
    st.markdown(f"- Most Active Volume: **{top_vol}**")
    st.caption(f"⚖️ {DISCLAIMER}")

# ================== Tabs ==================
tab_search, tab_recent, tab_insights, tab_about = st.tabs(
    ["🧠 Ask / Search", "📰 Recent", "📊 Insights", "📬 About"]
)

db, embedder = build_or_load_index(), load_embedder()

# ================== SEARCH ==================
with tab_search:
    st.markdown("### 🔎 Ask or Search the Archive")
    query = st.text_input("Enter topic, author, or question", placeholder="e.g., Privacy law, AI in courts, First Amendment")
    if query:
        if not db:
            st.warning("No PDFs found. Upload files to your Volume folders and refresh.")
        else:
            with st.spinner("Retrieving and summarizing…"):
                raw = db.similarity_search_with_score(query, k=20)
                grouped = {}
                for d, dist in raw:
                    path = d.metadata.get("path", "")
                    if not path: continue
                    fname = Path(path).name
                    vol = extract_volume(fname)
                    entry = grouped.get(path)
                    if (entry is None) or (dist < entry["best_dist"]):
                        grouped[path] = {"best_doc": d, "best_dist": float(dist), "vol": vol, "file": fname, "path": path, "chunks": [d.page_content]}
                items = list(grouped.values())
                if not items:
                    st.info("No matching results.")
                else:
                    st.markdown('<div class="result-grid">', unsafe_allow_html=True)
                    for i, x in enumerate(items[:12]):
                        title = prettify_filename(x["file"])
                        summary = summarize_multi_chunk(x["chunks"], embedder, n_sentences=6)
                        st.markdown(f"""
                        <div class="card">
                            <h4 style="margin:0">{title}</h4>
                            <p class="meta">{x['vol']} • {x['file']}</p>
                            <p style="text-align:justify">{summary}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.caption("Tip: Try 'privacy law', 'AI in courts', or 'First Amendment'.")

# ================== RECENT ==================
with tab_recent:
    st.markdown("### 📰 Recently Added Articles")
    pdfs = sorted(get_all_pdfs(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    if not pdfs:
        st.info("No PDFs found yet. Add them to /volumes folders.")
    else:
        for pdf in pdfs[:12]:
            title = prettify_filename(pdf.name)
            vol = extract_volume(pdf.name)
            mtime = datetime.fromtimestamp(pdf.stat().st_mtime).strftime("%b %d, %Y")
            st.markdown(f"""
            <div class="card">
                <h4 style="margin:0">{title}</h4>
                <p class="meta">{vol} • {pdf.name} • {mtime} • {format_bytes(pdf.stat().st_size)}</p>
            </div>
            """, unsafe_allow_html=True)

# ================== INSIGHTS ==================
with tab_insights:
    st.markdown("### 📊 Library Insights & Trends")
    dl_df = read_downloads_df()
    if dl_df.empty:
        st.info("No downloads logged yet.")
    else:
        top_dl = dl_df.groupby(["title","volume"]).size().reset_index(name="downloads").sort_values("downloads", ascending=False).head(10)
        fig = px.bar(top_dl, x="downloads", y="title", color="volume", orientation="h", title="Most Downloaded Articles")
        st.plotly_chart(fig, use_container_width=True)

# ================== ABOUT ==================
with tab_about:
    st.markdown("### 📬 About")
    st.markdown(f"""
    <div class="card">
        <p><b>{APP_TITLE}</b> helps students discover, summarize, and analyze Stetson Law Review articles.</p>
        <p><b>Features:</b> Semantic search (FAISS), AI-like summaries, downloads, reading list, and analytics.</p>
        <p><b>Storage:</b> Place PDFs under <code>volumes/Volume 30 … Volume 55</code>. Filenames can be random.</p>
        <p><b>Contact:</b> {CONTACT_EMAIL}</p>
        <p class="meta">{DISCLAIMER}</p>
    </div>
    """, unsafe_allow_html=True)

# ================== Footer ==================
st.markdown(f"""
<div class="footer">
  ⚖️ <em>{DISCLAIMER}</em>
</div>
""", unsafe_allow_html=True)
