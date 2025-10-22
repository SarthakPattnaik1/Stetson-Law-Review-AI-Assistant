"""
⚖️ Stetson Law Review AI Assistant — Local Semantic Search
Elegant, academic, and intuitive — built for Stetson Law students.
"""

# ================== DEPENDENCIES ==================
import importlib, subprocess, sys, os, io, re, time, itertools, csv
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import streamlit as st

def ensure(pkg):
    try:
        importlib.import_module(pkg.replace("-", "_"))
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for p in [
    "sentence-transformers",
    "faiss-cpu",
    "pypdf",
    "plotly",
]:
    ensure(p)

from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
import plotly.express as px

# ================== PATHS ==================
# Auto-detect volumes folder (works for both setups)
ROOT = Path(r"C:\Users\spattnaik\Downloads\stetson-law-review-ai")
POSSIBLE = [
    ROOT / "volumes",
    ROOT / "stetson_index" / "volumes",
]
VOLUMES_DIR = next((p for p in POSSIBLE if p.exists()), None)

if not VOLUMES_DIR:
    st.error("⚠️ No 'volumes' folder found. Place PDFs under Volume 30–55 folders.")
    st.stop()

VOLUME_PATHS = [p for i in range(30, 56) if (p := VOLUMES_DIR / f"Volume {i}").exists()]
INDEX_DIR = ROOT / "stetson_index"; INDEX_DIR.mkdir(exist_ok=True)
DOWNLOAD_LOG = ROOT / "downloads.csv"

# ================== HELPERS ==================
def get_all_pdfs():
    return list(itertools.chain.from_iterable(p.glob("*.pdf") for p in VOLUME_PATHS))

def prettify(name: str):
    return re.sub(r"[_\-]+", " ", Path(name).stem).title()

def extract_volume(path: str):
    m = re.search(r"Volume\s*\d+", path, re.I)
    return m.group(0) if m else "Unknown Volume"

def format_bytes(n):
    return f"{n/1024/1024:.1f} MB"

# ================== BUILD INDEX ==================
@st.cache_resource(show_spinner=False)
def build_index():
    pdfs = get_all_pdfs()
    if not pdfs:
        return None, None
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts, meta = [], []
    for pdf in pdfs:
        try:
            r = PdfReader(pdf)
            text = " ".join(p.extract_text() or "" for p in r.pages)
            chunks = [text[i:i+1000] for i in range(0, len(text), 900)]
            for c in chunks:
                texts.append(c)
                meta.append({"path": str(pdf), "file": pdf.name})
        except Exception:
            continue
    if not texts:
        return None, model
    emb = model.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb.astype("float32"))
    return (index, {"texts": texts, "meta": meta}), model

# ================== SUMMARIZER ==================
def summarize(text, model, n_sent=8):
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if len(s) > 40][:n_sent]
    return " ".join(sents) or "Summary unavailable."

# ================== LOG DOWNLOADS ==================
def log_download(path, title):
    DOWNLOAD_LOG.parent.mkdir(exist_ok=True)
    new = [datetime.now().isoformat(), date.today(), extract_volume(path), title, path]
    write_header = not DOWNLOAD_LOG.exists()
    with open(DOWNLOAD_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp","date","volume","title","path"])
        w.writerow(new)

def read_downloads():
    try: return pd.read_csv(DOWNLOAD_LOG)
    except: return pd.DataFrame(columns=["timestamp","date","volume","title","path"])

# ================== STREAMLIT UI ==================
st.set_page_config(APP_NAME:="⚖️ Stetson Law Review AI Assistant", layout="wide")
st.markdown(f"<h1 style='text-align:center;'>{APP_NAME}</h1>", unsafe_allow_html=True)
st.caption("Elegant, academic, and intuitive — built for Stetson Law students.")
st.markdown("---")

index_data, model = build_index()
if not index_data:
    st.error(f"⚠️ No PDFs found under {VOLUMES_DIR}. Please add PDF files in Volume folders.")
    st.stop()

index, meta = index_data
pdf_count = len(set(m['path'] for m in meta["meta"]))
st.success(f"📚 Detected {pdf_count} PDFs across Volumes 30 – 55.")

# ================== TABS ==================
tab_search, tab_insights, tab_about = st.tabs(["🔍 Search Articles", "📊 Insights", "📬 About"])

# ---------- SEARCH ----------
with tab_search:
    query = st.text_input("Enter topic, author, or question …", placeholder="e.g. Privacy law, AI in courts")
    if query:
        q_emb = model.encode([query])
        D, I = index.search(q_emb.astype("float32"), k=20)
        results = []
        for i, d in zip(I[0], D[0]):
            if i == -1: continue
            m = meta["meta"][i]
            results.append((m, float(d)))
        if not results:
            st.info("No results found.")
        else:
            dists = [r[1] for r in results]
            dmin, dmax = min(dists), max(dists)
            for m, d in results:
                rel = 100*(1-(d-dmin)/(dmax-dmin+1e-9))
                text = meta["texts"][meta["meta"].index(m)]
                st.markdown(f"### {prettify(m['file'])}  ({rel:.1f} % relevance)")
                st.caption(f"{extract_volume(m['file'])} • {format_bytes(os.path.getsize(m['path']))}")
                st.write(summarize(text, model))
                with open(m["path"], "rb") as f:
                    if st.download_button("📥 Download PDF", f.read(),
                                           file_name=m["file"],
                                           key=m["path"]):
                        log_download(m["path"], prettify(m["file"]))
                st.divider()
    else:
        st.caption("Tip: Try 'privacy law', 'AI in courts', or 'First Amendment'.")

# ---------- INSIGHTS ----------
with tab_insights:
    df = read_downloads()
    if df.empty:
        st.info("No downloads yet — charts will appear after you download PDFs.")
    else:
        top = df.groupby(["title","volume"]).size().reset_index(name="downloads").sort_values("downloads", ascending=False).head(10)
        st.plotly_chart(px.bar(top, x="downloads", y="title", color="volume", orientation="h", title="Most Downloaded Articles"), use_container_width=True)
        daily = df.groupby("date").size().reset_index(name="downloads")
        st.plotly_chart(px.line(daily, x="date", y="downloads", markers=True, title="Downloads Over Time"), use_container_width=True)

# ---------- ABOUT ----------
with tab_about:
    st.markdown("""
    ### 📬 About
    **Stetson Law Review AI Assistant** helps students discover, summarize, and download articles quickly.
    **Features**
    - Semantic search (FAISS)  
    - 8-line AI-like summaries  
    - Reading list & download tracking  
    - Analytics dashboard (trending topics & most downloaded)  

    📁 Place PDFs under `volumes/Volume 30 … Volume 55`  
    📧 Contact: lreview@law.stetson.edu  
    ⚖️ Unofficial academic project by Sarthak Pattnaik.
    """)
