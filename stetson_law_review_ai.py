"""
⚖️ Stetson Law Review AI Assistant — Streamlit Cloud Version
Elegant, academic, and functional. Auto-downloads volumes.zip from Google Drive.
If Google Drive fails, shows clear guidance (no manual upload widget).
"""

# ========== Imports ==========
import os, re, io, time, zipfile, requests, csv, itertools
from pathlib import Path
from datetime import datetime, date
import streamlit as st
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader

# ========== Configuration ==========
APP_TITLE = "⚖️ Stetson Law Review AI Assistant"
PRIMARY, GOLD, IVORY = "#0A3D62", "#C49E56", "#F5F3EE"
CONTACT_EMAIL = "lreview@law.stetson.edu"
DISCLAIMER = "Unofficial academic project by Sarthak Pattnaik using public Stetson Law Review archives."

ROOT = Path(".").resolve()
VOLUMES_DIR = ROOT / "volumes"
INDEX_DIR = ROOT / "stetson_index"
INDEX_DIR.mkdir(exist_ok=True)
DOWNLOAD_LOG = ROOT / "downloads.csv"

# Google Drive link (use your own file ID)
DRIVE_FILE_ID = "1kR1Shuy9JYIWUSPZPYKr1iiyOgjRjOcx"
DRIVE_DL_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"

# ========== Page Setup ==========
st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="wide")
st.markdown(f"""
<style>
html,body,[class*="css"]{{background:{IVORY};color:{PRIMARY};font-family:Georgia,serif;}}
.card{{background:#fff;border:1px solid {GOLD};border-radius:14px;padding:18px;margin-bottom:12px;
box-shadow:0 3px 10px rgba(0,0,0,.08);}}
.score-badge{{float:right;background:{GOLD};color:{PRIMARY};padding:3px 8px;border-radius:8px;font-size:12px;font-weight:700;}}
.meta{{color:#666;font-size:13px;margin-bottom:8px;}}
section[data-testid="stSidebar"]{{background-color:{PRIMARY};color:white;}}
section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{{color:{GOLD};}}
</style>
""", unsafe_allow_html=True)

# ========== Sidebar ==========
with st.sidebar:
    st.markdown("### 📬 Contact")
    st.markdown(f"- **Email:** {CONTACT_EMAIL}")
    st.markdown("- **Institution:** Stetson University College of Law")
    st.markdown("---")
    st.caption(f"⚖️ {DISCLAIMER}")

# ========== Helper Functions ==========
def gdrive_download(file_id: str, dest: Path):
    """Download volumes.zip from Google Drive and extract to destination."""
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        with requests.Session() as s:
            r = s.get(url, stream=True)
            token = None
            for k, v in r.cookies.items():
                if k.startswith("download_warning"):
                    token = v
            if token:
                r = s.get(url + f"&confirm={token}", stream=True)
            data = io.BytesIO()
            for chunk in r.iter_content(32768):
                if chunk:
                    data.write(chunk)
            data.seek(0)
        with zipfile.ZipFile(data, "r") as zf:
            zf.extractall(dest)
        return True
    except Exception as e:
        st.warning(f"⚠️ Auto-download failed: {e}")
        return False

def ensure_volumes():
    """Ensure Volume folders exist with PDFs."""
    VOLUMES_DIR.mkdir(exist_ok=True)
    pdfs = list(VOLUMES_DIR.rglob("*.pdf"))
    if not pdfs:
        st.info("📦 No PDFs found. Attempting to download volumes.zip from Google Drive…")
        ok = gdrive_download(DRIVE_FILE_ID, VOLUMES_DIR)
        if ok:
            st.success("✅ Volumes downloaded and extracted successfully.")
            pdfs = list(VOLUMES_DIR.rglob("*.pdf"))
        else:
            st.error("❌ Could not download automatically. Please ensure volumes.zip <200 MB or host externally.")
            st.stop()
    return pdfs

def extract_text(pdf: Path):
    try:
        reader = PdfReader(str(pdf))
        text = " ".join(p.extract_text() or "" for p in reader.pages)
        return text[:100000]
    except Exception:
        return ""

def prettify(name): return re.sub(r"[_\-]+", " ", Path(name).stem).title()
def extract_volume(name): 
    m = re.search(r"Volume\s*\d+", str(name), re.I)
    return m.group(0) if m else "Unknown Volume"
def format_bytes(n): return f"{n/1024/1024:.1f} MB" if n else "—"

# ========== Index Building ==========
@st.cache_resource(show_spinner=True)
def build_index():
    pdfs = ensure_volumes()
    if not pdfs:
        st.stop()
    st.info(f"📚 Building semantic index from {len(pdfs)} PDFs…")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts, meta = [], []
    for pdf in pdfs:
        txt = extract_text(pdf)
        if not txt.strip():
            continue
        for i in range(0, len(txt), 1000):
            chunk = txt[i:i+1000]
            if len(chunk) < 400:
                continue
            texts.append(chunk)
            meta.append({"path": str(pdf), "file": pdf.name})
    emb = model.encode(texts, show_progress_bar=True, batch_size=64)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb.astype("float32"))
    st.success("✅ Index built successfully.")
    return index, {"texts": texts, "meta": meta}, model

# ========== Summarization ==========
def summarize(text, model, n=8):
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if len(s.strip()) > 40]
    if not sents:
        return "Summary unavailable."
    emb = model.encode(sents)
    sc = (emb @ emb.T).sum(axis=1)
    top = [s for s, _ in sorted(zip(sents, sc), key=lambda z: z[1], reverse=True)[:n]]
    return " ".join(top)

# ========== Downloads ==========
def log_download(path, title):
    new = not DOWNLOAD_LOG.exists()
    with open(DOWNLOAD_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "date", "volume", "title", "path"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), date.today(), extract_volume(path), title, path])

def read_downloads():
    if not DOWNLOAD_LOG.exists():
        return pd.DataFrame(columns=["timestamp", "date", "volume", "title", "path"])
    try:
        return pd.read_csv(DOWNLOAD_LOG)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "date", "volume", "title", "path"])

# ========== Header ==========
st.markdown(f"<h1 style='text-align:center;'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.caption("Elegant, academic, and intuitive — built for Stetson Law students.")
st.markdown("---")

# ========== Build / Load ==========
index, store, model = build_index()
pdf_count = len({m["path"] for m in store["meta"]})
st.success(f"📚 Indexed {pdf_count} PDFs successfully.")

# ========== Tabs ==========
tab_search, tab_insights, tab_about = st.tabs(["🔍 Search", "📊 Insights", "📬 About"])

# --- SEARCH TAB ---
with tab_search:
    st.subheader("🔎 Search the Archive")
    query = st.text_input("Enter topic, author, or question:", placeholder="e.g., Privacy law, AI in courts, First Amendment")
    if query:
        with st.spinner("Searching and summarizing…"):
            q_emb = model.encode([query])
            D, I = index.search(q_emb.astype("float32"), k=30)
            results = []
            for idx, dist in zip(I[0], D[0]):
                if idx == -1: continue
                m = store["meta"][idx]; t = store["texts"][idx]
                results.append((m, t, float(dist)))
            if not results:
                st.info("No matching results.")
            else:
                dmin, dmax = min(r[2] for r in results), max(r[2] for r in results)
                for i, (m, txt, dist) in enumerate(results[:12], 1):
                    rel = 100 * (1 - (dist - dmin) / (dmax - dmin + 1e-9))
                    title = prettify(m["file"]); vol = extract_volume(m["path"])
                    summary = summarize(txt, model)
                    st.markdown(f"""
                    <div class='card'>
                    <h4>{title}<span class='score-badge'>{rel:.1f}%</span></h4>
                    <p class='meta'>{vol}</p>
                    <p>{summary}</p></div>""", unsafe_allow_html=True)
                    with open(m["path"], "rb") as f:
                        if st.download_button("📥 Download PDF", f.read(), file_name=Path(m["path"]).name, mime="application/pdf", key=f"d-{i}-{time.time()}"):
                            log_download(m["path"], title)
                    st.divider()
    else:
        st.caption("Tip → Try 'privacy law', 'AI in courts', or 'First Amendment'.")

# --- INSIGHTS TAB ---
with tab_insights:
    st.subheader("📊 Download Trends")
    df = read_downloads()
    if df.empty:
        st.info("No downloads yet.")
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        top = df.groupby(["title", "volume"]).size().reset_index(name="downloads").sort_values("downloads", ascending=False).head(10)
        fig1 = px.bar(top, x="downloads", y="title", color="volume", orientation="h", title="Most Downloaded Articles")
        st.plotly_chart(fig1, use_container_width=True)
        daily = df.groupby("date").size().reset_index(name="downloads")
        fig2 = px.line(daily, x="date", y="downloads", markers=True, title="Downloads Over Time")
        st.plotly_chart(fig2, use_container_width=True)

# --- ABOUT TAB ---
with tab_about:
    st.subheader("📬 About")
    st.markdown(f"""
    <div class='card'>
    <p>This assistant helps students discover, summarize, and download Stetson Law Review articles.</p>
    <p>Features: semantic search (FAISS), AI-like summaries, local analytics dashboard, and direct downloads.</p>
    <p><b>Contact:</b> {CONTACT_EMAIL}</p>
    <p><i>{DISCLAIMER}</i></p>
    </div>
    """, unsafe_allow_html=True)
