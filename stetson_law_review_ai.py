"""
⚖️ Stetson Law Review AI Assistant — Cloud/Local Hybrid
- Auto-downloads volumes.zip from Google Drive on first run (cloud-ready)
- Extracts PDFs into ./volumes/Volume 30..55 (random names supported)
- Builds semantic index (sentence-transformers + FAISS)
- Elegant UI: relevance %, 8-line summaries, downloads, insights
- OCR used only if Tesseract is available locally (safe on cloud)
"""

# ============== Imports (no auto-installs to keep Streamlit Cloud happy) ==============
import os, re, io, csv, time, itertools, zipfile, requests, platform
from pathlib import Path
from datetime import datetime, date

import streamlit as st
import pandas as pd
import plotly.express as px

# Vector search + PDF reading
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader

# Optional OCR (only if present locally; skipped on cloud)
USE_OCR = False
if platform.system() == "Windows":
    TESS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(TESS):
        try:
            import pytesseract
            from pdf2image import convert_from_path
            pytesseract.pytesseract.tesseract_cmd = TESS
            USE_OCR = True
        except Exception:
            USE_OCR = False

# ============== App/Theming ==============
APP_TITLE = "⚖️ Stetson Law Review AI Assistant"
PRIMARY, GOLD, IVORY = "#0A3D62", "#C49E56", "#F5F3EE"
CONTACT_EMAIL = "lreview@law.stetson.edu"
DISCLAIMER = "Unofficial academic project by Sarthak Pattnaik using public Stetson Law Review archives."

st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="wide")
st.markdown(
    f"""
<style>
html, body, [class*="css"] {{
  background-color: {IVORY};
  color: {PRIMARY};
  font-family: Georgia, serif;
}}
.card {{
  background: #fff;
  border-radius: 16px;
  padding: 18px 22px;
  border: 1px solid {GOLD};
  box-shadow: 0 3px 10px rgba(0,0,0,0.08);
  margin-bottom: 12px;
}}
.score-badge {{
  float:right;
  background:{GOLD};
  color:{PRIMARY};
  padding:3px 8px;
  border-radius:10px;
  font-size:12px;
  font-weight:700;
}}
.meta {{ color:#666; font-size:13px; margin-bottom:10px; }}
section[data-testid="stSidebar"] {{ background-color:{PRIMARY}; color:white; }}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{ color:{GOLD}; }}
div[data-testid="stTextInput"] input {{
  border-radius: 28px !important;
  border: 1.5px solid {GOLD} !important;
  text-align: center !important;
  padding: 12px !important;
  font-size: 16px !important;
  color: {PRIMARY} !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

# ============== Configuration (paths & Google Drive) ==============
ROOT = Path(".").resolve()
VOLUMES_DIR = ROOT / "volumes"                 # PDFs are extracted/loaded here
INDEX_DIR = ROOT / "stetson_index"             # for any cached assets if needed later
INDEX_DIR.mkdir(exist_ok=True)
DOWNLOAD_LOG = ROOT / "downloads.csv"

# 👉 YOUR DRIVE FILE ID (zip of the “volumes/Volume 30..55” folders)
DRIVE_FILE_ID = "1kR1Shuy9JYIWUSPZPYKr1iiyOgjRjOcx"  # provided by you
DRIVE_DL_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"

# ============== Utilities ==============
def gdrive_download_to_bytes(file_id: str) -> bytes:
    """
    Robust Google Drive download (handles 'virus scan too large' confirm token).
    """
    session = requests.Session()
    response = session.get(DRIVE_DL_URL, stream=True)
    token = None
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break
    if token:
        response = session.get(DRIVE_DL_URL + f"&confirm={token}", stream=True)
    content = io.BytesIO()
    for chunk in response.iter_content(32768):
        if chunk:
            content.write(chunk)
    content.seek(0)
    return content.read()

def ensure_volumes_present():
    """
    If ./volumes has no PDFs, auto-download volumes.zip from Google Drive and extract.
    """
    have_pdfs = any(VOLUMES_DIR.rglob("*.pdf"))
    if have_pdfs:
        return
    VOLUMES_DIR.mkdir(exist_ok=True, parents=True)
    st.info("📦 Downloading volumes archive from Google Drive…")
    try:
        data = gdrive_download_to_bytes(DRIVE_FILE_ID)
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            zf.extractall(VOLUMES_DIR)
        st.success("✅ Volumes downloaded and extracted.")
    except Exception as e:
        st.error(f"Failed to download/extract volumes: {e}")
        st.stop()

def get_all_pdfs():
    """
    Find PDFs only inside Volume 30..55 directories (random file names supported).
    Accepts nested structure inside the ZIP.
    """
    pdfs = []
    if not VOLUMES_DIR.exists():
        return pdfs
    # Accept nested (e.g., volumes/volumes/Volume 53) from some zips
    for i in range(30, 56):
        for candidate in VOLUMES_DIR.rglob(f"Volume {i}"):
            pdfs.extend(list(candidate.rglob("*.pdf")))
    return pdfs

def prettify(name: str) -> str:
    return re.sub(r"[_\-]+", " ", Path(name).stem).title()

def extract_volume(path_or_name: str) -> str:
    m = re.search(r"Volume\s*\d+", str(path_or_name), re.IGNORECASE)
    return m.group(0) if m else "Unknown Volume"

def format_bytes(n: int) -> str:
    try:
        return f"{n/1024/1024:.1f} MB"
    except Exception:
        return "—"

def log_download(path: str, title: str):
    write_header = not DOWNLOAD_LOG.exists()
    with open(DOWNLOAD_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "date", "volume", "title", "path"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), str(date.today()), extract_volume(path), title, path])

def read_downloads():
    if not DOWNLOAD_LOG.exists():
        return pd.DataFrame(columns=["timestamp", "date", "volume", "title", "path"])
    try:
        return pd.read_csv(DOWNLOAD_LOG)
    except Exception:
        return pd.DataFrame(columns=["timestamp", "date", "volume", "title", "path"])

# ============== Text Extraction ==============
def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Primary: PDF text layer via pypdf (cloud-safe).
    Fallback (local only): OCR via Tesseract if installed.
    """
    text = ""
    try:
        r = PdfReader(str(pdf_path))
        text = " ".join(p.extract_text() or "" for p in r.pages)
    except Exception:
        text = ""
    if len(text.strip()) > 100:
        return text

    if USE_OCR:
        # Only runs locally where Tesseract is available
        try:
            imgs = convert_from_path(str(pdf_path), dpi=200)
            ocr_text = []
            for img in imgs:
                ocr_text.append(pytesseract.image_to_string(img))
            return "\n".join(ocr_text)
        except Exception:
            return ""
    return ""

# ============== Build Semantic Index ==============
@st.cache_resource(show_spinner=False)
def build_index_and_model():
    ensure_volumes_present()
    pdfs = get_all_pdfs()
    if not pdfs:
        st.error("⚠️ No PDFs found under ./volumes/Volume 30..55. Upload a ZIP with that structure.")
        st.stop()

    st.info(f"📚 Building index from {len(pdfs)} PDFs (first run only)…")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts, meta = [], []
    for pdf in pdfs:
        try:
            full_text = extract_text_from_pdf(pdf)
            if not full_text.strip():
                continue
            # Chunking with overlap
            step = 900
            size = 1000
            for i in range(0, len(full_text), step):
                chunk = full_text[i:i+size]
                if len(chunk) < 400:  # skip tiny chunks
                    continue
                texts.append(chunk)
                meta.append({"path": str(pdf), "file": pdf.name})
        except Exception:
            continue

    if not texts:
        st.error("No readable text found in PDFs.")
        st.stop()

    emb = model.encode(texts, show_progress_bar=True, batch_size=64)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb.astype("float32"))
    st.success("✅ Semantic index built.")
    return (index, {"texts": texts, "meta": meta}), model

def summarize(text: str, model: SentenceTransformer, n_sent: int = 8) -> str:
    # Lightweight extractive summary: pick most informative sentences
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if len(s.strip()) > 40]
    if not sents:
        return "Summary unavailable."
    # Score sentences by mean cosine with all others (via embeddings)
    cap = sents[:200]  # cap for speed
    emb = model.encode(cap)
    score = (emb @ emb.T).sum(axis=1)
    top = [s for s, _ in sorted(zip(cap, score), key=lambda z: z[1], reverse=True)[:n_sent]]
    return " ".join(top)

# ============== Sidebar ==============
with st.sidebar:
    st.markdown("### 📬 Contact")
    st.markdown(f"- **Email:** {CONTACT_EMAIL}")
    st.markdown("- **Institution:** Stetson University College of Law")
    st.markdown("---")
    st.markdown(f"⚖️ *{DISCLAIMER}*")

# ============== Header ==============
st.markdown(f"<h1 style='text-align:center;'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.caption("Elegant, academic, and intuitive — built for Stetson Law students.")
st.markdown("---")

# ============== Build/Load ==============
(index, store), model = build_index_and_model()
pdf_count = len({m["path"] for m in store["meta"]})
st.success(f"📚 Detected {pdf_count} PDFs across Volume 30–55.")

# ============== Tabs ==============
tab_search, tab_browse, tab_insights, tab_about = st.tabs(
    ["🔍 Search", "📂 Browse by Volume", "📊 Insights", "📬 About"]
)

# ---------------------- SEARCH ----------------------
with tab_search:
    st.subheader("🔎 Search the Archive")
    query = st.text_input("Enter topic, author, or question", placeholder="e.g., Privacy law, AI in courts, Death penalty")
    if query:
        with st.spinner("Searching and summarizing…"):
            q_emb = model.encode([query])
            D, I = index.search(q_emb.astype("float32"), k=30)
            results = []
            seen_paths = set()
            # Group by article (first/best chunk per file)
            for idx, dist in zip(I[0], D[0]):
                if idx == -1:
                    continue
                meta = store["meta"][idx]
                path = meta["path"]
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                text = store["texts"][idx]
                results.append((meta, text, float(dist)))

            if not results:
                st.info("No results found.")
            else:
                dists = [r[2] for r in results]
                dmin, dmax = min(dists), max(dists) if len(dists) > 1 else (dists[0], dists[0] + 1e-6)
                for i, (m, txt, dist) in enumerate(results[:15], start=1):
                    rel = 100 * (1 - (dist - dmin) / (dmax - dmin + 1e-9))
                    title = prettify(m["file"])
                    vol = extract_volume(m["path"])
                    size = format_bytes(os.path.getsize(m["path"])) if os.path.exists(m["path"]) else "—"
                    summary = summarize(txt, model, n_sent=8)
                    st.markdown(
                        f"""
                        <div class="card">
                            <h4 style="margin:0">{title}
                                <span class="score-badge">{rel:.1f}%</span>
                            </h4>
                            <p class="meta">{vol} • {size}</p>
                            <p style="text-align:justify">{summary}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    # Download button
                    if os.path.exists(m["path"]):
                        with open(m["path"], "rb") as f:
                            if st.download_button(
                                "📥 Download PDF",
                                data=f.read(),
                                file_name=Path(m["path"]).name,
                                mime="application/pdf",
                                key=f"dl-{i}-{time.time()}",
                            ):
                                log_download(m["path"], title)
                    st.divider()
    else:
        st.caption("Tip → Try 'privacy law', 'AI in courts', or 'First Amendment'.")

# ---------------------- BROWSE ----------------------
with tab_browse:
    st.subheader("📂 Browse by Volume")
    # Detect any “Volume XX” dir under ./volumes (supports nested)
    volume_dirs = sorted({p for p in VOLUMES_DIR.rglob("Volume *") if p.is_dir()})
    if not volume_dirs:
        st.info("No Volume folders detected in /volumes.")
    else:
        names = [p.name for p in volume_dirs]
        choice = st.selectbox("Select a Volume", names)
        chosen = [p for p in volume_dirs if p.name == choice][0]
        pdfs = sorted(list(chosen.rglob("*.pdf")))
        if not pdfs:
            st.info("No PDFs found in that volume.")
        for pdf in pdfs[:50]:
            t = prettify(pdf.name)
            st.markdown(
                f"<div class='card'><b>{t}</b><br>{extract_volume(pdf)} • {format_bytes(pdf.stat().st_size)}</div>",
                unsafe_allow_html=True,
            )
            with open(pdf, "rb") as f:
                st.download_button("📥 Download PDF", f.read(), file_name=pdf.name, mime="application/pdf", key=f"b-{pdf}")

# ---------------------- INSIGHTS ----------------------
with tab_insights:
    st.subheader("📊 Download Trends & Analytics")
    df = read_downloads()
    if df.empty:
        st.info("No downloads yet — charts will appear after users download PDFs.")
    else:
        top = (
            df.groupby(["title", "volume"])
            .size()
            .reset_index(name="downloads")
            .sort_values("downloads", ascending=False)
            .head(10)
        )
        fig1 = px.bar(top, x="downloads", y="title", color="volume", orientation="h", title="Most Downloaded Articles")
        st.plotly_chart(fig1, use_container_width=True)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        daily = df.groupby("date").size().reset_index(name="downloads")
        fig2 = px.line(daily, x="date", y="downloads", markers=True, title="Downloads Over Time")
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------- ABOUT ----------------------
with tab_about:
    st.subheader("📬 Contact & About")
    st.markdown(
        f"""
        <div class='card'>
            <p><b>Email:</b> {CONTACT_EMAIL}</p>
            <p><b>Institution:</b> Stetson University College of Law</p>
            <p>This assistant helps students discover, summarize, and download Stetson Law Review articles using local, semantic search.</p>
            <p><i>{DISCLAIMER}</i></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
