"""
⚖️ Stetson Law Review AI Assistant — Final Edition
Elegant, academic, and intuitive — built for Stetson Law students.
Fully OCR-enabled for scanned PDFs across Volumes 30–55.
"""

# ================== IMPORTS ==================
import os, re, io, csv
from pathlib import Path
from datetime import datetime, date
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

# LangChain + NLP
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document
from sentence_transformers import SentenceTransformer

# OCR
import pytesseract
from pdf2image import convert_from_path

# Set tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ================== CONFIG ==================
APP_TITLE = "⚖️ Stetson Law Review AI Assistant"
PRIMARY, GOLD, IVORY = "#0A3D62", "#C49E56", "#F5F3EE"
CONTACT_EMAIL = "lreview@law.stetson.edu"

BASE_DIR = Path(__file__).resolve().parent
VOLUMES_DIR = BASE_DIR / "volumes"
VOLUME_PATHS = [VOLUMES_DIR / f"Volume {i}" for i in range(30, 56)]
INDEX_DIR = BASE_DIR / "stetson_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_LOG = BASE_DIR / "downloads.csv"

st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="wide")

# ================== STYLING ==================
st.markdown(f"""
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
}}
.score-badge {{
  float:right;
  background:{GOLD};
  color:{PRIMARY};
  padding:3px 8px;
  border-radius:8px;
  font-size:12px;
  font-weight:700;
}}
.meta {{ color:#666; font-size:13px; margin-bottom:10px; }}
</style>
""", unsafe_allow_html=True)

# ================== HELPERS ==================
def get_all_pdfs():
    """Collect all PDFs under Volume 30–55 folders."""
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

# ================== OCR TEXT EXTRACTION ==================
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF — fallback to OCR if no text layer."""
    try:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        text = " ".join([p.page_content for p in pages])
        if len(text.strip()) > 100:
            return text
    except Exception:
        pass

    # Fallback to OCR
    text_blocks = []
    try:
        images = convert_from_path(str(pdf_path), dpi=200)
        for img in images:
            text_blocks.append(pytesseract.image_to_string(img))
        return "\n".join(text_blocks)
    except Exception as e:
        print(f"OCR failed for {pdf_path}: {e}")
        return ""

# ================== INDEX BUILD ==================
@st.cache_resource
def build_index():
    pdfs = get_all_pdfs()
    if not pdfs:
        st.warning("⚠️ No PDFs found. Make sure Volume 30–55 folders contain files.")
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    all_docs = []
    for pdf in pdfs:
        text = extract_text_from_pdf(pdf)
        if not text.strip():
            continue
        chunks = splitter.split_text(text)
        for chunk in chunks:
            all_docs.append(Document(page_content=chunk, metadata={"path": str(pdf)}))
    if not all_docs:
        st.error("No readable text found in PDFs.")
        return None
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(all_docs, embed)
    db.save_local(str(INDEX_DIR))
    return db

@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ================== SUMMARIZER ==================
def summarize_text(text, model, n=8):
    sents = re.split(r'(?<=[.!?]) +', text)
    sents = [s.strip() for s in sents if len(s.strip()) > 40]
    if not sents:
        return "Summary unavailable."
    emb = model.encode(sents, convert_to_tensor=True)
    score = emb @ emb.T
    ranked = sorted(zip(sents, score.sum(1).tolist()), key=lambda x: x[1], reverse=True)
    return " ".join([s for s, _ in ranked[:n]])

# ================== APP BODY ==================
st.title(APP_TITLE)
st.caption("Elegant, academic, and intuitive — built for Stetson Law students.")
st.markdown("---")

embedder = load_embedder()
db = build_index()

pdf_count = len(get_all_pdfs())
st.success(f"📚 Detected {pdf_count} PDFs across Volumes 30–55.")

tabs = st.tabs(["🔍 Search", "📂 Browse by Volume", "📊 Insights", "📬 About"])

# === SEARCH TAB ===
with tabs[0]:
    st.subheader("🔎 Search the Archive")
    query = st.text_input("Enter topic, author, or question:", placeholder="e.g., Privacy law, Death penalty, AI in courts")
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
                    rel = round(max(0, 100 - score * 100), 1)
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
                                           key=f"dl-{i}")
                        log_download(path, title, vol)
    else:
        st.caption("Tip → Try 'privacy law', 'AI in courts', or 'death penalty'.")

# === BROWSE BY VOLUME TAB ===
with tabs[1]:
    st.subheader("📂 Browse Articles by Volume")
    vols = [v for v in VOLUME_PATHS if v.exists()]
    vol_choice = st.selectbox("Select a Volume", [v.name for v in vols])
    if vol_choice:
        folder = [v for v in vols if v.name == vol_choice][0]
        pdfs = list(folder.glob("*.pdf"))
        for pdf in pdfs:
            t = prettify_filename(pdf.name)
            st.markdown(f"<div class='card'><b>{t}</b><br>{extract_volume(pdf)} • {format_bytes(pdf.stat().st_size)}</div>", unsafe_allow_html=True)
            with open(pdf, "rb") as f:
                st.download_button("📥 Download PDF", f.read(), file_name=pdf.name, mime="application/pdf")

# === INSIGHTS TAB ===
with tabs[2]:
    st.subheader("📊 Download Trends & Analytics")
    df = read_downloads_df()
    if df.empty:
        st.info("No downloads yet.")
    else:
        top_dl = df.groupby(["title","volume"]).size().reset_index(name="downloads").sort_values("downloads", ascending=False).head(10)
        fig = px.bar(top_dl, x="downloads", y="title", color="volume", orientation="h", title="Most Downloaded Articles")
        st.plotly_chart(fig, use_container_width=True)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        daily = df.groupby("date").size().reset_index(name="downloads")
        fig2 = px.line(daily, x="date", y="downloads", markers=True, title="Downloads Over Time")
        st.plotly_chart(fig2, use_container_width=True)

# === ABOUT TAB ===
with tabs[3]:
    st.subheader("📬 Contact & About")
    st.markdown(f"""
    <div class='card'>
        <p><b>Email:</b> {CONTACT_EMAIL}</p>
        <p><b>Institution:</b> Stetson University College of Law</p>
        <p>This assistant helps students discover, summarize, and download Stetson Law Review articles using AI and OCR.</p>
        <p><i>Unofficial academic project by Sarthak Pattnaik.</i></p>
    </div>
    """, unsafe_allow_html=True)
