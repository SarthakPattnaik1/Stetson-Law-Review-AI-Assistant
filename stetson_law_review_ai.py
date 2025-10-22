"""
⚖️ Stetson Law Review AI Assistant — Data & Insights Edition (Windows / Streamlit)
Elegant, semantic search + 8-line summaries + insights + downloads tracking.
Fully local (no API keys). Designed for Stetson Law students.

HOW TO RUN (PowerShell):
  cd "C:\Users\spattnaik\Downloads\stetson-law-review-ai"
  python -m streamlit run stetson_law_review_ai.py
"""

# ================== Auto-Install Dependencies (first run) ==================
import sys, subprocess, importlib

def ensure(pkg):
    """Ensure a Python package is installed and importable."""
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

# ================== Theming & Page Config ==================
APP_TITLE = "⚖️ Stetson Law Review AI Assistant"
PRIMARY, GOLD, IVORY, FONT = "#0A3D62", "#C49E56", "#FAFAF8", "Georgia, serif"
CONTACT_EMAIL = "lreview@law.stetson.edu"
DISCLAIMER = "Unofficial academic project by Sarthak Pattnaik using public Stetson Law Review archives."

st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="wide")
st.markdown(f"""
<style>
html, body, [class*="css"]  {{ background-color: {IVORY}; }}
h1, h2, h3, h4 {{ font-family:{FONT}; color:{PRIMARY}; letter-spacing:.2px; }}
hr.gold {{ border:0; border-top:1px solid {GOLD}; margin: 8px 0 24px; }}

/* Cards */
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

/* Grid */
.result-grid {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
@media (min-width: 1100px) {{ .result-grid {{ grid-template-columns: 1fr 1fr; }} }}

/* Inputs */
div[data-testid="stTextInput"] input {{
  border-radius: 28px !important; border: 1.5px solid {GOLD} !important;
  text-align: center !important; padding: 12px !important; font-size: 16px !important; color: {PRIMARY} !important;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{ background-color: {PRIMARY}; color: #fff; }}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
  color: {GOLD} !important; font-family: {FONT};
}}

.stButton>button, .stDownloadButton>button {{ border-radius: 22px; }}
.footer {{ text-align:center; font-size:13px; color:#555; border-top:1px solid {GOLD}; margin-top:36px; padding-top:12px; }}
.small {{ font-size:12px; color:#666; }}
</style>
""", unsafe_allow_html=True)

# ================== PATH CONFIG (Your setup) ==================
# This is YOUR volumes directory path:
VOLUMES_ROOT = Path(r"C:\Users\spattnaik\Downloads\stetson-law-review-ai\volumes")

# Base dir is the parent of 'volumes'
BASE_DIR = VOLUMES_ROOT.parent
INDEX_DIR = BASE_DIR / "stetson_index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_LOG = BASE_DIR / "downloads.csv"

# ================== Utilities ==================
def prettify_filename(name: str) -> str:
    return re.sub(r"[_\-]+", " ", Path(name).stem).strip().title()

def extract_volume(name: str) -> str:
    m = re.search(r"vol(?:ume)?\s*(\d+)", name.lower())
    return f"Volume {m.group(1)}" if m else "Unknown Volume"

def get_all_pdfs() -> List[Path]:
    """Recursively find ALL PDFs anywhere under the volumes root."""
    if not VOLUMES_ROOT.exists():
        return []
    return list(VOLUMES_ROOT.rglob("*.pdf"))

def safe_find_pdf_by_hint(hint: str) -> Optional[str]:
    """Fallback: locate a PDF by fuzzy filename match if exact path missing."""
    if not hint:
        return None
    clean = re.sub(r"[^a-z0-9]", "", hint.lower())
    for pdf in get_all_pdfs():
        cand = re.sub(r"[^a-z0-9]", "", pdf.stem.lower())
        if clean in cand or cand in clean:
            return str(pdf)
    return None

def format_bytes(n: int) -> str:
    try:
        return f"{n/1024/1024:.1f} MB"
    except Exception:
        return "—"

# ================== Download Tracking ==================
def log_download(path: str, title: str, vol: str):
    """Append a download event to downloads.csv."""
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

# ================== Summaries (8-line from top chunks) ==================
from typing import List
def summarize_multi_chunk(chunks: List[str], model: SentenceTransformer, n_sentences: int = 8) -> str:
    text = " ".join(chunks)
    sents = re.split(r'(?<=[.!?]) +', text)
    sents = [s.strip() for s in sents if len(s.strip()) > 30]
    if not sents:
        return "Summary unavailable."
    sents = sents[:200]  # cap
    emb = model.encode(sents, convert_to_tensor=True)
    sim = emb @ emb.T
    imp = sim.sum(dim=1).tolist()
    # pick top n by centrality, keep original order
    top_idx = sorted(sorted(range(len(sents)), key=lambda i: imp[i], reverse=True)[:n_sentences])
    return " ".join(sents[i] for i in top_idx)

# ================== FAISS Index ==================
@st.cache_resource(show_spinner=True)
def build_or_load_index() -> Optional[FAISS]:
    emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_path = INDEX_DIR / "index.faiss"
    if faiss_path.exists():
        try:
            return FAISS.load_local(str(INDEX_DIR), emb, allow_dangerous_deserialization=True)
        except Exception:
            pass

    pdfs = get_all_pdfs()
    if not pdfs:
        st.info(f"No PDFs found. Put files under:\n{VOLUMES_ROOT}")
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
    if not docs:
        st.info("Index could not be built (no readable PDFs).")
        return None
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
        st.download_button("⬇️ Download List (CSV)", df.to_csv(index=False).encode(),
                           "reading_list.csv", mime="text/csv")
        if st.button("🧹 Clear Reading List"):
            st.session_state.reading_list = []
            st.experimental_rerun()
    else:
        st.caption("No items yet.")
    st.markdown("---")
    # Small live snapshot
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
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        query = st.text_input("Enter topic, author, or question",
                              placeholder="e.g., Privacy law, AI in courts, First Amendment")
    with c2:
        # volumes from folder names under root, friendly labels
        vols_found = sorted({extract_volume(p.name) for p in VOLUMES_ROOT.glob("*") if p.is_dir()})
        vol_filter = st.selectbox("Filter by Volume", ["All Volumes"] + vols_found)
    with c3:
        sort_choice = st.selectbox("Sort by", ["Relevance", "Most Recent", "Volume (asc)"])

    if query:
        if not db:
            st.warning(f"No PDFs found. Upload files into:\n{VOLUMES_ROOT}")
        else:
            with st.spinner("Retrieving, ranking, and summarizing…"):
                raw = db.similarity_search_with_score(query, k=60)
                grouped: Dict[str, Dict] = {}
                for d, dist in raw:
                    # derive path robustly
                    path = d.metadata.get("path", "") or safe_find_pdf_by_hint(d.metadata.get("source_file", ""))
                    if not path:
                        continue
                    fname = Path(path).name
                    vol = extract_volume(fname)
                    if vol_filter != "All Volumes" and vol != vol_filter:
                        continue
                    entry = grouped.get(path)
                    if (entry is None) or (dist < entry["best_dist"]):
                        grouped[path] = {
                            "best_doc": d, "best_dist": float(dist), "vol": vol,
                            "file": fname, "path": path, "chunks": [d.page_content]
                        }
                    else:
                        # keep up to 3 chunks for better summary
                        if len(entry["chunks"]) < 3:
                            entry["chunks"].append(d.page_content)

                items = list(grouped.values())
                if not items:
                    st.info("No matching results after filters.")
                else:
                    # relevance normalization
                    dists = [x["best_dist"] for x in items]
                    if len(dists) > 1:
                        dmin, dmax = min(dists), max(dists)
                    else:
                        dmin, dmax = dists[0], dists[0] + 1e-6
                    for x in items:
                        norm = 1.0 - ((x["best_dist"] - dmin) / max(1e-9, (dmax - dmin)))
                        x["relevance"] = round(100 * max(0.0, min(1.0, norm)), 1)

                    # sorting
                    if sort_choice == "Most Recent":
                        items.sort(key=lambda x: Path(x["path"]).stat().st_mtime if Path(x["path"]).exists() else 0, reverse=True)
                    elif sort_choice == "Volume (asc)":
                        # extract numeric part
                        def volnum(v):
                            m = re.search(r"\d+", v)
                            return int(m.group()) if m else 0
                        items.sort(key=lambda x: volnum(x["vol"]))
                    else:
                        items.sort(key=lambda x: x["relevance"], reverse=True)

                    # mini results summary
                    top_volumes = pd.Series([i["vol"] for i in items]).value_counts().head(3)
                    st.success(f"Found **{len(items)}** matching articles • Top volumes: " +
                               ", ".join([f"{v} ({c})" for v, c in top_volumes.items()]))

                    # actions row
                    act1, act2 = st.columns([1,1])
                    with act1:
                        if st.button("➕ Add All to Reading List"):
                            for x in items[:16]:
                                title = prettify_filename(x["file"])
                                if title not in st.session_state.reading_list:
                                    st.session_state.reading_list.append(title)
                            st.success("Added visible results to reading list.")
                    with act2:
                        valid_paths = [x["path"] for x in items[:16] if Path(x["path"]).exists()]
                        if valid_paths:
                            mem = io.BytesIO()
                            with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                                for p in valid_paths:
                                    try:
                                        zf.write(p, arcname=os.path.basename(p))
                                    except Exception:
                                        pass
                            mem.seek(0)
                            st.download_button("📦 Download All (ZIP)", data=mem, file_name="stetson_results.zip", mime="application/zip")
                        else:
                            st.button("No PDFs to Zip", disabled=True)

                    st.markdown('<div class="result-grid">', unsafe_allow_html=True)
                    for i, x in enumerate(items[:16]):
                        title = prettify_filename(x["file"])
                        summary = summarize_multi_chunk(x["chunks"], embedder, n_sentences=8)
                        st.markdown(f"""
                        <div class="card">
                            <h4 style="margin:0">{title}<span class="score-badge">{x['relevance']}%</span></h4>
                            <p class="meta">{x['vol']} • {x['file']}</p>
                            <p style="text-align:justify">{summary}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        cA, cB, cC = st.columns([1,1,4])
                        with cA:
                            if Path(x["path"]).exists():
                                try:
                                    with open(x["path"], "rb") as f:
                                        if st.download_button("📥 Download", f.read(),
                                                              file_name=os.path.basename(x["path"]),
                                                              mime="application/pdf", key=f"dl-{i}-{time.time()}"):
                                            log_download(x["path"], title, x["vol"])
                                except Exception:
                                    st.button("File Locked", disabled=True)
                            else:
                                st.button("PDF Not Found", disabled=True)
                        with cB:
                            if st.button("➕ Read List", key=f"add-{i}-{time.time()}"):
                                if title not in st.session_state.reading_list:
                                    st.session_state.reading_list.append(title)
                                    st.success("✅ Added")
                        with cC:
                            st.caption(f"Path: {x['path']}")
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.caption("Tip: Try 'privacy law', 'AI in courts', or 'First Amendment'.")

# ================== RECENT ==================
with tab_recent:
    st.markdown("### 📰 Recently Added Articles")
    pdfs = sorted(get_all_pdfs(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    if not pdfs:
        st.info(f"No PDFs found yet. Add them under:\n{VOLUMES_ROOT}")
    else:
        for pdf in pdfs[:18]:
            title = prettify_filename(pdf.name)
            vol = extract_volume(pdf.name)
            mtime = datetime.fromtimestamp(pdf.stat().st_mtime).strftime("%b %d, %Y")
            st.markdown(f"""
            <div class="card">
                <h4 style="margin:0">{title}</h4>
                <p class="meta">{vol} • {pdf.name} • {mtime} • {format_bytes(pdf.stat().st_size)}</p>
            </div>
            """, unsafe_allow_html=True)
            with open(pdf, "rb") as f:
                if st.download_button("📥 Download PDF", data=f.read(),
                                      file_name=pdf.name, mime="application/pdf",
                                      key=f"recent-{pdf}-{time.time()}"):
                    log_download(str(pdf), title, vol)

# ================== INSIGHTS ==================
with tab_insights:
    st.markdown("### 📊 Library Insights & Trends")
    colA, colB = st.columns([1,1])
    dl_df = read_downloads_df()

    if dl_df.empty:
        st.info("No downloads logged yet — charts will appear after users download PDFs.")
    else:
        with colA:
            top_dl = dl_df.groupby(["title", "volume"]).size().reset_index(name="downloads") \
                          .sort_values("downloads", ascending=False).head(10)
            fig = px.bar(top_dl, x="downloads", y="title", color="volume", orientation="h",
                         title="Most Downloaded Articles", height=420)
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            daily = dl_df.groupby("date").size().reset_index(name="downloads")
            fig2 = px.line(daily, x="date", y="downloads", markers=True, title="Downloads Over Time", height=420)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### 🔎 Topic Trend by Volume")
    topic = st.text_input("Enter a keyword to see its trend across volumes",
                          placeholder="e.g., privacy, sentencing, evidence")
    if topic:
        counts: Dict[str, int] = {}
        for pdf in get_all_pdfs():
            try:
                loader = PyPDFLoader(str(pdf))
                pages = loader.load()
                first_pages = pages[:3] if len(pages) > 3 else pages
                text = " ".join(p.page_content for p in first_pages if p.page_content)
                n = len(re.findall(fr"\b{re.escape(topic)}\b", text, flags=re.IGNORECASE))
                vol = extract_volume(pdf.name)
                counts[vol] = counts.get(vol, 0) + n
            except Exception:
                pass
        if counts:
            dfc = pd.DataFrame({"volume": list(counts.keys()), "count": list(counts.values())})
            dfc["vnum"] = dfc["volume"].str.extract(r"(\d+)").astype(float)
            dfc = dfc.sort_values(["vnum", "volume"])
            fig3 = px.area(dfc, x="volume", y="count", title=f"Mentions of “{topic}” by Volume", height=380)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No matches for that keyword in sampled pages.")

    st.markdown("#### 🏷️ Top Title Keywords")
    titles = [prettify_filename(p.name) for p in get_all_pdfs()]
    if titles:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(stop_words="english", max_features=30)
            X = vec.fit_transform(titles)
            vocab = vec.get_feature_names_out()
            scores = X.sum(axis=0).A1
            top_kw = pd.DataFrame({"keyword": vocab, "score": scores}).sort_values("score", ascending=False).head(15)
            fig4 = px.bar(top_kw, x="keyword", y="score", title="Top Keywords in Article Titles", height=360)
            st.plotly_chart(fig4, use_container_width=True)
        except Exception:
            st.caption("Keyword extraction unavailable (scikit-learn issue).")

# ================== ABOUT ==================
with tab_about:
    st.markdown("### 📬 About")
    st.markdown(f"""
    <div class="card">
        <p><b>{APP_TITLE}</b> helps students discover, summarize, and download Stetson Law Review articles quickly.</p>
        <p><b>Features:</b> Semantic search (FAISS), AI-like summaries, downloads, reading list, and a full analytics dashboard (trending topics, most downloaded, and keyword trends).</p>
        <p><b>Storage:</b> Place PDFs anywhere under <code>{VOLUMES_ROOT}</code>. Filenames can be random.</p>
        <p><b>Contact:</b> {CONTACT_EMAIL}</p>
        <p class="meta">Local-first; no cloud API required.</p>
    </div>
    """, unsafe_allow_html=True)

# ================== Footer ==================
st.markdown(f"""
<div class="footer">
  ⚖️ <em>{DISCLAIMER}</em>
</div>
""", unsafe_allow_html=True)
