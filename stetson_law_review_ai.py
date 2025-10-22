"""
⚖️ Stetson Law Review AI Assistant — Final Version
Elegant, semantic search + 8-line summaries + analytics + upload fallback.
Works locally and on Streamlit Cloud.
"""

# ========== Imports ==========
import os, re, io, csv, time, zipfile, requests, itertools
from pathlib import Path
from datetime import datetime, date
import streamlit as st
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader

# ========== Constants ==========
APP_TITLE = "⚖️ Stetson Law Review AI Assistant"
PRIMARY, GOLD, IVORY = "#0A3D62", "#C49E56", "#F5F3EE"
CONTACT_EMAIL = "lreview@law.stetson.edu"
DISCLAIMER = "Unofficial academic project by Sarthak Pattnaik using public Stetson Law Review archives."

# Folder structure
ROOT = Path(".").resolve()
VOLUMES_DIR = ROOT / "volumes"
INDEX_DIR = ROOT / "stetson_index"
INDEX_DIR.mkdir(exist_ok=True)
DOWNLOAD_LOG = ROOT / "downloads.csv"

# Google Drive zip file (replace ID if needed)
DRIVE_FILE_ID = "1kR1Shuy9JYIWUSPZPYKr1iiyOgjRjOcx"
DRIVE_DL_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"

# ========== Theming ==========
st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="wide")
st.markdown(f"""
<style>
html,body,[class*="css"] {{background:{IVORY};color:{PRIMARY};font-family:Georgia,serif;}}
.card {{background:#fff;border:1px solid {GOLD};border-radius:16px;padding:18px 22px;
box-shadow:0 3px 10px rgba(0,0,0,.08);margin-bottom:12px;}}
.score-badge {{float:right;background:{GOLD};color:{PRIMARY};
padding:3px 8px;border-radius:10px;font-size:12px;font-weight:700;}}
.meta{{color:#666;font-size:13px;margin-bottom:10px;}}
section[data-testid="stSidebar"]{{background-color:{PRIMARY};color:white;}}
section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{{color:{GOLD};}}
div[data-testid="stTextInput"] input{{border-radius:28px!important;
border:1.5px solid {GOLD}!important;text-align:center!important;
padding:12px!important;font-size:16px!important;color:{PRIMARY}!important;}}
</style>""", unsafe_allow_html=True)

# ========== Utility Functions ==========
def gdrive_download_to_bytes(file_id: str) -> bytes:
    """Download file from Google Drive (handles confirmation tokens)."""
    session = requests.Session()
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = session.get(url, stream=True)
    token = None
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break
    if token:
        resp = session.get(url + f"&confirm={token}", stream=True)
    data = io.BytesIO()
    for chunk in resp.iter_content(32768):
        if chunk:
            data.write(chunk)
    data.seek(0)
    return data.read()

def ensure_volumes_present():
    """Ensure ./volumes exists with PDFs; auto-download or let user upload."""
    VOLUMES_DIR.mkdir(exist_ok=True)
    have_pdfs = any(VOLUMES_DIR.rglob("*.pdf"))
    if have_pdfs:
        return
    st.info("📦 No PDFs found. Attempting to download volumes.zip from Google Drive…")
    try:
        data = gdrive_download_to_bytes(DRIVE_FILE_ID)
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            zf.extractall(VOLUMES_DIR)
        st.success("✅ Volumes downloaded and extracted.")
    except Exception:
        st.warning("⚠️ Auto-download failed. Please upload your volumes.zip manually.")
        file = st.file_uploader("Upload volumes.zip", type=["zip"])
        if file:
            try:
                with zipfile.ZipFile(file, "r") as zf:
                    zf.extractall(VOLUMES_DIR)
                st.success("✅ Uploaded archive extracted successfully.")
            except Exception as e:
                st.error(f"Extraction failed: {e}")
                st.stop()
        else:
            st.stop()

def get_all_pdfs():
    pdfs = []
    for i in range(30, 56):
        for d in VOLUMES_DIR.rglob(f"Volume {i}"):
            pdfs.extend(list(d.rglob("*.pdf")))
    return pdfs

def prettify(name): return re.sub(r"[_\-]+"," ",Path(name).stem).title()
def extract_volume(name): 
    m=re.search(r"Volume\s*\d+",str(name),re.I);return m.group(0) if m else "Unknown Volume"
def format_bytes(n): return f"{n/1024/1024:.1f} MB" if n else "—"

def log_download(path,title):
    new = not DOWNLOAD_LOG.exists()
    with open(DOWNLOAD_LOG,"a",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        if new: w.writerow(["timestamp","date","volume","title","path"])
        w.writerow([datetime.now().isoformat(timespec="seconds"),date.today(),extract_volume(path),title,path])

def read_downloads():
    if not DOWNLOAD_LOG.exists():
        return pd.DataFrame(columns=["timestamp","date","volume","title","path"])
    try: return pd.read_csv(DOWNLOAD_LOG)
    except: return pd.DataFrame(columns=["timestamp","date","volume","title","path"])

def extract_text(pdf):
    """Extract text safely from PDF."""
    try:
        reader=PdfReader(str(pdf))
        txt=" ".join(p.extract_text() or "" for p in reader.pages)
        return txt[:100000]
    except Exception:
        return ""

# ========== Index Building ==========
@st.cache_resource(show_spinner=False)
def build_index():
    ensure_volumes_present()
    pdfs = get_all_pdfs()
    if not pdfs:
        st.error("No PDFs found under Volume 30–55. Upload or check structure.")
        st.stop()

    st.info(f"📚 Building index from {len(pdfs)} PDFs…")
    model=SentenceTransformer("all-MiniLM-L6-v2")
    texts,meta=[],[]
    for pdf in pdfs:
        txt=extract_text(pdf)
        if not txt.strip(): continue
        step=900
        for i in range(0,len(txt),step):
            chunk=txt[i:i+1000]
            if len(chunk)<400: continue
            texts.append(chunk)
            meta.append({"path":str(pdf),"file":pdf.name})
    emb=model.encode(texts,show_progress_bar=True,batch_size=64)
    index=faiss.IndexFlatL2(emb.shape[1])
    index.add(emb.astype("float32"))
    st.success("✅ Semantic index built.")
    return (index,{"texts":texts,"meta":meta}),model

def summarize(text,model,n=8):
    sents=re.split(r"(?<=[.!?])\s+",text)
    sents=[s.strip() for s in sents if len(s.strip())>40]
    if not sents: return "Summary unavailable."
    cap=sents[:200]
    emb=model.encode(cap)
    sc=(emb@emb.T).sum(axis=1)
    top=[s for s,_ in sorted(zip(cap,sc),key=lambda z:z[1],reverse=True)[:n]]
    return " ".join(top)

# ========== Sidebar ==========
with st.sidebar:
    st.markdown("### 📬 Contact")
    st.markdown(f"- **Email:** {CONTACT_EMAIL}")
    st.markdown("- **Institution:** Stetson University College of Law")
    st.markdown("---")
    st.markdown(f"⚖️ *{DISCLAIMER}*")

# ========== Header ==========
st.markdown(f"<h1 style='text-align:center;'>{APP_TITLE}</h1>",unsafe_allow_html=True)
st.caption("Elegant, academic, and intuitive — built for Stetson Law students.")
st.markdown("---")

# ========== Load Index ==========
(index,store),model=build_index()
pdf_count=len({m["path"] for m in store["meta"]})
st.success(f"📚 Detected {pdf_count} PDFs across Vol.30–55.")

# ========== Tabs ==========
tab_search,tab_browse,tab_insights,tab_about=st.tabs(
    ["🔍 Search","📂 Browse by Volume","📊 Insights","📬 About"])

# --- SEARCH ---
with tab_search:
    st.subheader("🔎 Search the Archive")
    query=st.text_input("Enter topic, author, or question",
        placeholder="e.g., Privacy law, AI in courts, Death penalty")
    if query:
        with st.spinner("Searching and summarizing…"):
            q_emb=model.encode([query])
            D,I=index.search(q_emb.astype("float32"),k=30)
            results=[];seen=set()
            for idx,dist in zip(I[0],D[0]):
                if idx==-1: continue
                meta=store["meta"][idx];path=meta["path"]
                if path in seen: continue
                seen.add(path)
                text=store["texts"][idx]
                results.append((meta,text,float(dist)))
            if not results: st.info("No results found.")
            else:
                dmin,dmax=min(d[2] for d in results),max(d[2] for d in results)
                for i,(m,txt,dist) in enumerate(results[:15],1):
                    rel=100*(1-(dist-dmin)/(dmax-dmin+1e-9))
                    title=prettify(m["file"]);vol=extract_volume(m["path"])
                    size=format_bytes(os.path.getsize(m["path"]))
                    summary=summarize(txt,model)
                    st.markdown(f"""
                    <div class='card'>
                    <h4 style='margin:0'>{title}<span class='score-badge'>{rel:.1f}%</span></h4>
                    <p class='meta'>{vol} • {size}</p>
                    <p style='text-align:justify'>{summary}</p></div>""",unsafe_allow_html=True)
                    if os.path.exists(m["path"]):
                        with open(m["path"],"rb") as f:
                            if st.download_button("📥 Download PDF",f.read(),
                                file_name=Path(m["path"]).name,mime="application/pdf",
                                key=f"dl-{i}-{time.time()}"):
                                log_download(m["path"],title)
                    st.divider()
    else:
        st.caption("Tip → Try 'privacy law', 'AI in courts', or 'First Amendment'.")

# --- BROWSE ---
with tab_browse:
    st.subheader("📂 Browse by Volume")
    vols=sorted({p for p in VOLUMES_DIR.rglob("Volume *") if p.is_dir()})
    if not vols: st.info("No Volume folders detected.")
    else:
        choice=st.selectbox("Select a Volume",[v.name for v in vols])
        folder=[v for v in vols if v.name==choice][0]
        pdfs=sorted(folder.rglob("*.pdf"))
        if not pdfs: st.info("No PDFs in this volume.")
        for pdf in pdfs[:50]:
            title=prettify(pdf.name)
            st.markdown(f"<div class='card'><b>{title}</b><br>{extract_volume(pdf)} • {format_bytes(pdf.stat().st_size)}</div>",unsafe_allow_html=True)
            with open(pdf,"rb") as f:
                st.download_button("📥 Download PDF",f.read(),file_name=pdf.name,mime="application/pdf",key=f"b-{pdf}")

# --- INSIGHTS ---
with tab_insights:
    st.subheader("📊 Download Trends & Analytics")
    df=read_downloads()
    if df.empty: st.info("No downloads yet — charts appear after downloads.")
    else:
        top=df.groupby(["title","volume"]).size().reset_index(name="downloads").sort_values("downloads",ascending=False).head(10)
        fig1=px.bar(top,x="downloads",y="title",color="volume",orientation="h",title="Most Downloaded Articles")
        st.plotly_chart(fig1,use_container_width=True)
        df["date"]=pd.to_datetime(df["date"],errors="coerce")
        daily=df.groupby("date").size().reset_index(name="downloads")
        fig2=px.line(daily,x="date",y="downloads",markers=True,title="Downloads Over Time")
        st.plotly_chart(fig2,use_container_width=True)

# --- ABOUT ---
with tab_about:
    st.subheader("📬 Contact & About")
    st.markdown(f"""
    <div class='card'>
    <p><b>Email:</b> {CONTACT_EMAIL}</p>
    <p><b>Institution:</b> Stetson University College of Law</p>
    <p>This assistant helps students discover, summarize, and download Stetson Law Review articles using semantic search and local analytics.</p>
    <p><i>{DISCLAIMER}</i></p></div>""",unsafe_allow_html=True)
