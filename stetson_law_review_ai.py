"""
⚖️ Stetson Law Review AI Assistant — Hybrid Local Edition
Elegant, academic, and intuitive — built for Stetson Law students.
✅ Works with normal & scanned PDFs (OCR-enabled)
✅ Uses semantic FAISS search with summaries and relevance ranking
"""

# ================== IMPORTS ==================
import os, re, io, csv, platform
from pathlib import Path
from datetime import datetime, date
import streamlit as st
import pandas as pd
import plotly.express as px

# LangChain + NLP
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document
from sentence_transformers import SentenceTransformer

# ================== CONFIG ==================
APP_TITLE = "⚖️ Stetson Law Review AI Assistant"
PRIMARY, GOLD, IVORY = "#0A3D62", "#C49E56", "#F5F3EE"
CONTACT_EMAIL = "lreview@law.stetson.edu"

# ✅ Your absolute local path setup
BASE_DIR_
