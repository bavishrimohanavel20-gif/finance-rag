"""
Finance RAG — Infosys Financial Intelligence Assistant
========================================================
Streamlit front-end ONLY.

This file does not contain, duplicate, or modify any RAG / retrieval /
embedding / LLM logic. It only:
  - calls the existing rag.py as a subprocess to answer questions
  - calls the existing ingest.py as a subprocess to index documents
  - reads the existing persistent ChromaDB collection to report real
    (not fabricated) chunk counts
  - parses and displays the plain-text output rag.py already prints

Backend files that are NEVER modified or reimplemented here:
    rag.py, financial_extractor.py, ingest.py, retrieve.py,
    vector_store.py, pdf_reader.py, ChromaDB collection, Ollama/Qwen.
"""

import os
import sys
import re
import io
import glob
import subprocess
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

import streamlit as st
import pandas as pd

# ----------------------------------------------------------------------
# CONFIG / PATHS
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Finance RAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_SCRIPT = os.path.join(BASE_DIR, "rag.py")
INGEST_SCRIPT = os.path.join(BASE_DIR, "ingest.py")
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
CHROMA_COLLECTION_NAME = "infosys_financial_reports"

QUARTER_FILES = [
    ("Q1 FY26", "Infosys_Q1_FY26.pdf"),
    ("Q2 FY26", "Infosys_Q2_FY26.pdf"),
    ("Q3 FY26", "Infosys_Q3_FY26.pdf"),
    ("Q4 FY26", "Infosys_Q4_FY26.pdf"),
]

EXAMPLE_QUESTIONS = [
    "What was the revenue in Q1 FY26?",
    "What was the operating margin in Q4 FY26?",
    "What was the net profit in Q3 FY26?",
    "What was the free cash flow in Q2 FY26?",
]

RAG_TIMEOUT_SECONDS = 120
INGEST_TIMEOUT_SECONDS = 600
NOT_FOUND_PHRASES = (
    "could not find",
    "not present in the provided",
    "no information",
    "not found in the provided documents",
    "cannot find",
    "unable to find",
)

# ----------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------
DEFAULTS = {
    "history": [],          # list of result dicts, most recent first
    "current": None,        # the most recent result dict, or None
    "index_log": "",
    "index_error": "",
}
# Cache key for extracted financial dashboard rows so we don't re-run PDF
# extraction on every single Streamlit rerun (e.g. every keystroke).
DASHBOARD_CACHE_KEY = "dashboard_rows_cache"
for cache_key in (DASHBOARD_CACHE_KEY,):
    if cache_key not in st.session_state:
        st.session_state[cache_key] = None
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ----------------------------------------------------------------------
# STYLING — CSS restyles native Streamlit widgets only.
# No <div>/<span> layout elements are ever rendered into the page.
# ----------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #05070f 0%, #0a0e1a 100%);
        color: #e6e9f2;
    }
    section[data-testid="stSidebar"] {
        background: #070a13;
        border-right: 1px solid rgba(120, 140, 255, 0.12);
    }
    h1, h2, h3, h4, h5 { color: #f2f4fb !important; letter-spacing: 0.2px; }
    p, span, label, .stMarkdown { color: #b9c0d4; }
    div[data-testid="stMetric"] {
        background: linear-gradient(160deg, #10152a 0%, #0c101f 100%);
        border: 1px solid rgba(120, 140, 255, 0.18);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.35);
    }
    div[data-testid="stMetric"] label { color: #8f9ad1 !important; font-weight: 500; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; }
    .stButton > button {
        background: linear-gradient(135deg, #4f5ff7 0%, #7b5cf5 100%);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.1rem;
        font-weight: 600;
        box-shadow: 0 2px 10px rgba(90, 90, 245, 0.25);
    }
    .stButton > button:hover { box-shadow: 0 4px 16px rgba(120, 100, 255, 0.4); }
    .stButton > button:disabled { opacity: 0.4; }
    .stTextArea textarea, .stTextInput input {
        background-color: #0c1120 !important;
        color: #e6e9f2 !important;
        border: 1px solid rgba(120, 140, 255, 0.25) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stExpander"] {
        background: #0c1120;
        border: 1px solid rgba(120, 140, 255, 0.15);
        border-radius: 12px;
    }
    div[data-testid="stFileUploader"] {
        background: #0c1120;
        border: 1px dashed rgba(120, 140, 255, 0.35);
        border-radius: 12px;
        padding: 8px;
    }
    hr { border-color: rgba(120, 140, 255, 0.15); }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# BACKEND HELPERS
# ----------------------------------------------------------------------
def get_utf8_env() -> dict:
    """Environment that forces UTF-8 I/O so the ₹ symbol survives on Windows."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def count_report_files() -> int:
    """Counts actual PDF files present in data/. Never fabricated."""
    if not os.path.isdir(DATA_DIR):
        return 0
    return len(glob.glob(os.path.join(DATA_DIR, "*.pdf")))


def get_indexed_chunk_count() -> Tuple[Optional[int], str]:
    """
    Reads the REAL chunk count from the existing persistent ChromaDB
    collection. Does not create a new collection, embedding model, or
    RAG logic — this only opens the existing store to read .count().
    Returns (count_or_None, error_message).
    """
    if not os.path.isdir(CHROMA_DIR):
        return None, "ChromaDB directory not found. Documents may not be indexed yet."
    try:
        import chromadb  # already a backend dependency
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection(CHROMA_COLLECTION_NAME)
        return collection.count(), ""
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not read ChromaDB collection: {exc}"


def run_ingest() -> Tuple[bool, str, str]:
    """
    Runs the existing ingest.py to (re)index documents.
    Returns (success, stdout_log, error_message).
    """
    if not os.path.exists(INGEST_SCRIPT):
        return False, "", "ingest.py was not found in the project folder."
    try:
        result = subprocess.run(
            [sys.executable, "-X", "utf8", INGEST_SCRIPT],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=get_utf8_env(),
            cwd=BASE_DIR,
            timeout=INGEST_TIMEOUT_SECONDS,
        )
        log = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
        if result.returncode != 0:
            return False, log, "Indexing process exited with an error. See technical details."
        return True, log, ""
    except subprocess.TimeoutExpired:
        return False, "", "Indexing timed out. Please try again."
    except Exception as exc:  # noqa: BLE001
        return False, "", f"Indexing failed to start: {exc}"


def run_rag_subprocess(question: str) -> Tuple[str, str, int]:
    """
    Calls the existing rag.py and returns (stdout, stderr, returncode).

    IMPORTANT: rag.py reads the question via `input("Ask your question: ")`,
    not via sys.argv. So the question must be sent over stdin, NOT passed
    as a command-line argument — passing it as argv leaves rag.py blocked
    on input() forever, which is what caused the UI to hang on the spinner.

    No RAG logic lives here — this only launches the existing script and
    feeds it the same text a person would type at the "Ask your question:"
    prompt.
    """
    result = subprocess.run(
        [sys.executable, "-X", "utf8", RAG_SCRIPT],
        input=question + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=get_utf8_env(),
        cwd=BASE_DIR,
        timeout=RAG_TIMEOUT_SECONDS,
    )
    return result.stdout or "", result.stderr or "", result.returncode


def parse_rag_output(raw_output: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Parses rag.py's plain-text output:

        ================================
        ANSWER
        ================================
        Revenue: ₹44490 crore

        ================================
        SOURCES
        ================================
        - Infosys_Q2_FY26.pdf — Page 4

    Returns (answer_text, [{"file": ..., "page": ...}, ...]).
    The answer text is passed through exactly as produced — never
    rewritten, summarized, or recalculated.
    """
    if not raw_output or not raw_output.strip():
        return "", []

    text = raw_output.strip()

    answer_match = re.search(
        r"ANSWER\s*\n=+\s*\n(.*?)(?:\n=+\s*\nSOURCES|\Z)",
        text, flags=re.DOTALL | re.IGNORECASE,
    )
    sources_match = re.search(
        r"SOURCES\s*\n=+\s*\n(.*)",
        text, flags=re.DOTALL | re.IGNORECASE,
    )

    answer_block = answer_match.group(1).strip() if answer_match else ""
    sources_block = sources_match.group(1).strip() if sources_match else ""

    # If no banners were found at all, treat the whole output as the answer
    # rather than silently dropping it.
    if not answer_block and not sources_block:
        answer_block = text

    answer_lines = [
        line for line in answer_block.splitlines()
        if line.strip() and not re.match(r"^=+$", line.strip())
        and not line.strip().lower().startswith("detected quarter")
        and not line.strip().lower().startswith("detected metric")
    ]
    answer_text = "\n".join(answer_lines).strip()

    sources: List[Dict[str, str]] = []
    for line in sources_block.splitlines():
        line = line.strip()
        if not line or re.match(r"^=+$", line):
            continue
        line = line.lstrip("-•").strip()
        m = re.match(r"(.+?\.pdf)\s*[—\-–]\s*Page\s*(\d+)", line, flags=re.IGNORECASE)
        if m:
            sources.append({"file": m.group(1).strip(), "page": m.group(2).strip()})
        elif line:
            sources.append({"file": line, "page": ""})

    return answer_text, sources


def is_not_found_answer(answer_text: str) -> bool:
    """Detects rag.py's own 'not found in documents' style refusal, purely
    for display styling (info box instead of success box). Does not alter
    the answer text or override what rag.py decided."""
    lowered = answer_text.lower()
    return any(phrase in lowered for phrase in NOT_FOUND_PHRASES)


def run_rag(question: str) -> None:
    """
    Full pipeline for one question: validates, calls rag.py, parses the
    result, and stores it in session_state — all within the SAME script
    run that the button was clicked in, so the answer is never stale.
    """
    question = (question or "").strip()

    if not question:
        result = {
            "question": "",
            "answer": "",
            "sources": [],
            "error": "Please enter a question.",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        st.session_state.current = result
        return

    chunk_count, _ = get_indexed_chunk_count()
    if not chunk_count:
        result = {
            "question": question,
            "answer": "",
            "sources": [],
            "error": "Please index the documents before asking a question.",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        st.session_state.current = result
        return

    if not os.path.exists(RAG_SCRIPT):
        result = {
            "question": question,
            "answer": "",
            "sources": [],
            "error": "rag.py was not found in the project folder.",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        st.session_state.current = result
        return

    with st.spinner("Searching financial knowledge base and generating a grounded answer..."):
        try:
            stdout, stderr, code = run_rag_subprocess(question)
        except subprocess.TimeoutExpired:
            result = {
                "question": question,
                "answer": "",
                "sources": [],
                "error": "The request timed out. Please try again.",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
            st.session_state.current = result
            st.session_state.history.insert(0, result)
            return
        except Exception as exc:  # noqa: BLE001
            result = {
                "question": question,
                "answer": "",
                "sources": [],
                "error": f"Could not run the financial assistant: {exc}",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
            st.session_state.current = result
            st.session_state.history.insert(0, result)
            return

    answer_text, sources = parse_rag_output(stdout)

    if not answer_text:
        error_msg = "No answer was returned."
        if stderr.strip():
            error_msg += " Technical details are available below."
        result = {
            "question": question,
            "answer": "",
            "sources": [],
            "error": error_msg,
            "raw_error": stderr,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
    else:
        result = {
            "question": question,
            "answer": answer_text,
            "sources": sources,
            "error": "",
            "raw_error": stderr if code != 0 else "",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }

    st.session_state.current = result
    st.session_state.history.insert(0, result)


# ----------------------------------------------------------------------
# FY26 FINANCIAL DASHBOARD — reads real values via the existing
# financial_extractor.get_financial_data(pdf_path), which is the SAME
# function rag.py already relies on. It returns (page, financial_data),
# where financial_data is a dict keyed by:
#   "Revenues", "Operating Profit", "Operating Margin %",
#   "Net Profit (after non-controlling interests)"
# Nothing here reimplements extraction — this only calls the existing
# function and reports exactly what it returns. If extraction fails,
# the real exception/reason is captured (not swallowed) so it can be
# shown in a "Technical details" expander instead of a silent
# "Not available".
# ----------------------------------------------------------------------
EXPECTED_FINANCIAL_KEYS = {
    "revenue": "Revenues",
    "operating_profit": "Operating Profit",
    "operating_margin": "Operating Margin %",
    "net_profit": "Net Profit (after non-controlling interests)",
}


@st.cache_data(show_spinner=False)
def extract_quarter_financials(pdf_filename: str) -> Dict[str, Optional[Any]]:
    """
    Returns a dict with revenue / operating_profit / operating_margin /
    net_profit / page / error for one quarter's PDF, sourced from the
    existing financial_extractor.get_financial_data(). Missing metrics
    stay None (never fabricated). "error" holds the real failure reason
    if extraction did not succeed, so the UI can show it verbatim rather
    than hiding it behind a generic "Not available".
    """
    result: Dict[str, Optional[Any]] = {
        "revenue": None, "operating_profit": None,
        "operating_margin": None, "net_profit": None,
        "page": None, "error": None,
    }

    pdf_path = os.path.join(DATA_DIR, pdf_filename)
    if not os.path.exists(pdf_path):
        result["error"] = f"File not found: {pdf_path}"
        return result

    try:
        from financial_extractor import get_financial_data
    except Exception as exc:  # noqa: BLE001
        result["error"] = (
            f"Could not import get_financial_data from financial_extractor.py: {exc}"
        )
        return result

    try:
        extraction_result = get_financial_data(pdf_path)
    except Exception as exc:  # noqa: BLE001
        result["error"] = (
            f"get_financial_data('{pdf_filename}') raised an exception: {exc}"
        )
        return result

    if extraction_result is None:
        result["error"] = f"get_financial_data('{pdf_filename}') returned None."
        return result

    # Expected shape: (page, financial_data)
    if isinstance(extraction_result, tuple) and len(extraction_result) == 2:
        page, financial_data = extraction_result
    elif isinstance(extraction_result, dict):
        # Defensive fallback in case the function ever returns just the dict.
        financial_data = extraction_result
        page = extraction_result.get("page") or extraction_result.get("Financial page")
    else:
        result["error"] = (
            f"Unexpected return type from get_financial_data('{pdf_filename}'): "
            f"{type(extraction_result).__name__} — expected a (page, financial_data) tuple."
        )
        return result

    if not isinstance(financial_data, dict):
        result["error"] = (
            f"financial_data returned for '{pdf_filename}' was not a dict "
            f"(got {type(financial_data).__name__})."
        )
        return result

    result["page"] = page
    for target_key, source_key in EXPECTED_FINANCIAL_KEYS.items():
        result[target_key] = financial_data.get(source_key)

    if all(result[k] is None for k in EXPECTED_FINANCIAL_KEYS):
        result["error"] = (
            f"get_financial_data('{pdf_filename}') returned data, but none of the "
            f"expected keys {list(EXPECTED_FINANCIAL_KEYS.values())} were present. "
            f"Keys actually returned: {list(financial_data.keys())}"
        )

    return result


def build_financial_dashboard_data() -> List[Dict[str, Any]]:
    """Builds one row per quarter using extract_quarter_financials()."""
    rows = []
    for label, filename in QUARTER_FILES:
        data = extract_quarter_financials(filename)
        rows.append({"Quarter": label, **data})
    return rows


def format_currency(value: Optional[Any]) -> str:
    if value in (None, ""):
        return "Not available"
    try:
        return f"₹{float(value):,.0f} crore"
    except (TypeError, ValueError):
        return "Not available"


def format_percent(value: Optional[Any]) -> str:
    if value in (None, ""):
        return "Not available"
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "Not available"


def highest_quarter_label(chart_df: "pd.DataFrame", column: str) -> str:
    """Returns the quarter with the highest value for a column, computed
    only from actually-extracted data. Never guesses when data is missing."""
    if column not in chart_df.columns:
        return "Not available"
    series = chart_df[column].dropna()
    if series.empty:
        return "Not available"
    return str(series.idxmax())


def build_quarterly_insights(rows: List[Dict[str, Any]]) -> List[str]:
    """
    Computes plain-language insights strictly from extracted numeric data
    (percentage change Q1->Q4, operating-margin range). Never invents an
    explanation that isn't backed by the extracted numbers; if the needed
    values aren't available, that specific insight is simply skipped.
    """
    def get(quarter_label: str, key: str) -> Optional[float]:
        for r in rows:
            if r["Quarter"] == quarter_label:
                val = r.get(key)
                try:
                    return float(val) if val not in (None, "") else None
                except (TypeError, ValueError):
                    return None
        return None

    insights: List[str] = []

    q1_rev, q4_rev = get("Q1 FY26", "revenue"), get("Q4 FY26", "revenue")
    if q1_rev is not None and q4_rev is not None and q1_rev != 0:
        change = ((q4_rev - q1_rev) / q1_rev) * 100
        direction = "increased" if change >= 0 else "decreased"
        insights.append(f"Revenue {direction} by {abs(change):.1f}% from Q1 FY26 to Q4 FY26.")

    q1_np, q4_np = get("Q1 FY26", "net_profit"), get("Q4 FY26", "net_profit")
    if q1_np is not None and q4_np is not None and q1_np != 0:
        change = ((q4_np - q1_np) / q1_np) * 100
        direction = "increased" if change >= 0 else "decreased"
        insights.append(f"Net profit {direction} by {abs(change):.1f}% from Q1 FY26 to Q4 FY26.")

    margins = [get(label, "operating_margin") for label, _ in QUARTER_FILES]
    margins = [m for m in margins if m is not None]
    if len(margins) >= 2:
        insights.append(
            f"Operating margin ranged between {min(margins):.1f}% and "
            f"{max(margins):.1f}% across the reported quarters."
        )

    if not insights:
        insights.append("Insufficient extracted data to generate quarterly insights.")
    return insights


def generate_executive_summary_pdf(
    dashboard_rows: List[Dict[str, Any]],
    current_result: Optional[Dict[str, Any]],
) -> bytes:
    """
    Builds the Executive Summary PDF in memory with ReportLab and returns
    the raw bytes (no file is written to disk). The RAG answer/sources are
    reproduced exactly as returned by rag.py — nothing is recalculated,
    reworded, or replaced here.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    )

    current_result = current_result or {}
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=2 * cm, bottomMargin=2 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Finance RAG — Infosys FY26 Executive Summary", styles["Title"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("1. Project Overview", styles["Heading2"]))
    story.append(Paragraph(
        "This report summarizes the Finance RAG system's analysis of Infosys FY26 "
        "quarterly financial reports. It combines automated document extraction "
        "with a retrieval-augmented generation (RAG) assistant that answers "
        "financial questions with grounded, source-cited responses.",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. Financial Performance Summary", styles["Heading2"]))
    table_data = [["Quarter", "Revenue", "Operating Profit", "Operating Margin", "Net Profit"]]
    for row in dashboard_rows:
        table_data.append([
            row["Quarter"],
            format_currency(row.get("revenue")),
            format_currency(row.get("operating_profit")),
            format_percent(row.get("operating_margin")),
            format_currency(row.get("net_profit")),
        ])
    table = Table(table_data, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2f77")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f3fb")]),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("3. Key Quarterly Insights", styles["Heading2"]))
    for insight in build_quarterly_insights(dashboard_rows):
        story.append(Paragraph(f"• {insight}", styles["BodyText"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("4. Current RAG Query", styles["Heading2"]))
    story.append(Paragraph(current_result.get("question") or "No question has been asked yet.", styles["BodyText"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("5. Verified Answer", styles["Heading2"]))
    story.append(Paragraph(current_result.get("answer") or "No answer available.", styles["BodyText"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("6. Source Verification", styles["Heading2"]))
    sources = current_result.get("sources") or []
    if sources:
        for src in sources:
            page_txt = f" — Page {src['page']}" if src.get("page") else ""
            story.append(Paragraph(f"{src['file']}{page_txt}", styles["BodyText"]))
    else:
        story.append(Paragraph("No sources available for the current query.", styles["BodyText"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("7. Generated", styles["Heading2"]))
    story.append(Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), styles["BodyText"]))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ----------------------------------------------------------------------
# LIVE KNOWLEDGE-BASE STATS (real values, read fresh on every run)
# ----------------------------------------------------------------------
report_count = count_report_files()
chunk_count, chunk_err = get_indexed_chunk_count()
is_indexed = bool(chunk_count)
status_label = "Ready" if is_indexed else "Not Indexed"

# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 FINANCE RAG")
    st.caption("Financial Intelligence Workspace")
    st.divider()

    st.markdown("#### 📁 Quarterly Reports")
    for label, filename in QUARTER_FILES:
        exists = os.path.exists(os.path.join(DATA_DIR, filename))
        with st.container(border=True):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"**{label}**")
                st.caption(filename)
            with col_b:
                st.markdown("✅" if exists else "⚠️")

    st.divider()

    st.markdown("#### ⬆️ Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a quarterly report", type=["pdf"], label_visibility="collapsed"
    )
    st.caption("PDF only · Maximum 200 MB per file")
    if uploaded_file is not None:
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            save_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Saved: {uploaded_file.name}. Click 'Index Documents' to include it.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Upload failed: {exc}")

    st.divider()

    st.markdown("#### 🧠 Knowledge Base")
    kb_col1, kb_col2 = st.columns(2)
    with kb_col1:
        st.metric("Reports", str(report_count))
    with kb_col2:
        st.metric("Indexed Chunks", str(chunk_count) if chunk_count else "0")

    if st.button("🔄 Index Documents", use_container_width=True):
        with st.spinner("Indexing financial reports..."):
            success, log, err = run_ingest()
        st.session_state.index_log = log
        st.session_state.index_error = err
        if success:
            new_count, _ = get_indexed_chunk_count()
            new_reports = count_report_files()
            st.success(
                f"Indexing completed successfully.\n\n"
                f"Reports processed: {new_reports}\n\n"
                f"Chunks indexed: {new_count if new_count else 0}"
            )
        else:
            st.error(err or "Indexing did not complete successfully.")

    if st.session_state.index_log or st.session_state.index_error:
        with st.expander("Technical details (indexing)"):
            if st.session_state.index_error:
                st.code(st.session_state.index_error)
            if st.session_state.index_log:
                st.text(st.session_state.index_log)

    if not is_indexed:
        st.warning("Documents are not indexed yet. Click **Index Documents** above.")

    st.divider()

    st.markdown("#### ⚙️ AI Stack")
    st.caption("Ollama")
    st.caption("Qwen 2.5 3B")
    st.caption("Nomic Embed Text")
    st.caption("ChromaDB")

# ----------------------------------------------------------------------
# MAIN — HEADER
# ----------------------------------------------------------------------
st.markdown("## 📊 Finance RAG")
st.caption("Financial Intelligence Workspace")
st.write(
    "Ask questions about Infosys FY26 quarterly financial reports and "
    "receive grounded answers with exact document and page references."
)

st.write("")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Reports", str(report_count))
with m2:
    st.metric("Indexed Chunks", str(chunk_count) if chunk_count else "0")
with m3:
    st.metric("Model", "Qwen 2.5 3B")
with m4:
    st.metric("Status", status_label)

st.divider()

# ----------------------------------------------------------------------
# QUESTION AREA
# ----------------------------------------------------------------------
st.markdown("### 💬 Financial Assistant")
st.caption("Ask questions about the indexed Infosys FY26 reports.")

if not is_indexed:
    st.info("Please index the documents before asking a question.")

question_text = st.text_area(
    "Your question",
    placeholder="What was the revenue in Q2 FY26?",
    height=100,
    label_visibility="collapsed",
    key="question_box",
)

btn_col1, btn_col2, _ = st.columns([1.4, 1, 3])
with btn_col1:
    ask_clicked = st.button(
        "🔎 Ask Financial Assistant",
        use_container_width=True,
        disabled=not is_indexed,
    )
with btn_col2:
    clear_clicked = st.button("🗑️ Clear History", use_container_width=True)

if ask_clicked:
    run_rag(st.session_state.question_box)

if clear_clicked:
    st.session_state.history = []
    st.session_state.current = None
    st.success("History cleared.")

st.write("")
st.markdown("##### Example Questions")
eq_cols = st.columns(4)
for i, eq in enumerate(EXAMPLE_QUESTIONS):
    with eq_cols[i]:
        if st.button(eq, key=f"example_{i}", use_container_width=True, disabled=not is_indexed):
            run_rag(eq)

st.divider()

# ----------------------------------------------------------------------
# ANSWER AREA
# ----------------------------------------------------------------------
current = st.session_state.current
if current:
    if current["question"]:
        st.markdown("### QUESTION")
        st.info(current["question"])

    if current.get("answer"):
        if is_not_found_answer(current["answer"]):
            st.markdown("### ℹ️ ANSWER")
            with st.container(border=True):
                st.markdown(f"## {current['answer']}")
        else:
            st.markdown("### ✅ VERIFIED AI ANSWER")
            with st.container(border=True):
                st.markdown(f"## {current['answer']}")

            if current.get("sources"):
                st.markdown("### 📎 SOURCE & VERIFICATION")
                srcs = current["sources"]
                src_cols = st.columns(len(srcs))
                for i, src in enumerate(srcs):
                    with src_cols[i]:
                        with st.container(border=True):
                            st.markdown(f"**Source {i + 1}**")
                            st.markdown(f"{src['file']}")
                            if src.get("page"):
                                st.caption(f"Page {src['page']}")
            else:
                st.caption("No source references were returned for this answer.")
    elif current.get("error"):
        st.warning(current["error"])
        if current.get("raw_error"):
            with st.expander("Technical details"):
                st.code(current["raw_error"])

st.divider()

# ----------------------------------------------------------------------
# FY26 FINANCIAL DASHBOARD
# ----------------------------------------------------------------------
st.markdown("### 📈 FY26 Financial Dashboard")

dashboard_rows = build_financial_dashboard_data()

metric_keys = ("revenue", "operating_profit", "operating_margin", "net_profit")
any_value_extracted = any(row.get(k) is not None for row in dashboard_rows for k in metric_keys)
per_quarter_errors = [(row["Quarter"], row["error"]) for row in dashboard_rows if row.get("error")]

if any_value_extracted:
    st.caption("Extracted directly from the indexed Infosys FY26 quarterly reports via financial_extractor.get_financial_data().")
else:
    st.error("Financial dashboard extraction failed. See Technical details.")

if per_quarter_errors:
    with st.expander("Technical details (financial extraction)", expanded=not any_value_extracted):
        for quarter_label, err in per_quarter_errors:
            st.markdown(f"**{quarter_label}**")
            st.code(err)

display_table = pd.DataFrame([
    {
        "Quarter": row["Quarter"],
        "Revenue": format_currency(row.get("revenue")),
        "Operating Profit": format_currency(row.get("operating_profit")),
        "Operating Margin": format_percent(row.get("operating_margin")),
        "Net Profit": format_currency(row.get("net_profit")),
    }
    for row in dashboard_rows
])
st.dataframe(display_table, use_container_width=True, hide_index=True)

chart_df = pd.DataFrame(dashboard_rows).set_index("Quarter")
for col in metric_keys:
    if col in chart_df.columns:
        chart_df[col] = pd.to_numeric(chart_df[col], errors="coerce")
    else:
        chart_df[col] = pd.NA

chart_col1, chart_col2, chart_col3 = st.columns(3)
with chart_col1:
    st.markdown("**Revenue by Quarter**")
    if chart_df["revenue"].notna().any():
        st.bar_chart(chart_df["revenue"])
    else:
        st.caption("Not available")
with chart_col2:
    st.markdown("**Net Profit by Quarter**")
    if chart_df["net_profit"].notna().any():
        st.bar_chart(chart_df["net_profit"])
    else:
        st.caption("Not available")
with chart_col3:
    st.markdown("**Operating Margin by Quarter**")
    if chart_df["operating_margin"].notna().any():
        st.line_chart(chart_df["operating_margin"])
    else:
        st.caption("Not available")

st.markdown("##### Quarterly Comparison")
comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
with comp_col1:
    st.metric("Highest Revenue Quarter", highest_quarter_label(chart_df, "revenue"))
with comp_col2:
    st.metric("Highest Net Profit Quarter", highest_quarter_label(chart_df, "net_profit"))
with comp_col3:
    st.metric("Highest Operating Profit Quarter", highest_quarter_label(chart_df, "operating_profit"))
with comp_col4:
    st.metric("Highest Operating Margin Quarter", highest_quarter_label(chart_df, "operating_margin"))

st.divider()

# ----------------------------------------------------------------------
# EXECUTIVE SUMMARY
# ----------------------------------------------------------------------
st.markdown("### 📄 Executive Summary")
st.caption("Generates a PDF combining the financial dashboard with the most recent verified RAG answer.")

try:
    import reportlab  # noqa: F401
    _reportlab_available = True
except ImportError:
    _reportlab_available = False

if not _reportlab_available:
    st.warning("The 'reportlab' package is required for PDF generation. Install it with: `pip install reportlab`")
else:
    try:
        summary_pdf_bytes = generate_executive_summary_pdf(dashboard_rows, st.session_state.current)
        st.download_button(
            "📥 Download Executive Summary",
            data=summary_pdf_bytes,
            file_name=f"Finance_RAG_Executive_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )
    except Exception as exc:  # noqa: BLE001
        st.error("Could not generate the executive summary PDF.")
        with st.expander("Technical details"):
            st.code(str(exc))

st.divider()

# ----------------------------------------------------------------------
# HISTORY
# ----------------------------------------------------------------------
st.markdown("### 🕘 Previous Questions")

past_entries = [h for h in st.session_state.history if h is not current]
if not past_entries:
    st.caption("Your previous questions will appear here.")
else:
    for entry in past_entries:
        title = entry["question"] or "(empty question)"
        with st.expander(f"{title}  ·  {entry['timestamp']}"):
            st.markdown("**Q:**")
            st.write(entry["question"] or "—")
            st.markdown("**A:**")
            if entry.get("answer"):
                st.write(entry["answer"])
                for src in entry.get("sources", []):
                    page_txt = f" — Page {src['page']}" if src.get("page") else ""
                    st.caption(f"Source: {src['file']}{page_txt}")
            else:
                st.write(entry.get("error") or "No answer available.")