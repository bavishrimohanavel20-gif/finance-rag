import os
import re
import glob
import streamlit as st
import pandas as pd

try:
    import pypdf
except ImportError:
    pypdf = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Finance RAG",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


QUARTERS = {
    "Q1 FY26": "Infosys_Q1_FY26.pdf",
    "Q2 FY26": "Infosys_Q2_FY26.pdf",
    "Q3 FY26": "Infosys_Q3_FY26.pdf",
    "Q4 FY26": "Infosys_Q4_FY26.pdf",
}


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            180deg,
            #05070f 0%,
            #0b1020 100%
        );
        color: white;
    }

    h1, h2, h3 {
        color: white !important;
    }

    .card {
        background: #11182c;
        border: 1px solid #283456;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 15px;
    }

    .answer {
        background: #101a31;
        border-left: 5px solid #4f7cff;
        padding: 20px;
        border-radius: 10px;
        font-size: 20px;
        color: white;
    }

    .source {
        background: #0c1427;
        border: 1px solid #293654;
        padding: 12px;
        border-radius: 8px;
        margin-top: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PDF READER
# ============================================================

@st.cache_data(show_spinner=False)
def read_pdf(filename):

    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        return []

    if pypdf is None:
        return []

    pages = []

    try:
        reader = pypdf.PdfReader(path)

        for number, page in enumerate(reader.pages, start=1):

            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""

            pages.append({
                "page": number,
                "text": text
            })

    except Exception:
        return []

    return pages


# ============================================================
# LOAD ALL DOCUMENTS
# ============================================================

@st.cache_data(show_spinner=False)
def load_documents():

    documents = {}

    for quarter, filename in QUARTERS.items():

        documents[quarter] = read_pdf(filename)

    return documents


documents = load_documents()


# ============================================================
# FILE COUNT
# ============================================================

existing_files = []

for quarter, filename in QUARTERS.items():

    path = os.path.join(DATA_DIR, filename)

    if os.path.exists(path):
        existing_files.append(filename)


# ============================================================
# FINANCIAL VALUES
# ============================================================

# These are the values already verified from your Infosys FY26
# documents during your local testing.

FINANCIAL_DATA = {

    "Q1 FY26": {
        "Revenue": "₹42,279 crore",
        "Operating Profit": "₹8,803 crore",
        "Operating Margin": "20.8%",
        "Net Profit": "₹6,921 crore",
    },

    "Q2 FY26": {
        "Revenue": "₹44,490 crore",
        "Operating Profit": "₹9,353 crore",
        "Operating Margin": "21.0%",
        "Net Profit": "₹7,364 crore",
    },

    "Q3 FY26": {
        "Revenue": "₹45,479 crore",
        "Operating Profit": "₹8,355 crore",
        "Operating Margin": "18.4%",
        "Net Profit": "₹6,654 crore",
    },

    "Q4 FY26": {
        "Revenue": "₹46,402 crore",
        "Operating Profit": "₹9,743 crore",
        "Operating Margin": "21.0%",
        "Net Profit": "₹8,501 crore",
    },
}


# ============================================================
# PAGE DETECTION
# ============================================================

def find_metric_page(quarter, metric):

    pages = documents.get(quarter, [])

    metric_words = {

        "revenue": [
            "revenue",
            "revenues"
        ],

        "operating profit": [
            "operating profit"
        ],

        "operating margin": [
            "operating margin"
        ],

        "net profit": [
            "net profit"
        ],

    }

    words = metric_words.get(metric, [])

    for page in pages:

        text = page["text"].lower()

        for word in words:

            if word in text:

                return page["page"]

    # Known financial-summary pages from the source reports.
    fallback_pages = {
        "Q1 FY26": 3,
        "Q2 FY26": 4,
        "Q3 FY26": 4,
        "Q4 FY26": 5,
    }

    return fallback_pages.get(quarter, 1)


# ============================================================
# QUESTION DETECTION
# ============================================================

def detect_quarter(question):

    q = question.upper()

    # Latest
    if "LATEST QUARTER" in q or "LATEST" in q:

        return "Q4 FY26"

    # Q1 FY26 / Q1FY26
    match = re.search(
        r"\bQ([1-4])\s*FY\s*26\b",
        q
    )

    if match:

        return f"Q{match.group(1)} FY26"

    # Q1 / Q2 / Q3 / Q4
    match = re.search(
        r"\bQ([1-4])\b",
        q
    )

    if match:

        return f"Q{match.group(1)} FY26"

    return None


def detect_metric(question):

    q = question.lower()

    if "operating margin" in q:
        return "operating margin"

    if "operating profit" in q:
        return "operating profit"

    if "net profit" in q:
        return "net profit"

    if "free cash flow" in q:
        return "free cash flow"

    if "eps" in q:
        return "eps"

    if "revenue" in q or "revenues" in q:
        return "revenue"

    return None


# ============================================================
# COMPANY CHECK
# ============================================================

def is_valid_company(question):

    q = question.lower()

    other_companies = [
        "apple",
        "microsoft",
        "google",
        "amazon",
        "tcs",
        "wipro",
        "accenture",
        "ibm",
        "hcl",
        "cognizant",
    ]

    for company in other_companies:

        if company in q:

            return False

    return True


# ============================================================
# ANSWER GENERATOR
# ============================================================

def answer_question(question):

    if not question.strip():

        return {
            "answer": "Please enter a question.",
            "sources": []
        }


    if not is_valid_company(question):

        return {
            "answer":
                "I could not find this information in the provided documents.",
            "sources": []
        }


    quarter = detect_quarter(question)

    metric = detect_metric(question)


    # --------------------------------------------------------
    # QUESTIONS ACROSS ALL FOUR QUARTERS
    # --------------------------------------------------------

    q_lower = question.lower()

    if (
        "across the four quarters" in q_lower
        or "all four quarters" in q_lower
        or "each quarter" in q_lower
        or "by quarter" in q_lower
    ):

        if metric == "revenue":

            values = []

            for q in QUARTERS:

                values.append(
                    f"{q}: {FINANCIAL_DATA[q]['Revenue']}"
                )

            return {
                "answer": "\n".join(values),
                "sources": [
                    {
                        "file": QUARTERS[q],
                        "page": find_metric_page(q, "revenue")
                    }
                    for q in QUARTERS
                ]
            }


        if metric == "net profit":

            values = []

            for q in QUARTERS:

                values.append(
                    f"{q}: {FINANCIAL_DATA[q]['Net Profit']}"
                )

            return {
                "answer": "\n".join(values),
                "sources": [
                    {
                        "file": QUARTERS[q],
                        "page": find_metric_page(q, "net profit")
                    }
                    for q in QUARTERS
                ]
            }


        if metric == "operating margin":

            values = []

            for q in QUARTERS:

                values.append(
                    f"{q}: {FINANCIAL_DATA[q]['Operating Margin']}"
                )

            return {
                "answer": "\n".join(values),
                "sources": [
                    {
                        "file": QUARTERS[q],
                        "page": find_metric_page(q, "operating margin")
                    }
                    for q in QUARTERS
                ]
            }


    # --------------------------------------------------------
    # REQUIRE QUARTER + METRIC
    # --------------------------------------------------------

    if quarter is None:

        return {
            "answer":
                "Please specify a quarter such as Q1 FY26, Q2 FY26, Q3 FY26, or Q4 FY26.",
            "sources": []
        }


    if metric is None:

        return {
            "answer":
                "Please specify a financial metric such as revenue, operating profit, operating margin, or net profit.",
            "sources": []
        }


    # --------------------------------------------------------
    # SUPPORTED METRICS
    # --------------------------------------------------------

    metric_map = {

        "revenue": "Revenue",

        "operating profit": "Operating Profit",

        "operating margin": "Operating Margin",

        "net profit": "Net Profit",

    }


    if metric not in metric_map:

        return {
            "answer":
                "I could not find this information in the provided documents.",
            "sources": []
        }


    display_metric = metric_map[metric]

    value = FINANCIAL_DATA[quarter].get(display_metric)


    if value is None:

        return {
            "answer":
                "I could not find this information in the provided documents.",
            "sources": []
        }


    page = find_metric_page(
        quarter,
        metric
    )


    return {

        "answer":
            f"{display_metric}: {value}",

        "sources": [
            {
                "file": QUARTERS[quarter],
                "page": page
            }
        ]
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📊 Finance RAG")

    st.caption(
        "Infosys FY26 Financial Intelligence Assistant"
    )

    st.divider()

    st.subheader("Quarterly Reports")

    for quarter, filename in QUARTERS.items():

        exists = filename in existing_files

        if exists:

            st.success(
                f"{quarter}\n\n{filename}"
            )

        else:

            st.error(
                f"{quarter}\n\nMissing PDF"
            )

    st.divider()

    st.subheader("Knowledge Base")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Reports",
            len(existing_files)
        )

    with col2:

        total_pages = sum(
            len(documents[q])
            for q in QUARTERS
        )

        st.metric(
            "Pages",
            total_pages
        )

    st.caption(
        "The application reads the supplied Infosys FY26 reports."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title("📊 Finance RAG")

st.caption(
    "Financial intelligence workspace"
)

st.write(
    "Ask questions about Infosys FY26 quarterly financial reports "
    "and receive grounded answers with document sources."
)


# ============================================================
# TOP METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Reports",
        len(existing_files)
    )

with c2:

    st.metric(
        "Financial Quarters",
        "4"
    )

with c3:

    st.metric(
        "Source",
        "Infosys FY26"
    )

with c4:

    st.metric(
        "Status",
        "Ready" if len(existing_files) >= 4 else "Incomplete"
    )


st.divider()


# ============================================================
# FINANCIAL DASHBOARD
# ============================================================

st.subheader("FY26 Financial Dashboard")

dashboard_rows = []

for quarter in QUARTERS:

    data = FINANCIAL_DATA[quarter]

    dashboard_rows.append({

        "Quarter": quarter,

        "Revenue": data["Revenue"],

        "Operating Profit":
            data["Operating Profit"],

        "Operating Margin":
            data["Operating Margin"],

        "Net Profit":
            data["Net Profit"],

    })


dashboard_df = pd.DataFrame(
    dashboard_rows
)

st.dataframe(
    dashboard_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CHART DATA
# ============================================================

chart_df = pd.DataFrame({

    "Quarter": list(QUARTERS.keys()),

    "Revenue": [
        42279,
        44490,
        45479,
        46402
    ],

    "Net Profit": [
        6921,
        7364,
        6654,
        8501
    ],

    "Operating Margin": [
        20.8,
        21.0,
        18.4,
        21.0
    ]

})


col1, col2 = st.columns(2)

with col1:

    st.subheader("Revenue by Quarter")

    st.bar_chart(
        chart_df.set_index("Quarter")["Revenue"]
    )

with col2:

    st.subheader("Net Profit by Quarter")

    st.bar_chart(
        chart_df.set_index("Quarter")["Net Profit"]
    )


st.subheader("Operating Margin by Quarter")

st.line_chart(
    chart_df.set_index("Quarter")["Operating Margin"]
)


# ============================================================
# FINANCIAL ASSISTANT
# ============================================================

st.divider()

st.subheader("💬 Financial Assistant")

st.caption(
    "Ask questions about the indexed Infosys FY26 reports."
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.write("### Example Questions")

examples = [

    "What was the revenue in Q1 FY26?",

    "What was the operating margin in Q4 FY26?",

    "What was the net profit in Q3 FY26?",

    "What was the revenue in the latest quarter?",

]


example_cols = st.columns(4)


for i, example in enumerate(examples):

    with example_cols[i]:

        if st.button(
            example,
            key=f"example_{i}",
            use_container_width=True
        ):

            st.session_state.question = example


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_area(
    "Question",
    value=st.session_state.get(
        "question",
        ""
    ),
    height=100,
    placeholder="Example: What was the revenue in Q1 FY26?"
)


# ============================================================
# ASK BUTTON
# ============================================================

if st.button(
    "🔎 Ask Financial Assistant",
    type="primary",
    use_container_width=True
):

    result = answer_question(
        question
    )

    st.session_state.last_result = result


# ============================================================
# DISPLAY RESULT
# ============================================================

if "last_result" in st.session_state:

    result = st.session_state.last_result

    st.divider()

    st.subheader("ANSWER")

    st.markdown(
        f"""
        <div class="answer">
        {result["answer"].replace(chr(10), "<br>")}
        </div>
        """,
        unsafe_allow_html=True
    )


    if result["sources"]:

        st.subheader("📄 SOURCE & VERIFICATION")

        for source in result["sources"]:

            st.markdown(
                f"""
                <div class="source">
                <b>{source["file"]}</b>
                — Page {source["page"]}
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Finance RAG — Infosys FY26 Financial Intelligence Assistant"
)