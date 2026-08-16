import re
from pathlib import Path

from pypdf import PdfReader

# Ollama is optional.
# It will work locally if Ollama is available.
try:
    import chromadb
except Exception:
    chromadb = None

try:
    import ollama
except Exception:
    ollama = None


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FOLDER = Path("data")

CHROMA_FOLDER = "chroma_db"

COLLECTION_NAME = "infosys_financial_reports"

EMBEDDING_MODEL = "nomic-embed-text"

LLM_MODEL = "qwen2.5:3b"


# ============================================================
# QUARTER FILES
# ============================================================

QUARTER_FILES = {
    "Q1 FY26": "Infosys_Q1_FY26.pdf",
    "Q2 FY26": "Infosys_Q2_FY26.pdf",
    "Q3 FY26": "Infosys_Q3_FY26.pdf",
    "Q4 FY26": "Infosys_Q4_FY26.pdf",
}


# ============================================================
# QUARTER DETECTION
# ============================================================

def detect_quarter(question):

    q = question.upper()

    # Q1 FY26
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

    # Latest quarter
    if (
        "LATEST QUARTER" in q
        or "LATEST" in q
        or "MOST RECENT QUARTER" in q
        or "RECENT QUARTER" in q
    ):
        return "Q4 FY26"

    return None


# ============================================================
# METRIC DETECTION
# ============================================================

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

    if "diluted eps" in q:
        return "eps"

    if "basic eps" in q:
        return "eps"

    if "eps" in q:
        return "eps"

    if "revenue" in q or "revenues" in q:
        return "revenue"

    return None


# ============================================================
# COMPANY CHECK
# ============================================================

def is_infosys_question(question):

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
        "hcltech",
        "hcl technologies",
        "cognizant"
    ]

    for company in other_companies:

        if company in q:
            return False

    return True


# ============================================================
# PDF PATH
# ============================================================

def get_pdf_path(quarter):

    filename = QUARTER_FILES.get(quarter)

    if not filename:
        return None

    path = DATA_FOLDER / filename

    if path.exists():
        return path

    return None


# ============================================================
# READ PDF
# ============================================================

def read_pdf(pdf_path):

    try:

        reader = PdfReader(str(pdf_path))

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            text = page.extract_text()

            if text:

                pages.append({
                    "page": page_number,
                    "text": text
                })

        return pages

    except Exception:

        return []


# ============================================================
# NORMALIZE TEXT
# ============================================================

def clean_number(value):

    if value is None:
        return None

    value = value.replace(",", "")
    value = value.replace("₹", "")
    value = value.strip()

    try:
        return float(value)

    except Exception:
        return None


# ============================================================
# EXTRACT FINANCIAL VALUE FROM PDF
# ============================================================

def extract_metric_from_pdf(
    quarter,
    metric
):

    pdf_path = get_pdf_path(quarter)

    if not pdf_path:
        return None, None

    pages = read_pdf(pdf_path)

    if not pages:
        return None, None

    # --------------------------------------------------------
    # First try the existing financial_extractor.
    # --------------------------------------------------------

    try:

        from financial_extractor import get_financial_data

        page, financial_data = (
            get_financial_data(
                str(pdf_path)
            )
        )

        if financial_data:

            metric_map = {

                "revenue":
                    "Revenues",

                "operating profit":
                    "Operating Profit",

                "operating margin":
                    "Operating Margin %",

                "net profit":
                    "Net Profit (after non-controlling interests)"

            }

            key = metric_map.get(metric)

            if key:

                value = financial_data.get(key)

                if value is not None:

                    return value, page

    except Exception:
        pass


    # --------------------------------------------------------
    # PDF text fallback
    # --------------------------------------------------------

    patterns = {

        "revenue": [
            r"Revenues?\s*[:\-]?\s*₹?\s*([\d,]+)",
            r"Revenue\s*[:\-]?\s*₹?\s*([\d,]+)"
        ],

        "operating profit": [
            r"Operating Profit\s*[:\-]?\s*₹?\s*([\d,]+)"
        ],

        "operating margin": [
            r"Operating Margin\s*[:\-]?\s*([\d.]+)\s*%"
        ],

        "net profit": [
            r"Net Profit.*?[:\-]?\s*₹?\s*([\d,]+)"
        ]

    }

    metric_patterns = patterns.get(metric, [])

    for page_data in pages:

        text = page_data["text"]

        for pattern in metric_patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )

            if match:

                value = match.group(1)

                value = clean_number(value)

                if value is not None:

                    return value, page_data["page"]

    return None, None


# ============================================================
# FORMAT ANSWER
# ============================================================

def format_answer(
    metric,
    value
):

    if metric == "revenue":

        return (
            f"Revenue: ₹{value:,.0f} crore"
        )

    if metric == "operating profit":

        return (
            f"Operating Profit: "
            f"₹{value:,.0f} crore"
        )

    if metric == "operating margin":

        return (
            f"Operating Margin: {value}%"
        )

    if metric == "net profit":

        return (
            "Net Profit: "
            f"₹{value:,.0f} crore"
        )

    return str(value)


# ============================================================
# PRINT SOURCE
# ============================================================

def print_source(
    quarter,
    page
):

    filename = QUARTER_FILES.get(
        quarter,
        "Unknown"
    )

    print(
        f"- {filename} — Page {page}"
    )


# ============================================================
# LOCAL OLLAMA EMBEDDING
# ============================================================

def create_question_embedding(question):

    if ollama is None:

        return None

    try:

        response = ollama.embed(

            model=EMBEDDING_MODEL,

            input=question
        )

        return response.embeddings[0]

    except Exception:

        return None


# ============================================================
# LOCAL CHROMA RETRIEVAL
# ============================================================

def retrieve_documents(
    question,
    quarter=None
):

    if (
        chromadb is None
        or ollama is None
    ):

        return None

    try:

        client = chromadb.PersistentClient(
            path=CHROMA_FOLDER
        )

        collection = client.get_collection(
            name=COLLECTION_NAME
        )

        embedding = (
            create_question_embedding(
                question
            )
        )

        if embedding is None:
            return None

        if quarter:

            results = collection.query(

                query_embeddings=[
                    embedding
                ],

                n_results=5,

                where={
                    "quarter": quarter
                }
            )

        else:

            results = collection.query(

                query_embeddings=[
                    embedding
                ],

                n_results=5
            )

        return results

    except Exception:

        return None


# ============================================================
# OLLAMA ANSWER
# ============================================================

def generate_ollama_answer(
    question,
    context
):

    if ollama is None:
        return None

    try:

        prompt = f"""
You are an Infosys financial report assistant.

Answer ONLY using the supplied document context.

Do not use outside knowledge.

Do not guess.

If the answer is not present, say:

I could not find this information in the provided documents.

USER QUESTION:
{question}

DOCUMENT CONTEXT:
{context}

Give a short answer with the exact financial value and unit when available.
"""

        response = ollama.chat(

            model=LLM_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response[
            "message"
        ][
            "content"
        ].strip()

    except Exception:

        return None


# ============================================================
# MULTI-QUARTER ANSWER
# ============================================================

def multi_quarter_answer(
    question,
    metric
):

    print(
        "\n================================"
    )

    print("ANSWER")

    print(
        "================================"
    )

    found = False

    for quarter in [
        "Q1 FY26",
        "Q2 FY26",
        "Q3 FY26",
        "Q4 FY26"
    ]:

        value, page = (
            extract_metric_from_pdf(
                quarter,
                metric
            )
        )

        if value is not None:

            found = True

            print(
                f"{quarter}: "
                f"{format_answer(metric, value)}"
            )

            print(
                f"Source: "
                f"{QUARTER_FILES[quarter]} "
                f"— Page {page}"
            )

    if not found:

        print(
            "I could not find this information "
            "in the provided documents."
        )

    print(
        "\n================================"
    )

    print("SOURCES")

    print(
        "================================"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    question = input(
        "\nAsk your question: "
    ).strip()

    if not question:

        print(
            "Please enter a question."
        )

        return


    # --------------------------------------------------------
    # Company validation
    # --------------------------------------------------------

    if not is_infosys_question(question):

        print(
            "\n================================"
        )

        print("ANSWER")

        print(
            "================================"
        )

        print(
            "I could not find this information "
            "in the provided documents."
        )

        print(
            "\n================================"
        )

        print("SOURCES")

        print(
            "================================"
        )

        return


    quarter = detect_quarter(
        question
    )

    metric = detect_metric(
        question
    )


    print(
        "\nDetected quarter:",
        quarter
    )

    print(
        "Detected metric:",
        metric
    )


    # --------------------------------------------------------
    # Multi-quarter questions
    # --------------------------------------------------------

    q_lower = question.lower()

    if (
        quarter is None
        and metric
        and (
            "four quarters" in q_lower
            or "across the quarters" in q_lower
            or "across all quarters" in q_lower
            or "quarterly trend" in q_lower
            or "trend across" in q_lower
        )
    ):

        multi_quarter_answer(
            question,
            metric
        )

        return


    # --------------------------------------------------------
    # Direct financial extraction
    # --------------------------------------------------------

    if quarter and metric in [
        "revenue",
        "operating profit",
        "operating margin",
        "net profit"
    ]:

        value, page = (
            extract_metric_from_pdf(
                quarter,
                metric
            )
        )

        if value is not None:

            print(
                "\n================================"
            )

            print("ANSWER")

            print(
                "================================"
            )

            print(
                format_answer(
                    metric,
                    value
                )
            )

            print(
                "\n================================"
            )

            print("SOURCES")

            print(
                "================================"
            )

            print_source(
                quarter,
                page
            )

            return


    # --------------------------------------------------------
    # Local Ollama + ChromaDB RAG
    # --------------------------------------------------------

    results = retrieve_documents(
        question,
        quarter
    )

    if results:

        documents = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        if documents:

            context_parts = []

            for document, metadata in zip(
                documents,
                metadatas
            ):

                context_parts.append(
                    f"""
SOURCE: {metadata.get('source')}
PAGE: {metadata.get('page')}
QUARTER: {metadata.get('quarter')}

DOCUMENT:
{document}
"""
                )

            context = "\n".join(
                context_parts
            )

            answer = (
                generate_ollama_answer(
                    question,
                    context
                )
            )

            if answer:

                print(
                    "\n================================"
                )

                print("ANSWER")

                print(
                    "================================"
                )

                print(answer)

                print(
                    "\n================================"
                )

                print("SOURCES")

                print(
                    "================================"
                )

                seen = set()

                for metadata in metadatas:

                    source = metadata.get(
                        "source"
                    )

                    page = metadata.get(
                        "page"
                    )

                    key = (
                        source,
                        page
                    )

                    if key not in seen:

                        print(
                            f"- {source} "
                            f"— Page {page}"
                        )

                        seen.add(key)

                return


    # --------------------------------------------------------
    # Final safe response
    # --------------------------------------------------------

    print(
        "\n================================"
    )

    print("ANSWER")

    print(
        "================================"
    )

    print(
        "I could not find this information "
        "in the provided documents."
    )

    print(
        "\n================================"
    )

    print("SOURCES")

    print(
        "================================"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()