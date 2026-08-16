import re
from pathlib import Path

import chromadb
import ollama

from financial_extractor import get_financial_data


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_FOLDER = "chroma_db"

COLLECTION_NAME = "infosys_financial_reports"

EMBEDDING_MODEL = "nomic-embed-text"

LLM_MODEL = "qwen2.5:3b"


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path=CHROMA_FOLDER
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# ============================================================
# DETECT QUARTER
# ============================================================

def detect_quarter(question):

    question_upper = question.upper()

    # Q1 FY26 / Q1FY26
    match = re.search(
        r"\bQ([1-4])\s*FY\s*26\b",
        question_upper
    )

    if match:

        return (
            "Q"
            + match.group(1)
            + " FY26"
        )

    # Q1 / Q2 / Q3 / Q4
    match = re.search(
        r"\bQ([1-4])\b",
        question_upper
    )

    if match:

        return (
            "Q"
            + match.group(1)
            + " FY26"
        )

    return None


# ============================================================
# DETECT METRIC
# ============================================================

def detect_metric(question):

    q = question.lower()

    # Check more specific phrases first

    if "operating margin" in q:

        return "operating margin"

    if "operating profit" in q:

        return "operating profit"

    if "net profit" in q:

        return "net profit"

    if "revenue" in q or "revenues" in q:

        return "revenue"

    if "free cash flow" in q:

        return "free cash flow"

    if "basic eps" in q:

        return "eps"

    if "diluted eps" in q:

        return "eps"

    if "eps" in q:

        return "eps"

    return None


# ============================================================
# DETECT WHETHER QUESTION IS ABOUT INFOSYS
# ============================================================

def is_infosys_question(question):

    q = question.lower()

    infosys_words = [
        "infosys",
        "infy"
    ]

    # If Infosys is explicitly mentioned
    for word in infosys_words:

        if word in q:

            return True

    # If no other company is mentioned,
    # assume the question refers to the
    # supplied Infosys financial reports.

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
# GET PDF PATH
# ============================================================

def get_pdf_path(quarter):

    if not quarter:

        return None

    filename = (
        "Infosys_"
        + quarter.replace(" ", "_")
        + ".pdf"
    )

    path = (
        Path("data")
        / filename
    )

    if path.exists():

        return path

    return None


# ============================================================
# GET DIRECT FINANCIAL ANSWER
# ============================================================

def get_direct_financial_answer(
    quarter,
    metric
):

    if not quarter or not metric:

        return None, None


    # We currently have exact extraction
    # for these metrics.

    supported_metrics = [
        "revenue",
        "operating profit",
        "operating margin",
        "net profit"
    ]

    if metric not in supported_metrics:

        return None, None


    pdf_path = get_pdf_path(
        quarter
    )

    if not pdf_path:

        return None, None


    page, financial_data = (
        get_financial_data(
            pdf_path
        )
    )


    if not financial_data:

        return None, None


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


    label = metric_map.get(
        metric
    )


    if not label:

        return None, None


    value = financial_data.get(
        label
    )


    if value is None:

        return None, None


    return value, page


# ============================================================
# FORMAT DIRECT ANSWER
# ============================================================

def format_direct_answer(
    metric,
    value
):

    if metric == "revenue":

        return (
            f"Revenue: ₹{value} crore"
        )


    if metric == "operating profit":

        return (
            f"Operating Profit: "
            f"₹{value} crore"
        )


    if metric == "operating margin":

        return (
            f"Operating Margin: {value}%"
        )


    if metric == "net profit":

        return (
            "Net Profit "
            "(after non-controlling interests): "
            f"₹{value} crore"
        )


    return str(value)


# ============================================================
# PRINT DIRECT SOURCES
# ============================================================

def print_direct_source(
    quarter,
    page
):

    filename = (
        "Infosys_"
        + quarter.replace(" ", "_")
        + ".pdf"
    )

    print(
        f"- {filename} — Page {page}"
    )


# ============================================================
# CREATE QUESTION EMBEDDING
# ============================================================

def create_question_embedding(
    question
):

    response = ollama.embed(

        model=EMBEDDING_MODEL,

        input=question
    )

    return response.embeddings[0]


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    question,
    quarter=None
):

    question_embedding = (
        create_question_embedding(
            question
        )
    )


    # --------------------------------------------------------
    # Quarter-specific retrieval
    # --------------------------------------------------------

    if quarter:

        results = collection.query(

            query_embeddings=[
                question_embedding
            ],

            n_results=5,

            where={
                "quarter": quarter
            }
        )

    else:

        results = collection.query(

            query_embeddings=[
                question_embedding
            ],

            n_results=5
        )


    return results


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    results
):

    documents = (
        results.get(
            "documents",
            [[]]
        )[0]
    )

    metadatas = (
        results.get(
            "metadatas",
            [[]]
        )[0]
    )


    context_parts = []


    for document, metadata in zip(
        documents,
        metadatas
    ):

        context_parts.append(

            f"""
SOURCE: {metadata.get("source")}
PAGE: {metadata.get("page")}
QUARTER: {metadata.get("quarter")}

DOCUMENT:
{document}
"""
        )


    return (
        "\n".join(
            context_parts
        ),
        metadatas
    )


# ============================================================
# LLM ANSWER
# ============================================================

def generate_rag_answer(
    question,
    quarter,
    metric,
    context
):

    prompt = f"""
You are a financial-report question answering assistant.

You must answer ONLY from the supplied Infosys
financial report context.

USER QUESTION:
{question}

DETECTED QUARTER:
{quarter}

DETECTED METRIC:
{metric}

DOCUMENT CONTEXT:
{context}


IMPORTANT RULES:

1. Answer only from the supplied documents.

2. Do not use outside knowledge.

3. Do not guess.

4. Do not invent information.

5. If the requested information is not present,
   say exactly:

   I could not find this information in the provided documents.

6. If a financial metric is requested, identify
   the exact metric before answering.

7. Never confuse:

   Revenue
   Revenue Growth
   Client contribution to revenue
   Operating Profit
   Operating Margin
   Net Profit
   EPS
   Free Cash Flow
   Number of Clients

8. If the question asks about a company that is not
   present in the documents, say:

   I could not find this information in the provided documents.

9. Do not answer an Apple question using Infosys data.

10. Preserve the units given in the document.

11. For Indian financial statements, use ₹ crore
    when the source uses ₹ crore.

12. Give a short answer.

13. Include the metric name and value when possible.

14. Never use placeholders such as XX%, XXX,
    or unknown numerical values.

15. Do not calculate a value unless the question
    explicitly requires calculation.

16. If the context does not contain enough information,
    use the exact "could not find" response.

FINAL ANSWER:
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


# ============================================================
# PRINT RAG SOURCES
# ============================================================

def print_rag_sources(
    metadatas
):

    print(
        "\n================================"
    )

    print(
        "SOURCES"
    )

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


        if key in seen:

            continue


        print(
            f"- {source} — Page {page}"
        )


        seen.add(
            key
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
            "\nPlease enter a question."
        )

        return


    # --------------------------------------------------------
    # Detect quarter and metric
    # --------------------------------------------------------

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
    # Check company
    # --------------------------------------------------------

    if not is_infosys_question(
        question
    ):

        print(
            "\n================================"
        )

        print(
            "ANSWER"
        )

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

        print(
            "SOURCES"
        )

        print(
            "================================"
        )

        return


    # --------------------------------------------------------
    # EXACT FINANCIAL ANSWER
    # --------------------------------------------------------

    direct_value, direct_page = (
        get_direct_financial_answer(
            quarter,
            metric
        )
    )


    if direct_value is not None:

        answer = format_direct_answer(
            metric,
            direct_value
        )


        print(
            "\n================================"
        )

        print(
            "ANSWER"
        )

        print(
            "================================"
        )

        print(answer)


        print(
            "\n================================"
        )

        print(
            "SOURCES"
        )

        print(
            "================================"
        )


        print_direct_source(
            quarter,
            direct_page
        )


        return


    # --------------------------------------------------------
    # RAG FALLBACK
    # --------------------------------------------------------

    results = retrieve_documents(
        question,
        quarter
    )


    documents = results.get(
        "documents",
        [[]]
    )[0]


    if not documents:

        print(
            "\n================================"
        )

        print(
            "ANSWER"
        )

        print(
            "================================"
        )

        print(
            "I could not find this information "
            "in the provided documents."
        )

        print_rag_sources([])

        return


    context, metadatas = (
        build_context(
            results
        )
    )


    answer = generate_rag_answer(
        question,
        quarter,
        metric,
        context
    )


    print(
        "\n================================"
    )

    print(
        "ANSWER"
    )

    print(
        "================================"
    )

    print(answer)


    print_rag_sources(
        metadatas
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()