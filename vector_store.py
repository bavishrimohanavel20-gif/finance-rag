import chromadb
import ollama

from pathlib import Path
from pypdf import PdfReader


DATA_FOLDER = Path("data")
CHROMA_FOLDER = "chroma_db"

EMBEDDING_MODEL = "nomic-embed-text"


# -----------------------------------------
# Connect to ChromaDB
# -----------------------------------------

client = chromadb.PersistentClient(
    path=CHROMA_FOLDER
)


# Delete old collection
try:
    client.delete_collection(
        name="infosys_financial_reports"
    )

    print("Old ChromaDB collection deleted.")

except Exception:
    print("No old collection found.")


# Create fresh collection
collection = client.create_collection(
    name="infosys_financial_reports"
)


# -----------------------------------------
# Detect quarter
# -----------------------------------------

def get_quarter(filename):

    if "Q1" in filename:
        return "Q1 FY26"

    elif "Q2" in filename:
        return "Q2 FY26"

    elif "Q3" in filename:
        return "Q3 FY26"

    elif "Q4" in filename:
        return "Q4 FY26"

    return "Unknown"


# -----------------------------------------
# Read PDFs page by page
# -----------------------------------------

all_pages = []


for pdf_file in sorted(
    DATA_FOLDER.glob("*.pdf")
):

    print(
        f"\nProcessing: {pdf_file.name}"
    )

    reader = PdfReader(pdf_file)

    quarter = get_quarter(
        pdf_file.name
    )


    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if not text:
            continue


        # Keep the COMPLETE page together
        enriched_text = f"""
SOURCE: {pdf_file.name}
QUARTER: {quarter}
PAGE: {page_number}

DOCUMENT CONTENT:

{text}
""".strip()


        all_pages.append({
            "text": enriched_text,
            "source": pdf_file.name,
            "quarter": quarter,
            "page": page_number
        })


    print(
        f"Pages processed: {len(reader.pages)}"
    )


print("\n================================")
print("TOTAL PAGE DOCUMENTS:", len(all_pages))
print("================================")


# -----------------------------------------
# Create embeddings
# -----------------------------------------

for index, page_data in enumerate(
    all_pages
):

    print(
        f"Embedding {index + 1}/{len(all_pages)}...",
        end="\r"
    )


    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=page_data["text"]
    )


    embedding = response.embeddings[0]


    collection.add(

        ids=[
            f"page_{index}"
        ],

        documents=[
            page_data["text"]
        ],

        embeddings=[
            embedding
        ],

        metadatas=[
            {
                "source": page_data["source"],
                "page": page_data["page"],
                "quarter": page_data["quarter"]
            }
        ]
    )


print("\n")

print("================================")
print("ChromaDB indexing completed!")
print("Total documents:", collection.count())
print("================================")