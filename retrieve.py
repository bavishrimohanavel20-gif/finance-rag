import re
import chromadb
import ollama


CHROMA_FOLDER = "chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"


# Connect to ChromaDB
client = chromadb.PersistentClient(
    path=CHROMA_FOLDER
)

collection = client.get_collection(
    name="infosys_financial_reports"
)


# -----------------------------------------
# Detect quarter
# -----------------------------------------

def detect_quarter(question):

    question_upper = question.upper()

    match = re.search(
        r"Q([1-4])\s*FY26",
        question_upper
    )

    if match:
        return "Q" + match.group(1) + " FY26"

    match = re.search(
        r"\bQ([1-4])\b",
        question_upper
    )

    if match:
        return "Q" + match.group(1) + " FY26"

    return None


# -----------------------------------------
# Ask question
# -----------------------------------------

question = input(
    "\nEnter your question: "
)


quarter = detect_quarter(question)

print("\nDetected quarter:", quarter)


# -----------------------------------------
# Create question embedding
# -----------------------------------------

response = ollama.embed(
    model=EMBEDDING_MODEL,
    input=question
)

question_embedding = response.embeddings[0]


# -----------------------------------------
# Search ChromaDB
# -----------------------------------------

if quarter:

    results = collection.query(
        query_embeddings=[
            question_embedding
        ],
        n_results=4,
        where={
            "quarter": quarter
        }
    )

else:

    results = collection.query(
        query_embeddings=[
            question_embedding
        ],
        n_results=4
    )


# -----------------------------------------
# Display results
# -----------------------------------------

print("\n================================")
print("RETRIEVED CHUNKS")
print("================================")


documents = results.get(
    "documents",
    [[]]
)[0]

metadatas = results.get(
    "metadatas",
    [[]]
)[0]

distances = results.get(
    "distances",
    [[]]
)[0]


if not documents:

    print("\nNO DOCUMENTS WERE RETRIEVED.")

else:

    for i in range(
        len(documents)
    ):

        print(
            f"\n--- Result {i + 1} ---"
        )

        print(
            "Source:",
            metadatas[i]["source"]
        )

        print(
            "Page:",
            metadatas[i]["page"]
        )

        print(
            "Quarter:",
            metadatas[i]["quarter"]
        )

        print(
            "Distance:",
            distances[i]
        )

        print("\nText:")

        print(documents[i])