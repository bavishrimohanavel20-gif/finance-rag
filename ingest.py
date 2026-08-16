from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_FOLDER = Path("data")


# Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)


def process_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    all_chunks = []

    # Get the quarter from the filename
    quarter = pdf_path.stem.replace("Infosys_", "").replace("_FY26", "")

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if not text:
            continue

        # Split page text into smaller chunks
        chunks = text_splitter.split_text(text)

        for chunk in chunks:

            chunk_data = {
                "text": chunk,
                "source": pdf_path.name,
                "page": page_number,
                "quarter": quarter
            }

            all_chunks.append(chunk_data)

    return all_chunks


# Process all PDFs
all_chunks = []

for pdf_file in sorted(DATA_FOLDER.glob("*.pdf")):

    print(f"\nProcessing: {pdf_file.name}")

    chunks = process_pdf(pdf_file)

    print(f"Chunks created: {len(chunks)}")

    all_chunks.extend(chunks)


print("\n==============================")
print("TOTAL CHUNKS:", len(all_chunks))
print("==============================")