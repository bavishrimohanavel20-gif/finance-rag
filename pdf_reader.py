from pathlib import Path
from pypdf import PdfReader


DATA_FOLDER = Path("data")


def read_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    print(f"\nFile: {pdf_path.name}")
    print(f"Pages: {len(reader.pages)}")

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        print(f"\n--- Page {page_number} ---")

        if text:
            print(text[:500])
        else:
            print("No text extracted")


for pdf_file in sorted(DATA_FOLDER.glob("*.pdf")):
    read_pdf(pdf_file)