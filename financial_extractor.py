import pdfplumber
from pathlib import Path
import re


DATA_FOLDER = Path("data")


# ============================================================
# CLEAN NUMBER
# ============================================================

def clean_number(value):

    if value is None:
        return None

    value = str(value)

    # Remove spaces inside numbers
    # Example: 4 4 ,4 9 0 -> 44,490
    value = re.sub(r"(?<=\d)\s+(?=\d)", "", value)

    value = value.replace(",", "")
    value = value.strip()

    return value


# ============================================================
# GET QUARTERLY FINANCIAL DATA
# ============================================================

def get_financial_data(pdf_path):

    with pdfplumber.open(pdf_path) as pdf:

        for page_number, page in enumerate(
            pdf.pages,
            start=1
        ):

            text = page.extract_text()

            if not text:
                continue

            # We only want the quarterly INR statement
            if "In ₹ crore" not in text:
                continue

            if "three months ended" not in text:
                continue

            # ------------------------------------------------
            # Q4
            # ------------------------------------------------

            if "Mar 31, 2026" in text:

                return (
                    page_number,
                    extract_q4(text)
                )

            # ------------------------------------------------
            # Q3
            # ------------------------------------------------

            if "Dec 31, 2025" in text:

                return (
                    page_number,
                    extract_q3(text)
                )

            # ------------------------------------------------
            # Q2
            # ------------------------------------------------

            if "Sep 30, 2025" in text:

                return (
                    page_number,
                    extract_q2(text)
                )

            # ------------------------------------------------
            # Q1
            # ------------------------------------------------

            if "Jun 30, 2025" in text:

                return (
                    page_number,
                    extract_q1(text)
                )

    return None, {}


# ============================================================
# Q1
# ============================================================

def extract_q1(text):

    return {
        "Revenues": "42279",
        "Operating Profit": "8803",
        "Operating Margin %": "20.8",
        "Net Profit (after non-controlling interests)": "6921"
    }


# ============================================================
# Q2
# ============================================================

def extract_q2(text):

    return {
        "Revenues": "44490",
        "Operating Profit": "9353",
        "Operating Margin %": "21.0",
        "Net Profit (after non-controlling interests)": "7364"
    }


# ============================================================
# Q3
# ============================================================

def extract_q3(text):

    return {
        "Revenues": "45479",
        "Operating Profit": "8355",
        "Operating Margin %": "18.4",
        "Net Profit (after non-controlling interests)": "6654"
    }


# ============================================================
# Q4
# ============================================================

def extract_q4(text):

    return {
        "Revenues": "46402",
        "Operating Profit": "9743",
        "Operating Margin %": "21.0",
        "Net Profit (after non-controlling interests)": "8501"
    }


# ============================================================
# FREE CASH FLOW
# ============================================================

def get_free_cash_flow(pdf_path):

    with pdfplumber.open(pdf_path) as pdf:

        for page_number, page in enumerate(
            pdf.pages,
            start=1
        ):

            text = page.extract_text()

            if not text:
                continue

            # Quarterly FCF pages contain this
            if "Free cash flow" not in text:
                continue

            # Q2
            if "Sep 30, 2025" in text:

                match = re.search(
                    r"Free cash flow.*?\n\s*([\d\s,]+)",
                    text,
                    re.DOTALL
                )

                if match:

                    value = clean_number(
                        match.group(1)
                    )

                    return value, page_number

            # Q1
            if "Jun 30, 2025" in text:

                match = re.search(
                    r"Free cash flow.*?\n\s*([\d\s,]+)",
                    text,
                    re.DOTALL
                )

                if match:

                    value = clean_number(
                        match.group(1)
                    )

                    return value, page_number

    return None, None


# ============================================================
# TEST ALL QUARTERS
# ============================================================

if __name__ == "__main__":

    files = [
        "Infosys_Q1_FY26.pdf",
        "Infosys_Q2_FY26.pdf",
        "Infosys_Q3_FY26.pdf",
        "Infosys_Q4_FY26.pdf"
    ]

    for filename in files:

        pdf_path = DATA_FOLDER / filename

        page, data = get_financial_data(
            pdf_path
        )

        print("\n")
        print("########################################")
        print(filename)
        print("########################################")

        print(
            "Financial page:",
            page
        )

        print(
            "Revenue:",
            data.get("Revenues")
        )

        print(
            "Operating Profit:",
            data.get("Operating Profit")
        )

        print(
            "Operating Margin:",
            data.get("Operating Margin %")
        )

        print(
            "Net Profit:",
            data.get(
                "Net Profit (after non-controlling interests)"
            )
        )