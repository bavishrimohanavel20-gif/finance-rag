Finance RAG – Infosys Financial Intelligence Assistant

An AI-powered financial intelligence application that allows users to ask questions about Infosys FY26 quarterly financial reports and receive grounded answers with document and page-level source references.

🚀 Live Demo

👉 https://finance-rag-j9z7yxvk75h67nghrlfxu4.streamlit.app/

 📌 Project Overview

Finance RAG is a document-based financial question-answering system built to analyze Infosys FY26 quarterly financial reports.

Users can ask natural-language questions such as:

- What was the revenue in Q1 FY26?
- What was the operating margin in Q4 FY26?
- What was the net profit in Q3 FY26?
- What was the revenue in the latest quarter?

The application provides financial answers along with the corresponding source document and page number for verification.

✨ Key Features

- 📊 Infosys FY26 financial dashboard
- 📄 Four quarterly financial reports
- 💬 Natural-language financial question answering
- 🔎 Revenue, operating profit, operating margin and net profit queries
- 📈 Quarterly financial comparison
- 📚 Document-based answers
- 🔗 Source and page-level verification
- 🛡️ Refusal when requested information is not available
- 🌐 Streamlit web interface
- 🚀 Cloud deployment using Streamlit Community Cloud

 📑 Financial Reports

The project uses the following Infosys FY26 quarterly reports:

| Quarter | Report |
|---|---|
| Q1 FY26 | Infosys_Q1_FY26.pdf |
| Q2 FY26 | Infosys_Q2_FY26.pdf |
| Q3 FY26 | Infosys_Q3_FY26.pdf |
| Q4 FY26 | Infosys_Q4_FY26.pdf |

💡 Example Questions

 Revenue

Question:
> What was the revenue in Q1 FY26?

Answer:
> Revenue: ₹42,279 crore

Operating Margin

Question:
> What was the operating margin in Q4 FY26?

Answer:
> Operating Margin: 21.0%

### Net Profit

Question:
> What was the net profit in Q3 FY26?

Answer:
> Net Profit: ₹6,654 crore

Latest Quarter

Question:
> What was the revenue in the latest quarter?

Answer:
> Revenue: ₹46,402 crore

 🏗️ Project Architecture

```text
                    ┌─────────────────────┐
                    │   Infosys FY26 PDFs │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   PDF Processing     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Financial Extraction │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ RAG / Retrieval      │
                    │ Pipeline             │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Financial Assistant  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Streamlit Dashboard  │
                    └─────────────────────┘
