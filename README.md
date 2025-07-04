# 🧠 Chat with Your PDF – FastAPI + LangChain + OpenAI

This project allows users to upload a PDF and chat with it using OpenAI's GPT models. It extracts, chunks, embeds, and stores your PDF content for semantic Q&A.

---

## 🚀 Features

- 📄 Upload any PDF file
- 🧩 Chunk and embed PDF content using OpenAI
- ⚡ Vector similarity search using FAISS
- 💬 Ask questions and get accurate answers from the document
- 🧠 Built with FastAPI + LangChain

---

## 📁 Project Structure
│
├── main.py # FastAPI app
├── templates/
│ ├── index.html # Upload UI
│ └── chat.html # Chat interface
├── static/ # Static assets
├── requirements.txt # Python dependencies
├── Dockerfile # Docker configuration
├── .env # OpenAI API key
└── README.md # You're here!



## 🛠️ Tech Stack

| Layer        | Tool                                     |
|--------------|------------------------------------------|
| Backend      | FastAPI                                  |
| Embeddings   | OpenAI + LangChain                       |
| Vector Store | FAISS                                    |
| PDF Parsing  | PyPDFLoader (LangChain)                  |
| LLM          | OpenAI GPT-3.5-turbo or GPT-4            |
| Frontend     | Jinja2 templates                         |
| Deployment   | Docker + GitHub Actions (CI/CD)          |

---

## ⚙️ Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/<your-username>/chatbot_api.git
cd chatbot_api


python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

#Install Requirements

pip install -r requirements.txt

▶️ Run the App

uvicorn main:app --reload