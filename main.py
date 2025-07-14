from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import tempfile
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import uuid
from dotenv import load_dotenv
import re

app = FastAPI()

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

sessions = {}
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class Session:
    def __init__(self):
        self.vector_store = None
        self.filename = None
        self.threads = {} 

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_pdf(request: Request, files: List[UploadFile] = File(...)):
    session_id = str(uuid.uuid4())
    session = Session()
    sessions[session_id] = session
    session.filename = ", ".join([file.filename for file in files])

    all_chunks = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(pages)
        all_chunks.extend(chunks)

        os.unlink(tmp_path)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    session.vector_store = FAISS.from_documents(all_chunks, embeddings)

    return RedirectResponse(url=f"/chat/{session_id}", status_code=303)

@app.get("/chat/{session_id}", response_class=HTMLResponse)
async def chat_page(request: Request, session_id: str):
    session = sessions.get(session_id)
    if not session:
        return RedirectResponse(url="/")
    return templates.TemplateResponse(
        "chat.html", 
        {
            "request": request,
            "filename": session.filename,
            "session_id": session_id
        }
    )

@app.post("/ask/{session_id}/{thread_id}")
async def ask_question(session_id: str, thread_id: str, question: str = Form(...)):
    session = sessions.get(session_id)
    if not session or not session.vector_store:
        raise HTTPException(status_code=404, detail="Session not found")

    thread = session.threads.get(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    retriever = session.vector_store.as_retriever(
        search_kwargs={
            "k": 8,
            "score_threshold": 0.5,
            "fetch_k": 30
        }
    )

    try:
        docs = retriever.get_relevant_documents(question)
    except Exception as e:
        print(f"Retrieval error: {str(e)}")
        return {"answer": "Error searching document", "sources": []}

    if not docs or all(len(doc.page_content.strip()) < 30 for doc in docs):
        answer = "This information is not available in the uploaded PDF."
        session.threads[thread_id].append((question, answer))
        return {"answer": answer, "sources": []}

    prompt_template = """
    You are a helpful assistant summarizing and explaining information from uploaded documents.
    Use only the given context to answer the question in a helpful, clear, and concise way.

    Answer structure:
    1. Give a direct answer
    2. Highlight key points as a list if appropriate
    3. Explain using plain English

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    llm = ChatOpenAI(
        model="gpt-4-turbo",
        temperature=0.5,
        max_tokens=800
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    result = qa({"query": question})
    answer = result["result"]
    source_docs = result.get("source_documents", [])

    if "not mentioned" in answer.lower() or "not available" in answer.lower():
        answer = "This information is not available in the uploaded PDF."
    elif len(answer.split()) < 30:
        answer = qa({
            "query": f"Expand this answer with more details from the context: {question}"
        })["result"]

    answer = re.sub(r"\*\*", "", answer)
    sources = []
    for doc in source_docs:
        content = doc.page_content.strip()
        if not content:
            continue
        sources.append({
            "page": doc.metadata.get("page", "N/A"),
            "file": doc.metadata.get("source", "Uploaded File"),
            "text": content[:300] + "..." if len(content) > 300 else content
        })

    session.threads[thread_id].append((question, answer))

    return {
        "answer": answer,
        "sources": sources
    }

@app.post("/chat/{session_id}/new-thread")
async def new_thread(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    thread_id = str(uuid.uuid4())
    session.threads[thread_id] = []

    return {"thread_id": thread_id}

@app.get("/chat/{session_id}/{thread_id}/history")
async def get_thread_history(session_id: str, thread_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    thread = session.threads.get(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="Thread not found")

    return {"messages": [{"question": q, "answer": a} for q, a in thread]}

@app.get("/chat/{session_id}/threads")
async def list_threads(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"thread_ids": list(session.threads.keys())}