# 🩺 Medical ChatBot using RAG (Retrieval-Augmented Generation)

This project builds a **medical question-answering chatbot** that reads a medical PDF book and answers questions in **simple, understandable language**.

Unlike normal chatbots that rely only on pre-trained knowledge, this system uses **RAG (Retrieval-Augmented Generation)** — meaning it retrieves the correct information from the PDF first, then generates the answer.

---

## 📌 Problem Statement

Most chatbots:

- ❌ hallucinate answers  
- ❌ cannot read documents  
- ❌ give generic or incorrect medical advice  

Doctors, students, and patients need **trustworthy, document-based answers**.

> 🎯 **Goal:** Build a chatbot that reads a medical PDF and gives accurate, contextual responses — like a medical assistant — but easy to understand.

RAG solves this by combining:

✔ Information Retrieval  
✔ Transformer-based Generation  

---

## 📊 Data Source

We use a **Medical Reference PDF Book** provided by the user.

The process:

- Extract text from the PDF  
- Split it into smaller chunks  
- Store them in a searchable vector database  

When a question is asked, only relevant sections are retrieved — reducing hallucinations.

---

## 🧠 Technology Used

| Component | Tool |
|----------|------|
| UI | Streamlit |
| Framework | LangChain |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | FAISS |
| LLM | Google FLAN-T5 |
| PDF Loader | PyPDFLoader |

---

## ⚙️ System Architecture

- **PDF Loader** → extracts text  
- **Chunk Splitter** → breaks text into small segments  
- **Embedding Model** → converts text to numeric vectors  
- **FAISS DB** → stores vectors for fast search  
- **Retriever** → finds most relevant chunks  
- **Transformer Model** → generates final answer using retrieved context  

---

## 🔍 Workflow (RAG Pipeline)

1️⃣ Load PDF  
2️⃣ Split text into chunks  
3️⃣ Convert chunks → embeddings  
4️⃣ Store embeddings in FAISS  
5️⃣ User asks a question  
6️⃣ Retriever finds relevant chunks  
7️⃣ Transformer generates answer using only the retrieved context  

