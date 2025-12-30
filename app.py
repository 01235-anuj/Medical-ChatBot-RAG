import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


PDF_FILE = "Medical_Book.pdf"

# Load PDF
loader = PyPDFLoader(PDF_FILE)
documents = loader.load()

# Small chunks
text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
docs = text_splitter.split_documents(documents)

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector DB
vectordb = FAISS.from_documents(docs, embeddings)

# Retriever (ONLY 1 BEST CHUNK)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})


# Transformer model with truncation
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    max_length=250,
    truncation=True
)

llm = HuggingFacePipeline(pipeline=generator)

# ---- CUSTOM PROMPT (short context only) ----
template = """
You are a medical assistant.

Use ONLY the context below to answer.
Explain clearly in 4 to 6 sentences.
If answer is not present, say: "I don't know from the document."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"],
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=False,
)


def ask_bot(query):
    try:
        answer = qa.run(query)
        return answer + "\n\n⚠️ Educational only — not medical advice."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


# UI
st.title("🚀 Medical ChatBot AI (Transformer + RAG)")
q = st.text_input("Enter your medical question:", key="question_box")

if st.button("Get Answer") and q.strip():
    st.write(ask_bot(q))
