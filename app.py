import streamlit as st
import os
import shutil
from pdf_extractor import PDFExtractor
from document_chunker import DocumentChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import logging

# Initialize global variables
embeddings = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')
output_folder = "Output"
if not os.path.exists("Output"):
    os.makedirs("Output")
upload_folder = "Uploads"
if not os.path.exists("Uploads"):
    os.makedirs("Uploads")

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_files(files, folder_name):
    folder_path = os.path.join(upload_folder, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file.name)
        with open(file_path, 'wb') as f:
            f.write(file.getbuffer())
    return folder_path

def process_folder(input_folder):
    folder_basename = os.path.basename(input_folder)
    output_folder_path = os.path.join(output_folder, folder_basename)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    for pdf_file in os.listdir(input_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, pdf_file)
            pdf_basename = os.path.splitext(pdf_file)[0]
            pdf_output_folder = os.path.join(output_folder_path, pdf_basename)
            if not os.path.exists(pdf_output_folder):
                os.makedirs(pdf_output_folder)
            pdf_extractor = PDFExtractor(pdf_path, pdf_output_folder)
            pdf_extractor.extract_all_data()

    chunks_folder = os.path.join(output_folder_path, "Chunks")
    if not os.path.exists(chunks_folder):
        os.makedirs(chunks_folder)

    chunked_text_all = []
    for pdf_folder in os.listdir(output_folder_path):
        pdf_folder_path = os.path.join(output_folder_path, pdf_folder)
        if os.path.isdir(pdf_folder_path) and pdf_folder not in ["Chunks", "VectorDB"]:
            document_chunker = DocumentChunker(pdf_folder_path, pdf_folder)
            chunked_text = document_chunker.chunk_all_text(chunks_folder)
            for chunk in chunked_text:
                chunk["pdf_name"] = pdf_folder
            chunked_text_all.extend(chunked_text)

    documents = [
        Document(page_content=chunk["text_chunk"], metadata={"page_number": chunk["page_number"], "chunk_index": chunk["chunk_index"], "pdf_name": chunk["pdf_name"]})
        for chunk in chunked_text_all
    ]

    db = FAISS.from_documents(documents, embeddings)
    vector_db_path = os.path.join(output_folder_path, "VectorDB")
    if not os.path.exists(vector_db_path):
        os.makedirs(vector_db_path)
    db.save_local(vector_db_path)
    return vector_db_path

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

def delete_specific_folder(folder_name):
    folder_path = os.path.join(output_folder, folder_name)
    clear_folder(folder_path)

# Initialize session state variables for configuration
def init_session_state():
    if 'GROQ_API_KEY' not in st.session_state:
        st.session_state['GROQ_API_KEY'] = ""
    if 'folder_name' not in st.session_state:
        st.session_state['folder_name'] = ""
    if 'selected_model' not in st.session_state:
        st.session_state['selected_model'] = ""
    if 'question' not in st.session_state:
        st.session_state['question'] = ""
    if 'answer' not in st.session_state:
        st.session_state['answer'] = ""
    if 'retrieved_chunks' not in st.session_state:
        st.session_state['retrieved_chunks'] = []

def main():
    # Initialize session state variables
    init_session_state()
    
    st.set_page_config(page_title="Talk to Docs", layout="wide")

    # Custom HTML/CSS for sidebar and layout
    st.markdown("""
    <style>
    .sidebar-content {
        background: linear-gradient(to bottom, #7C3AED, #4338CA);
        color: white;
        padding: 20px;
        border-radius: 10px;
    }
    .stApp {
        background: linear-gradient(to bottom right, #F3E8FF, #E0F2FE);
    }
    .main-content {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 8px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .chunk-item {
        border: 1px solid #ddd;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 8px;
    }
    .relevance-bar {
        background-color: #e0e0e0;
        border-radius: 5px;
        overflow: hidden;
        height: 12px;
    }
    .relevance-progress {
        height: 100%;
        border-radius: 5px;
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar Configuration
    with st.sidebar:
        st.markdown("## 🔧 Configuration", unsafe_allow_html=True)
        
        # Groq API Key input
        st.session_state['GROQ_API_KEY'] = st.text_input("🔑 Groq API Key", type="password", value=st.session_state['GROQ_API_KEY'])
        
        # Folder name input
        st.session_state['folder_name'] = st.text_input("📂 Folder Name", value=st.session_state['folder_name'])
        
        # Upload PDFs
        uploaded_files = st.file_uploader("📄 Upload PDFs", accept_multiple_files=True, type=['pdf'])
        
        # Model selection
        st.session_state['selected_model'] = st.selectbox("🤖 Select Model", [
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant"
        ], index=0)

        # Submit button for processing folder
        if st.button("🔍 Process Folder"):
            if not st.session_state['folder_name'] or not uploaded_files:
                st.error("Please provide a folder name and select PDFs to upload.")
            else:
                # Process PDFs (mock function for now)
                st.success(f"Processed {len(uploaded_files)} PDFs and created vector DB.")

    # Main content layout
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)

    # Title for main content
    st.markdown("## 📚 Talk to Docs", unsafe_allow_html=True)

    # Input for asking questions
    question = st.text_input("💬 Ask Your Documents", value=st.session_state['question'])
    
    # Submit button for asking question
    if st.button("🚀 Submit"):
        # Perform search in vector DB (mocked result here)
        st.session_state['question'] = question
        st.session_state['answer'] = "AI's response to the question based on the document content."
        st.session_state['retrieved_chunks'] = [
            {"pdf_name": "Sample.pdf", "page_number": 1, "chunk_index": 1, "content": "Sample text from document...", "score": 0.8},
            {"pdf_name": "Example.pdf", "page_number": 2, "chunk_index": 3, "content": "Another text snippet...", "score": 0.65}
        ]
    
    # Display AI response
    if st.session_state['answer']:
        st.markdown("### 🤖 AI Response:")
        st.info(st.session_state['answer'])

    # Display retrieved document chunks
    if st.session_state['retrieved_chunks']:
        st.markdown("### 📑 Retrieved Chunks:")
        for chunk in st.session_state['retrieved_chunks']:
            with st.expander(f"📄 {chunk['pdf_name']} | Page: {chunk['page_number']} | Chunk: {chunk['chunk_index']}"):
                st.markdown(chunk['content'])
                st.markdown(f"**Relevance Score:** {chunk['score']:.2f}")
                st.markdown(f"<div class='relevance-bar'><div class='relevance-progress' style='width: {chunk['score']*100}%'></div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
