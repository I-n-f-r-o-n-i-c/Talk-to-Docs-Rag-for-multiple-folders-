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
import fitz
import pdfplumber
from PIL import Image
import io
import csv
import json

# ... (keep the rest of the imports and global variables)

Enhanced RAG Streamlit Application

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
import fitz
import pdfplumber
from PIL import Image
import io
import csv
import json

# ... (keep the rest of the imports and global variables)

def init_session_state():
    if 'folder_name' not in st.session_state:
        st.session_state['folder_name'] = ""
    if 'selected_folder' not in st.session_state:
        st.session_state['selected_folder'] = None
    if 'answer_visible' not in st.session_state:
        st.session_state['answer_visible'] = False
    if 'serialized_docs' not in st.session_state:
        st.session_state['serialized_docs'] = []
    if 'uploader_key' not in st.session_state:
        st.session_state['uploader_key'] = 0
    if 'question' not in st.session_state:
        st.session_state['question'] = ""
    if 'answer' not in st.session_state:
        st.session_state['answer'] = ""
    if 'expanded_chunk' not in st.session_state:
        st.session_state['expanded_chunk'] = None
    if 'GROQ_API_KEY' not in st.session_state:
        st.session_state['GROQ_API_KEY'] = ""
    if 'selected_model' not in st.session_state:
        st.session_state['selected_model'] = ""

def main():
    st.set_page_config(page_title="Talk to Docs", page_icon=":books:", layout="wide")
    st.title("📚 Talk to Your Documents")

    init_session_state(

def main():
    st.set_page_config(page_title="Talk to Docs", page_icon=":books:", layout="wide")
    st.title("📚 Talk to Your Documents")

    init_session_state()

    # Sidebar with improved styling
    with st.sidebar:
        st.markdown("## 🔧 Configuration")
        GROQ_API_KEY = st.text_input("🔑 Enter Groq API Key", type="password", value=st.session_state['GROQ_API_KEY'])
        if GROQ_API_KEY:
            st.session_state['GROQ_API_KEY'] = GROQ_API_KEY
        else:
            st.warning("⚠️ Please enter a valid Groq API Key")
        
        st.markdown("---")
        st.markdown("## 📁 Upload Documents")
        folder_name = st.text_input("📂 Enter Folder Name", value=st.session_state['folder_name'], key="folder_name_input")
        files = st.file_uploader("📄 Upload PDF Files", accept_multiple_files=True, type=["pdf"], key=f"uploader_{st.session_state['uploader_key']}")

        if st.button("🔍 Process Folder", key="process_folder"):
            if not folder_name or not files:
                st.error("❌ Please provide a folder name and select files to upload.")
            else:
                with st.spinner("Processing files..."):
                    folder_path = save_files(files, folder_name)
                    st.success(f"✅ Files saved to {folder_path}")
                    vector_db_path = process_folder(folder_path)
                    clear_folder(folder_path)
                    st.success(f"✅ Vector DB created at {vector_db_path} and upload folder cleared")
                    reset_question_and_chunks()
                    st.session_state['folder_name'] = ""
                    st.session_state['uploader_key'] += 1
                    st.rerun()

        st.markdown("---")
        st.markdown("## 🤖 Model Selection")
        models_list = [
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "llama-3.2-11b-text-preview",
            "llama-3.2-11b-vision-preview",
            "llama-3.2-1b-preview",
            "llama-3.2-3b-preview",
            "llama-3.2-90b-text-preview",
            "llama-guard-3-8b",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8×7b-32768"
        ]
        selected_model = st.selectbox("🧠 Select Model", models_list)

        st.markdown("---")
        st.markdown("## 📊 Processed Folders")
        folders = [f for f in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, f))]
        for folder in folders:
            col1, col2 = st.columns([9, 1])
            with col1:
                if st.button(f"📁 {folder}", key=folder):
                    st.session_state['selected_folder'] = folder
                    reset_question_and_chunks()
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete-{folder}"):
                    delete_specific_folder(folder)
                    st.success(f"🗑️ Deleted {folder} from output folder")
                    if st.session_state['selected_folder'] == folder:
                        st.session_state['selected_folder'] = None
                    st.rerun()

        st.markdown("---")
        if st.button("🧹 Clear Output Folder"):
            clear_folder(output_folder)
            os.makedirs(output_folder)
            st.success("🧹 Cleared output folder")
            reset_question_and_chunks()
            st.session_state['selected_folder'] = None
            st.rerun()

    # Main content
    selected_folder = st.session_state['selected_folder']

    if selected_folder:
        vector_db_path = os.path.join(output_folder, selected_folder, "VectorDB")
        if os.path.exists(vector_db_path):
            st.sidebar.success(f"🔗 Connected to Vector DB: {vector_db_path}")
            db = FAISS.load_local(vector_db_path, embeddings=embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Loaded vector store from {vector_db_path}")
        else:
            if st.sidebar.button("🔄 Process Vector DB"):
                with st.spinner("Processing Vector DB..."):
                    process_folder(os.path.join(upload_folder, selected_folder))
                    st.sidebar.success(f"✅ Vector DB processed for {selected_folder}")
                    st.rerun()
        
        # Layout: Two columns - main content and retrieved chunks
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("## 🔍 Ask Your Documents")
            question = st.text_input("💬 Ask a question", value=st.session_state['question'])
            if st.button("🚀 Submit", key="submit_question"):
                if not vector_db_path or not os.path.exists(vector_db_path):
                    st.error("❌ Please process the folder to create a Vector DB first.")
                else:
                    with st.spinner("Thinking..."):
                        docs = db.similarity_search_with_score(question, k=10)
                        logger.info(f"Performed similarity search for question: {question}")

                        serialized_docs = []
                        for doc_tuple in docs:
                            document, score = doc_tuple
                            serialized_doc = {
                                "pdf_name": document.metadata['pdf_name'],
                                "page_number": document.metadata['page_number'],
                                "chunk_index": document.metadata['chunk_index'],
                                "page_content": document.page_content,
                                "score": float(score)
                            }
                            serialized_docs.append(serialized_doc)

                        llm = ChatGroq(model=selected_model, api_key=st.session_state['GROQ_API_KEY'])

                        prompt = ChatPromptTemplate.from_messages([
                            ("system", "You are an AI assistant with access to a large knowledge base. Your task is to provide accurate and helpful responses solely based on the retrieved information. Follow these guidelines: 1. Analyze the user's question carefully. 2. Review the retrieved information provided in the context. 3. Formulate a response that directly addresses the user's query. 4. If the retrieved information is insufficient, just say Not specified in the context. 6. If you're unsure or the information is ambiguous, communicate this clearly to the user. 7. Keep your answers precise and concise. Remember, your primary goal is to assist the user with accurate and relevant information."),
                            ("human", "context:{context} \n question:{question}"),
                        ])

                        chain = prompt | llm
                        answer = chain.invoke({"context": docs, "question": question})

                        if not answer.content:
                            st.error("❌ No response received from the AI model.")
                            st.session_state['answer_visible'] = False
                            st.session_state['serialized_docs'] = []
                        else:
                            st.session_state['answer'] = answer.content
                            st.session_state['answer_visible'] = True
                            st.session_state['serialized_docs'] = serialized_docs
                            st.session_state['question'] = question
                            st.rerun()

            if st.session_state['answer_visible']:
                st.markdown("### 🤖 AI Response:")
                st.info(st.session_state['answer'])

        with col2:
            st.markdown("## 📑 Retrieved Chunks")
            
            if st.session_state['serialized_docs'] and st.session_state['answer_visible']:
                for i, doc in enumerate(st.session_state['serialized_docs']):
                    with st.expander(f"📄 {doc['pdf_name']} | Page: {doc['page_number']} | Chunk: {doc['chunk_index']}"):
                        st.markdown(f"**Content:** {doc['page_content'][:500]}...")
                        st.progress(1 - doc['score'])  # Convert score to a progress bar
                        st.markdown(f"**Relevance Score:** {1 - doc['score']:.2f}")
            else:
                st.info("No chunks to display. Ask a question to see relevant document sections.")
    else:
        st.info("👈 No folder selected. Please select a folder from the sidebar or upload new documents.")

if __name__ == "__main__":
    main()
