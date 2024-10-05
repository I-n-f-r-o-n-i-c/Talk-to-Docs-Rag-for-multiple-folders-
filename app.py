import streamlit as st
import os
import shutil
from pdf_extractor import PDFExtractor
from document_chunker import DocumentChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
# from ollama import Client
import ollama
import logging

# Initialize global variables
embeddings = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')
output_folder = "Output"
upload_folder = "Uploads"

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

def reset_question_and_chunks():
    st.session_state['question'] = ""
    st.session_state['serialized_docs'] = []
    st.session_state['answer_visible'] = False
    st.session_state['answer'] = ""

def main():
    st.set_page_config(page_title="Talk to Docs", page_icon=":page_with_curl:")
    st.title("Talk to Docs")

    init_session_state()

    # Sidebar
    with st.sidebar:
        folder_name = st.text_input("Enter Folder Name", value=st.session_state['folder_name'], key="folder_name_input")
        files = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"], key=f"uploader_{st.session_state['uploader_key']}")

        if st.button("Process Folder"):
            if not folder_name or not files:
                st.error("Please provide a folder name and select files to upload.")
            else:
                output_folder_path = os.path.join(output_folder, folder_name)
                if os.path.exists(output_folder_path):
                    st.warning(f"A folder named '{folder_name}' already exists.")
                    if st.button("Delete existing folder and proceed"):
                        shutil.rmtree(output_folder_path)
                        folder_path = save_files(files, folder_name)
                        st.success(f"Files saved to {folder_path}")
                        vector_db_path = process_folder(folder_path)
                        clear_folder(folder_path)
                        st.success(f"Vector DB created at {vector_db_path} and upload folder cleared")
                        reset_question_and_chunks()
                        st.session_state['folder_name'] = ""
                        st.session_state['uploader_key'] += 1
                        st.rerun()
                    elif st.button("Cancel"):
                        st.stop()
                else:
                    folder_path = save_files(files, folder_name)
                    st.success(f"Files saved to {folder_path}")
                    vector_db_path = process_folder(folder_path)
                    clear_folder(folder_path)
                    st.success(f"Vector DB created at {vector_db_path} and upload folder cleared")
                    reset_question_and_chunks()
                    st.session_state['folder_name'] = ""
                    st.session_state['uploader_key'] += 1
                    st.rerun()

        st.markdown("---")

        # Fetch and list models dynamically
        models_list = client.list()
        model_names = [model['name'] for model in models_list['models']]
        selected_model = st.selectbox("Select Model", model_names)

        st.markdown("---")

        folders = [f for f in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, f))]
        for folder in folders:
            col1, col2 = st.columns([9, 1])
            with col1:
                if st.button(folder, key=folder):
                    st.session_state['selected_folder'] = folder
                    reset_question_and_chunks()
                    st.rerun()
            with col2:
                if st.button("❌", key=f"delete-{folder}"):
                    delete_specific_folder(folder)
                    st.success(f"Deleted {folder} from output folder")
                    if st.session_state['selected_folder'] == folder:
                        st.session_state['selected_folder'] = None
                    st.rerun()

        st.markdown("---")

        if st.button("Clear Output Folder"):
            clear_folder(output_folder)
            os.makedirs(output_folder)
            st.success("Cleared output folder")
            reset_question_and_chunks()
            st.session_state['selected_folder'] = None
            st.rerun()

    # Main content
    selected_folder = st.session_state['selected_folder']

    if selected_folder:
        vector_db_path = os.path.join(output_folder, selected_folder, "VectorDB")
        if os.path.exists(vector_db_path):
            st.sidebar.success(f"Connected to Vector DB at {vector_db_path}")
            db = FAISS.load_local(vector_db_path, embeddings=embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Loaded vector store from {vector_db_path}")
        else:
            if st.sidebar.button("Process Vector DB"):
                process_folder(os.path.join(upload_folder, selected_folder))
                st.sidebar.success(f"Vector DB processed for {selected_folder}")
                st.rerun()
        
        # Layout: Two columns - main content and retrieved chunks
        col1, col2 = st.columns([20, 20])

        with col1:
            question = st.text_input("Ask a question", value=st.session_state['question'])
            if st.button("Submit"):
                if not vector_db_path or not os.path.exists(vector_db_path):
                    st.error("Please process the folder to create a Vector DB first.")
                else:
                    # db = FAISS.load_local(vector_db_path, embeddings=embeddings, allow_dangerous_deserialization=True)
                    # logger.info(f"Loaded vector store from {vector_db_path}")

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

                    response = ollama.chat(
                        model=selected_model,
                        messages=[{'role': 'user', 'content': f"Context: {docs} \n Question: {question} \n Answer: "}],
                        stream=False,
                    )
                    if not response['message']['content']:
                        st.error("No response received from the AI model.")
                        st.session_state['answer_visible'] = False
                        st.session_state['serialized_docs'] = []
                    else:
                        st.session_state['answer'] = response['message']['content']
                        st.session_state['answer_visible'] = True
                        st.session_state['serialized_docs'] = serialized_docs
                        st.session_state['question'] = question
                        st.rerun()

            if st.session_state['answer_visible']:
                st.write("AI:", st.session_state['answer'])

        with col2:
            st.subheader("Retrieved Chunks")
            
            if st.session_state['serialized_docs'] and st.session_state['answer_visible']:
                for i in range(0, len(st.session_state['serialized_docs']), 2):
                    cols = st.columns([3, 3])
                    for j, col in enumerate(cols):
                        if i + j < len(st.session_state['serialized_docs']):
                            doc = st.session_state['serialized_docs'][i + j]
                            is_expanded = (st.session_state['expanded_chunk'] == i + j)
                            
                            with col:
                                with st.expander(f"PDF Name: {doc['pdf_name']} | Page Number: {doc['page_number']} | Chunk Index: {doc['chunk_index']}", expanded=is_expanded):
                                    st.markdown(f"**Content:** {doc['page_content'][:500]}...")
                                    st.markdown(f"**Score:** {doc['score']}")
                                
                                if is_expanded:
                                    st.session_state['expanded_chunk'] = i + j
                                elif st.session_state['expanded_chunk'] == i + j:
                                    st.session_state['expanded_chunk'] = None
            else:
                st.write("No chunks to display.")

            st.subheader("Logs")
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    logger.removeHandler(handler)
    else:
        st.write("No folder selected. Please select a folder from the sidebar.")

if __name__ == "__main__":
    main()
