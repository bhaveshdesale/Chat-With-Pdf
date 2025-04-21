
# Fiass PDF chatbot using local LLM

import streamlit as st
import os
import base64
import torch
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Configure Streamlit page
st.set_page_config(page_title="Chat with PDFs", layout="wide")

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model checkpoint
checkpoint = "MBZUAI/LaMini-T5-738M"

# Load model & tokenizer
@st.cache_resource
def load_model():
    os.makedirs('./offload_weights', exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    return tokenizer, model

# Read PDFs manually and split text
@st.cache_resource
def data_ingestion():
    try:
        documents = []
        for root, _, files in os.walk("docs"):
            for file in files:
                if file.endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        full_text = ""
                        for page in reader.pages:
                            full_text += page.extract_text() or ""
                        documents.append(Document(page_content=full_text))

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_documents(texts, embeddings)
        return db
    except Exception as e:
        st.error(f"Error during embedding creation: {e}")
        return None

# Setup the pipeline
@st.cache_resource
def llm_pipeline(_tokenizer, _base_model):
    pipe = pipeline(
        'text2text-generation',
        model=_base_model,
        tokenizer=_tokenizer,
        device=device.index if torch.cuda.is_available() else -1,
        max_length=512,
        min_length=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    return HuggingFacePipeline(pipeline=pipe)

# Create QA chain
@st.cache_resource
def qa_chain(_db, _tokenizer, _base_model):
    retriever = _db.as_retriever()
    llm = llm_pipeline(_tokenizer, _base_model)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

# Display PDF
@st.cache_data
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Main app
def main():
    st.title("Chat with Your PDF 📄")

    tokenizer, base_model = load_model()

    uploaded_file = st.file_uploader("Upload a PDF:", type="pdf")
    if uploaded_file:
        os.makedirs("docs", exist_ok=True)
        file_path = os.path.join("docs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.markdown("### File Preview")
        display_pdf(file_path)

        # Create embeddings
        with st.spinner("Processing PDF and creating embeddings..."):
            db = data_ingestion()

        if db:
            st.success("Embeddings created successfully!")

            qa = qa_chain(db, tokenizer, base_model)

            if "conversation_history" not in st.session_state:
                st.session_state.conversation_history = []

            user_query = st.text_input("Ask a question about the PDF:")
            if user_query:
                with st.spinner("Fetching response..."):
                    response = qa({"query": user_query})
                    answer = response['result']
                    st.session_state.conversation_history.append({"question": user_query, "answer": answer})

                st.markdown("### Q&A History")
                for qa_pair in st.session_state.conversation_history:
                    st.write(f"**Q:** {qa_pair['question']}")
                    st.write(f"**A:** {qa_pair['answer']}")
                    st.write("---")
        else:
            st.error("Failed to process the PDF. Please check logs for details.")

if __name__ == "__main__":
    main()




# Chroma pdf chatbot


# import os
# import fitz  # PyMuPDF
# import streamlit as st
# from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.chains import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAI
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import torch

# # Global Variables
# checkpoint = "facebook/bart-large-cnn"  # Pretrained model checkpoint
# device = "cuda" if torch.cuda.is_available() else "cpu"  # Choose device based on availability

# # Function to display PDF content (for preview purposes)
# def display_pdf(file_path):
#     with open(file_path, "rb") as file:
#         st.download_button("Download PDF", file, file_name=file_path, mime="application/pdf")
#         st.markdown(f'<embed src="data:application/pdf;base64,{file.read().encode("base64")}" width="100%" height="800px">', unsafe_allow_html=True)

# # Function for PDF data ingestion and embedding creation
# @st.cache_data  # Cache processed data (documents and embeddings) instead of models
# def data_ingestion():
#     try:
#         documents = []
#         for root, _, files in os.walk("docs"):
#             for file in files:
#                 if file.endswith(".pdf"):
#                     file_path = os.path.join(root, file)
#                     # Using PyMuPDF to load PDF content
#                     doc = fitz.open(file_path)
#                     text = ""
#                     for page in doc:
#                         text += page.get_text()

#                     # Split the text into smaller chunks
#                     documents.append({"text": text, "metadata": {"file_name": file}})

#         # Increase chunk size and overlap for better context
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         texts = text_splitter.split_documents(documents)

#         # Using SentenceTransformer with better embeddings
#         embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
        
#         # Using Chroma for vector storage
#         db = Chroma.from_documents(texts, embeddings)
#         return db
#     except Exception as e:
#         st.error(f"Error during embedding creation: {e}")
#         return None

# # Function to load the model and tokenizer
# def load_model():
#     tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#     model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
#     return tokenizer, model

# # Main function to run the Streamlit app
# def main():
#     st.set_page_config(page_title="Chat with Your PDF 📄", layout="wide")
#     st.title("Chat with Your PDF 📄")

#     # Load the model synchronously
#     tokenizer, base_model = load_model()

#     uploaded_file = st.file_uploader("Upload a PDF:", type="pdf")
#     if uploaded_file:
#         # Save uploaded file
#         os.makedirs("docs", exist_ok=True)
#         file_path = os.path.join("docs", uploaded_file.name)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.read())

#         st.markdown("### File Preview")
#         display_pdf(file_path)

#         # Create embeddings
#         with st.spinner("Processing PDF and creating embeddings..."):
#             db = data_ingestion()
#         if db:
#             st.success("Embeddings created successfully!")

#             # Load QA chain
#             qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model="gpt-3.5-turbo"), db.as_retriever())

#             # Memory for storing conversation history
#             if "conversation_history" not in st.session_state:
#                 st.session_state.conversation_history = []

#             # User interaction
#             user_query = st.text_input("Ask a question about the PDF:")
#             if user_query:
#                 with st.spinner("Fetching response..."):
#                     response = qa({"query": user_query})
#                     answer = response['result']

#                     # Store the question and answer
#                     st.session_state.conversation_history.append({"question": user_query, "answer": answer})

#                 # Display all questions and answers
#                 st.markdown("### Q&A History")
#                 for qa_pair in st.session_state.conversation_history:
#                     st.write(f"**Q:** {qa_pair['question']}")
#                     st.write(f"**A:** {qa_pair['answer']}")
#                     st.write("---")

#         else:
#             st.error("Failed to process the PDF. Please check logs for details.")

# if __name__ == "__main__":
#     main()


































# import streamlit as st
# import os
# import base64
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# from langchain.document_loaders import PDFMinerLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.llms import HuggingFacePipeline
# from langchain.chains import RetrievalQA
# from streamlit_chat import message

# # Configure the Streamlit page
# st.set_page_config(page_title="Chat with PDFs", layout="wide")

# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Model checkpoint and offload folder
# checkpoint = "MBZUAI/LaMini-T5-738M"
# offload_folder = "./offload_weights"

# # Cache and load the tokenizer and model
# @st.cache_resource
# def load_model():
#     os.makedirs(offload_folder, exist_ok=True)
#     tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#     base_model = AutoModelForSeq2SeqLM.from_pretrained(
#         checkpoint,
#         device_map="auto",
#         torch_dtype=torch.float32,
#         offload_folder=offload_folder
#     )
#     return tokenizer, base_model

# # Process documents and create embeddings using FAISS
# @st.cache_resource
# def data_ingestion():
#     try:
#         documents = []
#         for root, _, files in os.walk("docs"):
#             for file in files:
#                 if file.endswith(".pdf"):
#                     loader = PDFMinerLoader(os.path.join(root, file))
#                     documents += loader.load()

#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         texts = text_splitter.split_documents(documents)

#         embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#         db = FAISS.from_documents(texts, embeddings)
#         return db
#     except Exception as e:
#         st.error(f"Error during embedding creation: {e}")
#         return None

# # Create a pipeline for the model with adjustments to generate longer responses
# @st.cache_resource
# def llm_pipeline(_tokenizer, _base_model):
#     pipe = pipeline(
#         'text2text-generation',
#         model=_base_model,
#         tokenizer=_tokenizer,
#         max_length=512,  # Increased max length to generate longer responses
#         min_length=100,  # Set a minimum length for the answer
#         do_sample=True,
#         temperature=0.7,  # Slightly higher temperature for more varied but controlled output
#         top_p=0.9
#     )
#     return HuggingFacePipeline(pipeline=pipe)

# # Create a QA chain using the model and database
# @st.cache_resource
# def qa_chain(_db, _tokenizer, _base_model):
#     retriever = _db.as_retriever()
#     llm = llm_pipeline(_tokenizer, _base_model)
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True
#     )
#     return qa

# # Display PDF in the app
# @st.cache_data
# def display_pdf(file_path):
#     with open(file_path, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')
#     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600"></iframe>'
#     st.markdown(pdf_display, unsafe_allow_html=True)

# # Main function to run the Streamlit app
# def main():
#     st.title("Chat with Your PDF 📄")

#     tokenizer, base_model = load_model()

#     uploaded_file = st.file_uploader("Upload a PDF:", type="pdf")
#     if uploaded_file:
#         # Save uploaded file
#         os.makedirs("docs", exist_ok=True)
#         file_path = os.path.join("docs", uploaded_file.name)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.read())

#         st.markdown("### File Preview")
#         display_pdf(file_path)

#         # Create embeddings
#         with st.spinner("Processing PDF and creating embeddings..."):
#             db = data_ingestion()
#         if db:
#             st.success("Embeddings created successfully!")

#             # Load QA chain
#             qa = qa_chain(db, tokenizer, base_model)

#             # User interaction
#             user_query = st.text_input("Ask a question about the PDF:")
#             if user_query:
#                 with st.spinner("Fetching response..."):
#                     response = qa({"query": user_query})
#                     st.write("**Answer:**", response['result'])
#         else:
#             st.error("Failed to process the PDF. Please check logs for details.")

# if __name__ == "__main__":
#     main()
