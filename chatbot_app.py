import streamlit as st
import os
import base64
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Configure the Streamlit page
st.set_page_config(page_title="Chat with PDFs", layout="wide")

# Device configuration (use GPU if available, else fallback to CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model checkpoint and offload folder
checkpoint = "MBZUAI/LaMini-T5-738M"

# Cache and load the tokenizer and model
@st.cache_resource
def load_model():
    os.makedirs('./offload_weights', exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Load model directly to the desired device (CPU/GPU)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)

    return tokenizer, model

# Process documents and create embeddings using FAISS
@st.cache_resource
def data_ingestion():
    try:
        documents = []
        for root, _, files in os.walk("docs"):
            for file in files:
                if file.endswith(".pdf"):
                    loader = PDFMinerLoader(os.path.join(root, file))
                    documents += loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_documents(texts, embeddings)
        return db
    except Exception as e:
        st.error(f"Error during embedding creation: {e}")
        return None

# Create a pipeline for the model with adjustments to generate longer responses
@st.cache_resource
def llm_pipeline(_tokenizer, _base_model):
    pipe = pipeline(
        'text2text-generation',
        model=_base_model,
        tokenizer=_tokenizer,
        device=device.index if torch.cuda.is_available() else -1,  # Use GPU if available, else CPU
        max_length=512,  # Increased max length to generate longer responses
        min_length=100,  # Set a minimum length for the answer
        do_sample=True,
        temperature=0.7,  # Slightly higher temperature for more varied but controlled output
        top_p=0.9
    )
    return HuggingFacePipeline(pipeline=pipe)

# Create a QA chain using the model and database
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

# Display PDF in the app
@st.cache_data
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    st.title("Chat with Your PDF 📄")

    tokenizer, base_model = load_model()

    uploaded_file = st.file_uploader("Upload a PDF:", type="pdf")
    if uploaded_file:
        # Save uploaded file
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

            # Load QA chain
            qa = qa_chain(db, tokenizer, base_model)

            # Memory for storing conversation history
            if "conversation_history" not in st.session_state:
                st.session_state.conversation_history = []

            # User interaction
            user_query = st.text_input("Ask a question about the PDF:")
            if user_query:
                with st.spinner("Fetching response..."):
                    response = qa({"query": user_query})
                    answer = response['result']

                    # Store the question and answer
                    st.session_state.conversation_history.append({"question": user_query, "answer": answer})

                # Display all questions and answers
                st.markdown("### Q&A History")
                for qa_pair in st.session_state.conversation_history:
                    st.write(f"**Q:** {qa_pair['question']}")
                    st.write(f"**A:** {qa_pair['answer']}")
                    st.write("---")

        else:
            st.error("Failed to process the PDF. Please check logs for details.")

if __name__ == "__main__":
    main()















































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
