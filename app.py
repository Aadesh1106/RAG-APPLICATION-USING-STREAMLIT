import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Validate if file is a proper PDF
def is_valid_pdf(file):
    file.seek(0)
    header = file.read(4)
    file.seek(0)
    return header == b'%PDF'

# Extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf_file in pdf_docs:
        if not is_valid_pdf(pdf_file):
            st.error(f"{pdf_file.name} is not a valid PDF.")
            continue
        try:
            pdf_file.seek(0)
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.warning(f"Failed to read {pdf_file.name}: {str(e)}")
    return text

# Split long text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Create and save FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Load the QA chain with Gemini
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, just say "answer is not available in the context".
    Don't provide incorrect information.

    Context:
    {context}

    Question:  
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",  # Updated model name
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Handle user question
def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply:", response["output_text"])
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}. Please make sure you've uploaded and processed PDF files first.")

# Main app logic
def main():
    st.set_page_config("Chat With Multiple PDF")
    st.header("Chat with MULTIPLE PDFs using Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF Files and click Submit & Process", accept_multiple_files=True, type=['pdf'])
        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.error("No valid text found in the uploaded PDF(s).")
                    return
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

# Run app
if __name__ == "__main__":
    main()