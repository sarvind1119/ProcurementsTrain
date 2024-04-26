import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
#from htmlTemplates import css, bot_template, user_template
import requests
from PyPDF2 import PdfReader
import io

def get_pdf_text_from_url(pdf_urls):
    text = ""
    for pdf_url in pdf_urls:
        # Fetch the PDF file from the URL
        response = requests.get(pdf_url)
        response.raise_for_status()  # Ensure the request was successful

        # Read the PDF file from the binary response content
        with io.BytesIO(response.content) as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() if page.extract_text() else ''
    return text

# Load environment variables and configure API
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Predefined PDF files (assume they are stored in a directory accessible by the script)
# pdf_files = [Manual for Procurement of Consultancy & Other Services_0.pdf",
#             Manual for Procurement of Works_0.pdf",
#             Procurement Manual of Goods.pdf",
#             Compilation of amendments in GFRs, 2017 upto 31.07.2023_1.pdf"]


pdf_files = ["Compilation of amendments in GFRs, 2017 upto 31.07.2023_1.pdf"]

def get_pdf_text(pdf_files):
    text = ""
    for pdf_path in pdf_files:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ''
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
provided context just say, 'answer is not available in the context', don't provide the wrong answer

Context:
{context}?

Question: 
{question}

Answer:
"""

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)

    # Printing response and providing a link to the documents in bold
    response_text = f"{response['output_text']}\n\n**In order to download the relevant documents please visit: [Download Documents](https://shorturl.at/dDZ14)**"
    st.write("Reply: ", response_text)

def main():
    st.set_page_config(page_title="Procurement of goods, works and consultancy services assistant",
                       page_icon=":books:")
    st.header("Procurement Rules GPT")
    
    # Define the URL of the PDF file
    pdf_urls = ["https://dsel.education.gov.in/sites/default/files/guidelines/NGIFEIE_dosel.pdf"]
    
    # Fetch and extract text from the PDF file at the URL
    raw_text = get_pdf_text_from_url(pdf_urls)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    
    user_question = st.text_input("Ask a Question on procurement rules and guidelines")
    if user_question:
        user_input(user_question)

    st.sidebar.markdown("""
        This custom GPT, named Procurement Rules Assistant, is designed to analyze and interpret procurement-related documents and rules, focusing on the documents provided. It concentrates on identifying key information, summarizing guidelines, and highlighting deviations from standard procurement practices outlined in the available manuals for the procurement of works, goods, and consultancy services.

        Currently it contains only the document linked from the provided URL. This document contains instructions for the preparation of detailed estimates of expenditure from the Consolidated Fund of India. It provides guidance on various aspects of budgeting, including the preparation of estimates, classification of accounts, and submission of reports. The document also includes forms for recording and reporting financial transactions, such as the Liability Register and the Statement of Liabilities.

        Procurement Rules documents will be uploaded soon.... 
    """)

if __name__ == "__main__":
    main()

