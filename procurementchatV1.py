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

# Load environment variables and configure API
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Predefined PDF files (assume they are stored in a directory accessible by the script)
pdf_files = ["C:/Users/HP/Downloads/procurement/Manual for Procurement of Consultancy & Other Services_0.pdf",
            "C:/Users/HP/Downloads/procurement/Manual for Procurement of Works_0.pdf",
            "C:/Users/HP/Downloads/procurement/Procurement Manual of Goods.pdf",
            "C:/Users/HP/Downloads/procurement/Compilation of amendments in GFRs, 2017 upto 31.07.2023_1.pdf"]

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
    #st.write(css, unsafe_allow_html=True)
        # Sidebar description

    raw_text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    
    user_question = st.text_input("Ask a Question on procurement rules and guidelines")
    st.sidebar.markdown("""
        **ProcurementRules_Training GPT Overview:**
        This custom GPT, named Procurement Rules Assistant, is designed to analyze and interpret procurement-related documents and rules within the context of provided files. It focuses on identifying key information, summarizing guidelines, and highlighting deviations from standard procurement practices as outlined in various manuals and rules for the procurement of works, goods, and consultancy services.

        **Main Functions:**
        - **Document Analysis**: Reads and extracts important information from procurement manuals and rules, such as the "Manual for Procurement of Works," "Procurement Manual of Goods," and the "Manual for Procurement of Consultancy & Other Services."
        - **Rule Interpretation**: Interprets financial and procurement rules from documents like the "Compilation of amendments in GFRs, 2017 up to 31.07.2023," ensuring that procurement practices align with the latest guidelines and financial regulations.
        - **Deviation Identification**: Identifies deviations from standard procedures in tender documents or proposals, helping to ensure compliance and mitigate risks associated with non-adherence to established norms.

        This setup is tailored to assist in the management and oversight of procurement processes within government departments, ensuring adherence to financial propriety, efficient resource use, and transparency.
    """)
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
