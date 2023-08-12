import string
import re
import time
import io
import PyPDF2
import streamlit as st
import openai
from typing import List, Tuple
from langchain.langchain import LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set page style
st.set_page_config(page_title="DataBot: Chat with your Data!", page_icon=":book:")

# Initialize LangChain OCR
lc = LangChain()

# Import OpenAI credentials from config.py
from config import api_key, deployment_name, api_base, api_version

# Define function to extract texts and page numbers from PDF pages
def extract_text_from_pdf(pdf_file: object) -> List[Tuple[str, str, int]]:
    text_and_page = []
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    for page_num in range(0, pdf_reader.getNumPages()):
        pdf_page = pdf_reader.getPage(page_num)
        page_text = pdf_page.extractText()
        
        lc.load_images_from_pdf(pdf_file)
        ocr_text = lc.ocr_text
        
        # Append OCR text to page
        if ocr_text:
            page_text += '\n\n' + ocr_text
        
        # Append text and page number to list
        text_and_page.append((pdf_file.name, page_text, page_num))
    
    return text_and_page

# Define function to preprocess text before chunking
def preprocess_text(text: str) -> str:
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\xff]', '', text)  # Remove non-printable characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra white spaces
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    return text

# Define function to split text into chunks
def split_text_to_chunks(text_and_page: List[Tuple[str, str, int]], chunk_size: int, chunk_overlap: int) -> List[Tuple[str, str, int]]:
    text_and_page_chunks = []
    for pdf_file, text, page_num in text_and_page:
        text = preprocess_text(text)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
        chunks = splitter.split_text(text)
        for chunk in chunks:
            if re.search(user_input, chunk, re.IGNORECASE):
                text_and_page_chunks.append((pdf_file, chunk, page_num))
    return text_and_page_chunks

# Define function to display loading spinner
@st.cache(allow_output_mutation=True)
def display_loading_spinner():
    with st.spinner(text="Processing PDF files..."):
        time.sleep(5)

# Define Streamlit app
if __name__ == '__main__':
    # Set up app title
    st.title("DataBot: Chat with your Data! :book:")
    
    # Upload PDF files
    uploaded_files = st.file_uploader("Choose a PDF file", type=["pdf"], accept_multiple_files=True, key="upload")
    
    # Show uploaded PDF files
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} PDF file(s).")
        
        # Get user input
        user_input = st.text_input("Ask a question about the uploaded PDF files:")
        
        # Process uploaded PDF files
        if uploaded_files and user_input:
            # Create loading spinner if processing is slow
            spinner = st.empty()
            with spinner:
                display_loading_spinner()
            
            try:
                # Query OpenAI for a response
                response = openai.Completion.create(
                    engine=deployment_name,
                    prompt=user_input,
                    max_tokens=5096,
                    n=1,
                    stop=None,
                    temperature=0.2,
                    api_key=api_key,
                    api_base=api_base,
                )
                answer = response.choices[0].text.strip()
                
                # Search for answers in PDF files
                pdf_answers = []
                for pdf_file, text, page_num in split_text_to_chunks(extract_text_from_pdf(uploaded_files[0]), 5000, 1000):
                    if re.search(user_input, text, re.IGNORECASE):
                        pdf_answers.append((pdf_file, text, page_num))
                
                # Show chatbot reply with file and page information for each answer
                answer_with_pdf_answers = answer + "\n\n"
                if len(pdf_answers) > 0:
                    answer_with_pdf_answers += "Here are the answers I found in the uploaded PDF file(s):\n\n"
                    for pdf_file, text, page_num in pdf_answers:
                        answer_with_pdf_answers += f"- File: {pdf_file}, Page: {page_num + 1}\nText: {text}\n\n"
                else:
                    answer_with_pdf_answers += "Sorry, I could not find any answers in the uploaded PDF file(s)."
                
                st.text_area("Chatbot: ", answer_with_pdf_answers)
            
            except:
                st.text_area("Chatbot: ", "Sorry, I couldn't understand your question.")
            
            # Store chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            # Add user input to chat history
            if user_input:
                st.session_state.chat_history.append({"user_input": user_input})
            
            # Add chatbot reply to chat history
            if answer:
                st.session_state.chat_history.append({"chatbot_reply": answer})
            
            # Show chat history
            st.write("Chat History:")
            for item in st.session_state.chat_history:
                if "user_input" in item:
                    st.text_area("User: ", item["user_input"])
                elif "chatbot_reply" in item:
                    st.text_area("Chatbot: ", item["chatbot_reply"])
