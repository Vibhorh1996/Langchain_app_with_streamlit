import sys
import openai
import streamlit as st
from typing import NoReturn
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from streamlit_extras.add_vertical_space import add_vertical_space
from utils import extract_text_from_pdf, split_text_to_chunks
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


st.set_page_config(page_title='Ask the Doc App')
st.title('📚💡Ask the Doc App')
st.header("PDF Parser for Question Answering")


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def get_response(db:object, query:str, openai_api_key:str) -> str:

    # Create retriever interface
    retriever = db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 2}
                )

    # create a chain to answer questions
    chain = RetrievalQA.from_chain_type(
                    llm=OpenAI(openai_api_key=openai_api_key),
                    chain_type="stuff",
                    retriever=retriever
            )
    with st.spinner('Generating response...'):
        response = chain.run(query)
    return response


def main() -> NoReturn:

    pdf_file = st.file_uploader(label='Upload PDF file', type='pdf', accept_multiple_files=True)

    if pdf_file:

        with st.form('myform'):

            query = st.text_input(label="Ask a question about document")
            openai_api_key = st.text_input('Enter OpenAI API Key', type='password')
            submitted = st.form_submit_button('Submit')

            if submitted:

                combined_texts = extract_text_from_pdf(pdf_file)
                chunks = split_text_to_chunks(combined_texts, chunk_size=1000, chunk_overlap=100)

                try:
                    # Create a vectorstore from texts using Chroma
                    db = Chroma.from_texts(
                            chunks,
                            OpenAIEmbeddings(openai_api_key=openai_api_key)
                        )

                    add_vertical_space(2)
                    response = get_response(db, query, openai_api_key)
                except openai.error.AuthenticationError:
                    st.info('Incorrect API key provided', icon="⚠️")
                except openai.error.Timeout as e:
                    st.info(f'OpenAI API request timed out: {e}', icon="⚠️")
                except openai.error.RateLimitError as e:
                    st.info(f'OpenAI API request exceeded rate limit: {e}', icon="⚠️")
                except openai.error.PermissionError as e:
                    st.info(f"OpenAI API request was not permitted: {e}", icon="⚠️")
                except Exception as e:
                    st.info(f"Error Occured: {e}", icon="⚠️")

        try:
            st.success(response,  icon="ℹ️")
        except UnboundLocalError:
            pass


if __name__ == '__main__':
    main()
