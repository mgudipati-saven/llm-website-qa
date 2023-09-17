import streamlit as st
import openai

from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
from llama_index import SimpleDirectoryReader
from llama_index import download_loader
from llama_index import BeautifulSoupWebReader, PromptHelper

from scrape_utils import scrape

# Set the OpenAI API key
openai.api_key = st.secrets.openai_key


@st.cache_resource(show_spinner=False)
def load_from_website(url):
    # Get all the URLs from the website
    urls = scrape(url)

    # Load the website data
    BeautifulSoupWebReader = download_loader(
        "BeautifulSoupWebReader", custom_path="./llama_index")
    reader = BeautifulSoupWebReader()
    docs = reader.load_data(urls)

    # Initialize the prompt helper
    max_input_size = 4096
    num_output = 256
    chunk_overlap_ratio = 0.1
    prompt_helper = PromptHelper(
        max_input_size, num_output, chunk_overlap_ratio)

    # Index the website data
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(
            model="gpt-3.5-turbo",
            temperature=0.5,
        ),
        prompt_helper=prompt_helper
    )
    index = VectorStoreIndex.from_documents(
        docs, service_context=service_context)

    return index


st.header("Question an online documentation 💬 📚")

# Load the website data
website = st.text_input("Enter the website URL")
if st.button("Load documents"):
    with st.spinner(text="Loading..."):
        index = load_from_website(website)

# Ask a question
question = st.text_input("Ask a question")
if st.button("Ask"):
    # Get the answer from the index
    answer = index.as_query_engine().query(question)
    st.write(answer)
