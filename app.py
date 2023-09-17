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
    BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")
    reader = BeautifulSoupWebReader()
    docs = reader.load_data(urls)

    # Initialize the prompt helper
    max_input_size = 4096
    num_output = 256
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

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


st.header("Chat with a website 💬 📚")
website = st.text_input("Enter a website URL")

with st.spinner("Loading website..."):
    index = load_from_website(website)

    # Get the user input
    question = st.text_input("Enter a question")

    if st.button("Ask"):
        # Get the answer from the index
        answer = index.as_query_engine().query(question)
        st.write(answer)
