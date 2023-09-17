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

st.header("Chat with a website 💬 📚")
st.text_input("Enter a website URL", value="https://www.streamlit.io/")

# Initialize the chat message history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about Streamlit's open-source Python library!"
        }
    ]


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


# Load the Streamlit docs and index them
@st.cache_resource(show_spinner=False)
def load_from_dir():
    with st.spinner(text="Loading and indexing the Streamlit docs...hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(
                model="gpt-3.5-turbo",
                temperature=0.5,
                system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical and based on facts – do not hallucinate features."
            )
        )
        index = VectorStoreIndex.from_documents(
            docs, service_context=service_context)
        return index


index = load_from_website()

# Chat with the Streamlit docs
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"):
    # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}

            # Add response to message history
            st.session_state.messages.append(message)
