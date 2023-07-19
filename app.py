import streamlit as st
from langchain.load import load_chain
from langchain.llms.openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# Setup the langchain library
chain = load_chain("map_reduce_llm")
llm = ChatOpenAI(temperature=.25, model_name='gpt-3.5-turbo-16k')

# Get the LinkedIn url from the user
linkedin_url = st.text_input("Enter the LinkedIn URL:")

# Get the OpenAI API Key from the user
OPENAI_API_KEY = st.text_input("Enter the OpenAI API Key:", type="password")

if st.button('Generate'):
    if linkedin_url and OPENAI_API_KEY:
        llm.api_key = OPENAI_API_KEY
        
        # Fetch and parse the LinkedIn page
        response = requests.get(linkedin_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()
        text = md(text)
        
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=2000)
        docs = text_splitter.create_documents([text])
        
        # Use the langchain library to generate the output
        output = chain({"input_documents": docs})
        st.write(output)
    else:
        st.write("Please enter both the LinkedIn URL and the OpenAI API Key.")
