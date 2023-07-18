pip install langchain
pip install openai


import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# Initialize the necessary classes
url_loader = UnstructuredURLLoader()
text_splitter = RecursiveCharacterTextSplitter()
map_prompt = PromptTemplate("{content}\nSummarize:")
combine_prompt = PromptTemplate("{summaries}\nGenerate an outreach email based on the above:")
summarize_chain = load_summarize_chain()

# Create Streamlit interface
st.title("Personalized Outreach Generator")

api_key = st.text_input("Enter your OpenAI API Key")
source_url = st.text_input("Enter the source URL (LinkedIn profile, tweets, or blog posts)")

if st.button("Generate"):
    if api_key and source_url:
        lang_model = OpenAI(api_key)
        
        # Load and prepare the data
        documents = url_loader.load([source_url])
        documents = text_splitter.split(documents)

        # Generate the personalized outreach email
        email = summarize_chain.execute(documents, lang_model, map_prompt, combine_prompt)
        
        st.write(email)
    else:
        st.write("Please provide both the API key and source URL.")
