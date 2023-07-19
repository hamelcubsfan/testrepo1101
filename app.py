# -*- coding: utf-8 -*-
import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# Create Streamlit interface
st.title("Personalized Outreach Generator")

api_key = st.text_input("Enter your OpenAI API Key", key='input_api_key')
source_url = st.text_input("Enter the source URL (LinkedIn profile, tweets, or blog posts)", key='input_source_url')

if st.button("Generate"):
    if api_key and source_url:
        # Initialize the OpenAI model
        # NOTE: This assumes that `OpenAI` takes an API key as an argument. You might need to revise this
        #       line according to the actual usage of the `OpenAI` class in the `langchain` library.
        lang_model = OpenAI(api_key)

        # Initialize the necessary classes
        url_loader = UnstructuredURLLoader()
        text_splitter = RecursiveCharacterTextSplitter()
        map_prompt = PromptTemplate("{content}\nSummarize:")
        combine_prompt = PromptTemplate("{summaries}\nGenerate an outreach email based on the above:")
        summarize_chain = load_summarize_chain()

        # Load and prepare the data
        documents = url_loader.load([source_url])
        documents = text_splitter.split(documents)

        # Generate the personalized outreach email
        email = summarize_chain.execute(documents, lang_model, map_prompt, combine_prompt)

        st.write(email)
    else:
        st.write("Please provide both the API key and source URL.")
