import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# Define your input variables
input_variables = ["content"]

# Define your prompt templates
map_prompt = "{content}\nSummarize:"
combine_prompt = "{summaries}\nGenerate an outreach email based on the above:"

# Initialize the PromptTemplate with the templates and input variables
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=input_variables)
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=input_variables)

# Create Streamlit interface
st.title("Personalized Outreach Generator")

api_key = st.text_input("Enter your OpenAI API Key", key="openai_key")
source_url = st.text_input("Enter the source URL (LinkedIn profile, tweets, or blog posts)", key="source_url")

if st.button("Generate"):
    if api_key and source_url:
        # Initialize OpenAI model
        lang_model = OpenAI(openai_api_key=api_key)

        # Initialize the necessary classes
        url_loader = UnstructuredURLLoader(urls=[source_url])
        text_splitter = RecursiveCharacterTextSplitter()
        summarize_chain = load_summarize_chain()

        # Load and prepare the data
        documents = url_loader.load()
        documents = text_splitter.split(documents)

        # Generate the personalized outreach email
        email = summarize_chain.execute(documents, lang_model, map_prompt_template, combine_prompt_template)

        st.write(email)
    else:
        st.write("Please provide both the API key and source URL.")
