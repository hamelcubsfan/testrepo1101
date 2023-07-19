import streamlit as st
from langchain.chains import StuffDocumentsChain
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# Create Streamlit interface
st.title("Personalized Outreach Generator")

api_key = st.text_input("Enter your OpenAI API Key")
source_url = st.text_input("Enter the source URL (website or blog post)")

# Function to scrape data from a website
def pull_from_website(url):
    try:
        response = requests.get(url)
    except:
        st.write("Whoops, error")
        return
    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove unnecessary tags
    for tag in soup.find_all(['nav', 'footer', 'aside', 'header', 'style', 'script']):
        tag.decompose()

    text = soup.get_text()
    text = md(text)
    return text

map_prompt = """Below is a section of a website about {prospect}

Write a concise summary about {prospect}. If the information is not about {prospect}, exclude it from your summary.

{text}

% CONCISE SUMMARY:"""

if st.button("Generate"):
    if api_key and source_url:
        # Scrape data from the website
        scraped_data = pull_from_website(source_url)

        # Initialize the necessary classes
        lang_model = OpenAI(openai_api_key=api_key)
        text_splitter = RecursiveCharacterTextSplitter()
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "prospect"])
        map_llm_chain = LLMChain(llm=lang_model, prompt=map_prompt_template)

        # Prepare the data
        documents = text_splitter.create_documents([scraped_data])

        # Initialize and run StuffDocumentsChain
        summarize_chain = StuffDocumentsChain(llm_chain=map_llm_chain, document_variable_name="text")
        summary = summarize_chain.run({"input_documents": documents, "prospect": "Prospect Name"})
        
        st.write(summary)
    else:
        st.write("Please provide both the API key and source URL.")
