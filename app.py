import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains import StuffDocumentsChain
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# Create Streamlit interface
st.title("Personalized Outreach Generator")

api_key = st.text_input("Enter your OpenAI API Key")
source_url = st.text_input("Enter the source URL (LinkedIn profile, tweets, or blog posts)")

map_prompt = """Below is a section of a website about {prospect}

Write a concise summary about {prospect}. If the information is not about {prospect}, exclude it from your summary.

{text}

% CONCISE SUMMARY:

% TONE
- Don't use any emojis or hashtags.
- Respond in the tone of Bill Gates
"""

if st.button("Generate"):
    if api_key and source_url:
        # Initialize the necessary classes
        lang_model = OpenAI(openai_api_key=api_key)
        url_loader = UnstructuredURLLoader(urls=[source_url])
        text_splitter = RecursiveCharacterTextSplitter()
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "prospect"])
        map_llm_chain = LLMChain(llm=lang_model, prompt=map_prompt_template)
        summarize_chain = StuffDocumentsChain(llm_chain=map_llm_chain)

        # Load and prepare the data
        documents = url_loader.load()

        # Extract the page_content from each Document object
        texts = [doc.page_content for doc in documents]

        # Split the texts into chunks
        documents = text_splitter.create_documents(texts)

        # Generate the personalized outreach email
        email = summarize_chain.run({"input_documents": documents, "prospect": "Prospect Name"})
        
        st.write(email)
    else:
        st.write("Please provide both the API key and source URL.")
