import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# Create Streamlit interface
st.title("Personalized Outreach Generator")

api_key = st.text_input("Enter your OpenAI API Key")
source_url = st.text_input("Enter the source URL (LinkedIn profile, tweets, or blog posts)")

if st.button("Generate"):
    if api_key and source_url:
        # Initialize the necessary classes
        lang_model = OpenAI(openai_api_key=api_key)
        url_loader = UnstructuredURLLoader(urls=[source_url])
        text_splitter = RecursiveCharacterTextSplitter()

        map_prompt = "{content}\nSummarize:"
        combine_prompt = "{summaries}\nGenerate an outreach email based on the above:"

        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=['content'])
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=['summaries'])

        summarize_chain = load_summarize_chain(llm=lang_model)

        # Load and prepare the data
        documents = url_loader.load()
        documents = [doc.page_content for doc in documents]  # Extract the text from the Document objects
        documents = text_splitter.create_documents(documents)

        # Generate the personalized outreach email
        email = summarize_chain.execute(documents, lang_model, map_prompt_template, combine_prompt_template)

        st.write(email)
    else:
        st.write("Please provide both the API key and source URL.")
