# -*- coding: utf-8 -*-

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

map_prompt = "Below is a section of a website about {prospect}\nWrite a concise summary about {prospect}. If the information is not about {prospect}, exclude it from your summary.\n{text}\n% CONCISE SUMMARY:"
combine_prompt = "Your goal is to write a personalized outbound email from {sales_rep}, a sales rep at {company} to {prospect}.\nA good email is personalized and combines information about the two companies on how they can help each other. Be sure to use value selling: A sales methodology that focuses on how your product or service will provide value to the customer instead of focusing on price or solution.\n% INFORMATION ABOUT {company}:\n{company_information}\n% INFORMATION ABOUT {prospect}:\n{text}\n% INCLUDE THE FOLLOWING PIECES IN YOUR RESPONSE:\n- Start the email with the sentence: 'We love that {prospect} helps teams...' then insert what they help teams do.\n- The sentence: 'We can help you do XYZ by ABC' Replace XYZ with what {prospect} does and ABC with what {company} does\n- A 1-2 sentence description about {company}, be brief\n- End your email with a call-to-action such as asking them to set up time to talk more\n% YOUR RESPONSE:"

if st.button("Generate"):
    if api_key and source_url:
        lang_model = OpenAI(openai_api_key=api_key)

        # Initialize the necessary classes
        url_loader = UnstructuredURLLoader(urls=[source_url])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=512, length_function=len)

        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["prospect", "text"])
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["sales_rep", "company", "prospect", "text", "company_information"])

        # Load and prepare the data
        documents = url_loader.load()
        documents = text_splitter.create_documents(documents)

        # Load the summarize chain
        summarize_chain = load_summarize_chain(llm=lang_model, chain_type="map_reduce", map_prompt=map_prompt_template, combine_prompt=combine_prompt_template, verbose=True)

        # Generate the personalized outreach email
        email = summarize_chain.run({"input_documents": documents, "company": "Your Company", "company_information": "Your Company Information", "sales_rep": "Your Name", "prospect": "Prospect Name"})

        st.write(email)
    else:
        st.write("Please provide both the API key and source URL.")
