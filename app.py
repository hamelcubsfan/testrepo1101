import os
import requests
import streamlit as st
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Scrapping function
def pull_from_website(url):
    try:
        response = requests.get(url)
    except:
        print ("Whoops, error")
        return
    
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    text = md(text)
     
    return text

# Streamlit main function
def main():
    st.title("LinkedIn Profile Summarizer")
    url = st.text_input("Enter LinkedIn profile URL")
    OPENAI_API_KEY = st.text_input("Enter OpenAI API Key", type="password")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    if st.button("Generate Summary"):
        website_data = pull_from_website(url)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=2000)
        docs = text_splitter.create_documents([website_data])

        map_prompt = """You are a helpful AI bot that aids a user in research.
        Below is information about a person.
        Your goal is to generate a brief summary of the person's professional background and expertise.

        % START OF INFORMATION:
        {text}
        % END OF INFORMATION:

        Please summarize the above information into a brief professional bio.

        YOUR RESPONSE:"""
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

        combine_prompt = """
        You are a helpful AI bot that aids a user in research.
        You will be given a list of potential summaries.

        Please consolidate the summaries into a single coherent bio

        % SUMMARIES
        {text}
        """
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

        llm = ChatOpenAI(temperature=.25, model_name='gpt-3.5-turbo')

        chain = load_summarize_chain(llm,
                                     chain_type="map_reduce",
                                     map_prompt=map_prompt_template,
                                     combine_prompt=combine_prompt_template)

        output = chain({"input_documents": docs})
        st.write(output['output_text'])

if __name__ == "__main__":
    main()
