import streamlit as st
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from langchain.document_loaders import Document

def main():
    st.title("LinkedIn Profile Summarizer")

    # Streamlit text input for LinkedIn profile URL
    linkedin_url = st.text_input("Please input the LinkedIn profile URL you want to summarize:")
    
    # Streamlit text input for OpenAI API key
    OPENAI_API_KEY = st.text_input("Please input your OpenAI API key:")

    # Function to scrape a website
    def pull_from_website(url):
        try:
            response = requests.get(url)
        except:
            print ("Whoops, error")
            return

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract only the section containing the profile
        profile_section = soup.find('section', {'id': 'profile'})

        if profile_section is None:
            return None

        text = profile_section.get_text()
        text = md(text)

        return text

    if linkedin_url:
        st.write(f"Fetching data from {linkedin_url}")
        fetched_data = pull_from_website(linkedin_url)
        
        if fetched_data is None:
            st.write("Failed to fetch data from the website.")
        else:
            st.write("Fetched data successfully. Now splitting into documents.")
            # Split the fetched data into documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=2000)
            docs = text_splitter.create_documents([fetched_data])

            if len(docs) == 0:
                st.write("Failed to split the fetched data into documents.")
            else:
                st.write("Data split into documents successfully. Now generating summary.")
                # Generate the summary
                llm = ChatOpenAI(temperature=.25, model_name='gpt-3.5-turbo-16k', api_key=OPENAI_API_KEY)
                chain = load_summarize_chain(llm)

                output = chain({"input_documents": docs})

                st.write("Generated summary successfully.")
                st.write(output['output_text'])

if __name__ == "__main__":
    main()
