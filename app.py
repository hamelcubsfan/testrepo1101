import streamlit as st
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate

# This function is a placeholder. You need to replace it with actual function to fetch profile HTML.
def fetch_profile_html(url):
    return "<html></html>"

def html_to_markdown(html):
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()
    return md(text)

def split_text(text, chunk_size=20000, chunk_overlap=2000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.create_documents([text])

def create_summarize_chain(llm, map_prompt_template, combine_prompt_template):
    return load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_prompt_template, combine_prompt=combine_prompt_template)

def run_summarize_chain(chain, docs, persons_name):
    output = chain({"input_documents": docs, "persons_name": persons_name})
    return output['output_text']

def main():
    st.title("LinkedIn Profile Summarizer")
    url = st.text_input("Enter LinkedIn profile URL")
    openai_key = st.text_input("Enter OpenAI API Key")

    if st.button("Summarize"):
        html = fetch_profile_html(url)
        markdown = html_to_markdown(html)
        docs = split_text(markdown)

        map_prompt = """You are a helpful AI bot that aids a user in research. Below is information about a person named {persons_name}. Information will include details about their professional background, skills, and experience. Your goal is to generate a summary of {persons_name}'s LinkedIn profile. Use specifics from the research when possible.
                        % START OF INFORMATION ABOUT {persons_name}:
                        {text}
                        % END OF INFORMATION ABOUT {persons_name}:
                        Please provide a concise summary of the professional profile based on the information above.
                        YOUR RESPONSE:"""
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "persons_name"])

        combine_prompt = """You are a helpful AI bot that aids a user in research. You will be given a list of summaries about {persons_name}'s professional profile. Please consolidate the summaries and return a final, comprehensive summary.
                            % SUMMARY
                            {text}
                            """
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text", "persons_name"])

        llm = ChatOpenAI(temperature=.25, model_name='gpt-3.5-turbo-16k', api_key=openai_key)
        chain = create_summarize_chain(llm, map_prompt_template, combine_prompt_template)
        summary = run_summarize_chain(chain, docs, "LinkedIn User")  # Replace "LinkedIn User" with actual name if available

        st.write(summary)

if __name__ == "__main__":
    main()
