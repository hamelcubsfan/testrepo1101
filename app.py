import streamlit as st
from langchain import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, Document
from langchain.schema import PromptTemplate
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# Create Streamlit interface
st.title("Personalized Outreach Generator")

# Replace the placeholders with Streamlit text input functions
api_key = st.text_input("Enter your OpenAI API Key")
source_url = st.text_input("Enter the source URL (website or blog post)")
candidate_name = st.text_input("Enter the candidate's name")

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

map_prompt = """Below is a section of a website about {candidate}

Write a concise summary about {candidate}. If the information is not about {candidate}, exclude it from your summary.

{text}

% CONCISE SUMMARY:"""

combine_prompt = """
You are a helpful AI bot that aids a user in summarizing information.
You will be given a list of summaries about {candidate}.

Please consolidate the summaries and return a unified, coherent summary

% SUMMARIES
{text}
"""

map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "candidate"])
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text", "candidate"])

if st.button("Generate"):
    if api_key and source_url and candidate_name:
        # Scrape data from the website
        scraped_data = pull_from_website(source_url)
        st.write("Scraped Data:", scraped_data)

        # Initialize the necessary classes
        lang_model = ChatOpenAI(openai_api_key=api_key, model_name='gpt-3.5-turbo', temperature=.25)

        # Prepare the data
        documents = RecursiveCharacterTextSplitter().create_documents([scraped_data])

        # Initialize and run StuffDocumentsChain
        summarize_chain = load_summarize_chain(lang_model,
                                               chain_type="map_reduce",
                                               map_prompt=map_prompt_template,
                                               combine_prompt=combine_prompt_template)
        
        summaries = summarize_chain.run({"input_documents": documents, "candidate": candidate_name})

        st.write(summaries)
    else:
        st.write("Please provide all necessary information.")
