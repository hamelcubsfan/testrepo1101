import streamlit as st
from langchain.chains import StuffDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter, Document
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import re

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
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = md(text)
    return text

map_prompt = """Below is a section of a website about {candidate}

Based only on the text provided, write a concise summary about {candidate}. Do not invent or add any additional information.

{text}

% CONCISE SUMMARY:"""

combine_prompt = """
You are a helpful AI bot that aids a user in summarizing information.
You will be given a list of summaries about {candidate}.

Please consolidate the summaries and return a unified, coherent summary, ensuring to only use the information provided and not to fabricate any details.

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
        map_llm_chain = LLMChain(llm=lang_model, prompt=map_prompt_template)
        combine_llm_chain = LLMChain(llm=lang_model, prompt=combine_prompt_template)

        # Prepare the data
        documents = RecursiveCharacterTextSplitter().create_documents([scraped_data])

        # Initialize and run StuffDocumentsChain
        summarize_chain = StuffDocumentsChain(llm_chain=map_llm_chain, document_variable_name="text")
        summaries = summarize_chain.run({"input_documents": documents, "candidate": candidate_name})

        # Convert summaries to a list of Document objects
        summaries = [Document(page_content=summary) for summary in summaries]

        summarize_chain = StuffDocumentsChain(llm_chain=combine_llm_chain, document_variable_name="text")
        consolidated_summary = summarize_chain.run({"input_documents": summaries, "candidate": candidate_name})
        
        st.write(consolidated_summary)
    else:
        st.write("Please provide all necessary information.")
