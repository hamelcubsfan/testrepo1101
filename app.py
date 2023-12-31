import streamlit as st
from langchain.chains import StuffDocumentsChain
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter, Document
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
import html2text

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
        response.raise_for_status()
    except requests.HTTPError as http_err:
        st.write(f"HTTP error occurred: {http_err}")
        return
    except Exception as err:
        st.write(f"Other error occurred: {err}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove unnecessary tags
    for tag in soup.find_all(['nav', 'footer', 'aside', 'header', 'style', 'script']):
        tag.decompose()

    # Scrape only the necessary sections
    user_info = soup.find('div', class_='user-info')
    repositories = soup.find('div', class_='repo-list')

    text = ""
    if user_info is not None:
        text += user_info.get_text()
    if repositories is not None:
        text += repositories.get_text()

    # Convert HTML to Markdown
    h = html2text.HTML2Text()
    text = h.handle(text)

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
        if scraped_data:
            st.write("Scraped Data:", scraped_data)

            # Initialize the necessary classes
            lang_model = OpenAI(openai_api_key=api_key, model_name='gpt-3.5-turbo-16k', temperature=.25)
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
            st.write("Failed to scrape the website.")
    else:
        st.write("Please provide all necessary information.")
