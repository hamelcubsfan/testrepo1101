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
    except:
        st.error("There was an error in accessing the URL.")
        return None
    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove unnecessary tags
    for tag in soup.find_all(['nav', 'footer', 'aside', 'header', 'style', 'script']):
        tag.decompose()

    text = soup.get_text()

    # Convert HTML to Markdown
    h = html2text.HTML2Text()
    text = h.handle(text)

    return text

map_prompt = """We have a section of a website that contains some information about {candidate}. 
We need you to write a concise summary about the {candidate}-related content only. 
If you find information that is not about {candidate}, please exclude it from your summary.

Here is the section:

{text}

Now, please provide a concise summary about {candidate} based on the above section."""

combine_prompt = """
Now, you have a list of summaries about {candidate}. Your task is to consolidate these summaries and create a unified, coherent summary about {candidate}. 

Here are the summaries:

{text}

Now, please provide a consolidated summary about {candidate}."""

map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "candidate"])
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text", "candidate"])

if st.button("Generate"):
    if not api_key or not source_url or not candidate_name:
        st.error("Please provide all necessary information.")
        return

    # Scrape data from the website
    scraped_data = pull_from_website(source_url)
    
    if scraped_data is None:
        st.error("There was an error in scraping the webpage. Please check the URL and try again.")
        return

    if candidate_name not in scraped_data:
        st.error("The scraped webpage does not seem to contain information about the candidate.")
        return

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
