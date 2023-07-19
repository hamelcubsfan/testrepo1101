import streamlit as st
from langchain.chains import StuffDocumentsChain
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# Create Streamlit interface
st.title("Personalized Outreach Generator")

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

map_prompt = """You are a helpful AI bot that aids a user in research.
Below is information about a person named {candidate}.
Information will include tweets, interview transcripts, and blog posts about {candidate}
Your goal is to generate interview questions that we can ask {candidate}
Use specifics from the research when possible

% START OF INFORMATION ABOUT {candidate}:
{text}
% END OF INFORMATION ABOUT {candidate}:

Please respond with list of a few interview questions based on the topics above

YOUR RESPONSE:"""

map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "candidate"])

combine_prompt = """
You are a helpful AI bot that aids a user in research.
You will be given a list of potential interview questions that we can ask {candidate}.

Please consolidate the questions and return a list

% INTERVIEW QUESTIONS
{text}
"""

combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text", "candidate"])

if st.button("Generate"):
    if api_key and source_url and candidate_name:
        # Scrape data from the website
        scraped_data = pull_from_website(source_url)
        st.write("Scraped Data:", scraped_data)

        # Initialize the necessary classes
        lang_model = OpenAI(openai_api_key=api_key, model_name='gpt-3.5-turbo-16k', temperature=.25)
        map_llm_chain = LLMChain(llm=lang_model, prompt=map_prompt_template)
        combine_llm_chain = LLMChain(llm=lang_model, prompt=combine_prompt_template)

        # Prepare the data
        documents = RecursiveCharacterTextSplitter().create_documents([scraped_data])

        # Initialize and run StuffDocumentsChain
        summarize_chain = StuffDocumentsChain(llm_chain=map_llm_chain, document_variable_name="text")
        questions = summarize_chain.run({"input_documents": documents, "candidate": candidate_name})

        summarize_chain = StuffDocumentsChain(llm_chain=combine_llm_chain, document_variable_name="text")
        consolidated_questions = summarize_chain.run({"input_documents": questions, "candidate": candidate_name})
        
        st.write(consolidated_questions)
    else:
        st.write("Please provide all necessary information.")
