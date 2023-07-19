import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

# Create Streamlit interface
st.title("Personalized Outreach Generator")

api_key = st.text_input("Enter your OpenAI API Key")
source_url = st.text_input("Enter the source URL (LinkedIn profile, tweets, or blog posts)")

# Define the map and combine prompts
map_prompt = "Below is a section of a website about {prospect}\nWrite a concise summary about {prospect}. If the information is not about {prospect}, exclude it from your summary.\n{text}\n% CONCISE SUMMARY:"
combine_prompt = """
Your goal is to write a personalized outbound email from {sales_rep}, a sales rep at {company} to {prospect}.
A good email is personalized and combines information about the two companies on how they can help each other.
Be sure to use value selling: A sales methodology that focuses on how your product or service will provide value to the customer instead of focusing on price or solution.
% INFORMATION ABOUT {company}:
{company_information}
% INFORMATION ABOUT {prospect}:
{text}
% INCLUDE THE FOLLOWING PIECES IN YOUR RESPONSE:
- Start the email with the sentence: "We love that {prospect} helps teams..." then insert what they help teams do.
- The sentence: "We can help you do XYZ by ABC" Replace XYZ with what {prospect} does and ABC with what {company} does 
- A 1-2 sentence description about {company}, be brief
- End your email with a call-to-action such as asking them to set up time to talk more
% YOUR RESPONSE:
"""
input_variables = ["prospect", "text", "sales_rep", "company", "company_information"]
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=input_variables)
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=input_variables)

if st.button("Generate"):
    if api_key and source_url:
        # Initialize the necessary classes
        lang_model = OpenAI(openai_api_key=api_key)

        # Load and prepare the data
        url_loader = UnstructuredURLLoader(urls=[source_url])
        documents = url_loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split(documents)

        # Load the summarization chain
        summarize_chain = load_summarize_chain(lang_model, chain_type="map_reduce")

        # Generate the personalized outreach email
        email = summarize_chain.run(documents, map_prompt_template, combine_prompt_template)

        st.write(email)
    else:
        st.write("Please provide both the API key and source URL.")
