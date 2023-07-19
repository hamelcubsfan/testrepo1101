import streamlit as st
from langchain import OpenAI, RecursiveCharacterTextSplitter, UnstructuredURLLoader, Document
from langchain.prompts import PromptTemplate
from langchain.chains import StuffDocumentsChain

# Ask the user for their OpenAI key and the LinkedIn URL
openai_key = st.text_input("Please enter your OpenAI key:")
linkedin_url = st.text_input("Please enter the LinkedIn URL you want to process:")

# Proceed with the rest of the code only if both inputs are provided
if openai_key and linkedin_url:
    # Initialize the language model with the provided key
    lang_model = OpenAI(openai_api_key=openai_key)

    # Define URLs to load with the provided URL
    urls = [linkedin_url]

    # Load the contents of the URLs into documents
    url_loader = UnstructuredURLLoader(urls=urls)
    documents = url_loader.load()

    # Define how the documents should be split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20, length_function=len)
    documents = text_splitter.create_documents(documents)

    # Define the map prompt
    map_prompt = """Below is a section of a website about {prospect}

Write a concise summary about {prospect}. If the information is not about {prospect}, exclude it from your summary.

{text}

% CONCISE SUMMARY:"""
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "prospect"])

    # Define the combine prompt
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
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["sales_rep", "company", "prospect", "text", "company_information"])

    # Create the chain to summarize the documents
    summarize_chain = StuffDocumentsChain(llm=lang_model, map_prompt=map_prompt_template, combine_prompt=combine_prompt_template)

    # Run the chain on the documents
    email = summarize_chain.run({"input_documents": documents})

    # Display the result
    st.write(email)
