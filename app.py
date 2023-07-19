import streamlit as st
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Define the prompt templates
document_prompt_template = PromptTemplate(
    input_variables=["page_content"],
    template="{page_content}"
)

map_prompt_template = PromptTemplate.from_template(
    "Map this content: {context}"
)

combine_prompt_template = PromptTemplate.from_template(
    "Combine this content: {context}"
)

# Initialize the language model
llm = OpenAI(temperature=0.5)

# Create two LLM chains: one for mapping, one for combining
map_llm_chain = LLMChain(
    llm=llm,
    prompt=map_prompt_template
)

combine_llm_chain = LLMChain(
    llm=llm,
    prompt=combine_prompt_template
)

# The name of the variable in which to put the documents
document_variable_name = "context"

# Create a StuffDocumentsChain instance
summarize_chain = StuffDocumentsChain(
    llm_chain=map_llm_chain,  # or combine_llm_chain depending on your use case
    document_prompt=document_prompt_template,
    document_variable_name=document_variable_name
)

# Streamlit app
st.title('Summarize Documents')

# Get the input from the user
input_docs = st.text_area("Input Documents", value='', height=None, max_chars=None, key=None)

# If there is any input, process it
if input_docs:
    # Split the input into separate documents
    input_docs_list = input_docs.split('\n')

    # Use the StuffDocumentsChain to summarize the documents
    result = summarize_chain({"input_documents": input_docs_list})

    # Display the result
    st.text_area("Result", value=result, height=None, max_chars=None, key=None)
