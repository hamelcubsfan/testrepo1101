import streamlit as st
from langchain.chains import StuffDocumentsChain
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Create Streamlit interface
st.title("Personalized Outreach Generator")

api_key = st.text_input("Enter your OpenAI API Key")
about_section = st.text_input("Paste the 'About' section from the LinkedIn profile")
experience_section = st.text_input("Paste the 'Experience' section from the LinkedIn profile")
candidate_name = st.text_input("Enter the candidate's name")

map_prompt = """Below is some information about {candidate}

{about_section}

{experience_section}

% CONCISE SUMMARY:"""

if st.button("Generate"):
    if api_key and about_section and experience_section and candidate_name:
        # Initialize the necessary classes
        lang_model = OpenAI(openai_api_key=api_key)
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["about_section", "experience_section", "candidate"])
        map_llm_chain = LLMChain(llm=lang_model, prompt=map_prompt_template)

        # Run StuffDocumentsChain
        summarize_chain = StuffDocumentsChain(llm_chain=map_llm_chain, document_variable_name="text")
        summary = summarize_chain.run({"input_documents": [about_section, experience_section], "candidate": candidate_name})
        
        st.write(summary)
    else:
        st.write("Please provide all necessary information.")
