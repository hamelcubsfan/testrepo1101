import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains import StuffDocumentsChain
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# Create Streamlit interface
st.title("Personalized Outreach Generator")

api_key = st.text_input("Enter your OpenAI API Key")
source_url = st.text_input("Enter the source URL (LinkedIn profile, tweets, or blog posts)")

map_prompt = """Below is a section of a website about {prospect}

Write a concise summary about {prospect}. If the information is not about {prospect}, exclude it from your summary.

{text}

% CONCISE SUMMARY:"""

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

company_information = """
* RapidRoad helps product teams build product faster
* We have a platform that allows product teams to talk more, exchange ideas, and listen to more customers
* Automated project tracking: RapidRoad could use machine learning algorithms to automatically track project progress, identify potential bottlenecks, and suggest ways to optimize workflows. This could help product teams stay on track and deliver faster results.
* Collaboration tools: RapidRoad could offer built-in collaboration tools, such as shared task lists, real-time messaging, and team calendars. This would make it easier for teams to communicate and work together, even if they are in different locations or time zones.
* Agile methodology support: RapidRoad could be specifically designed to support agile development methodologies, such as Scrum or Kanban. This could include features like sprint planning, backlog management, and burndown charts, which would help teams stay organized and focused on their goals.
"""

if st.button("Generate"):
    if api_key and source_url:
        # Initialize the necessary classes
        lang_model = OpenAI(temperature=0.5)
        url_loader = UnstructuredURLLoader(urls=[source_url])
        text_splitter = RecursiveCharacterTextSplitter()
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text", "prospect"])
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["sales_rep", "company", "prospect", "text", "company_information"])
        map_llm_chain = LLMChain(llm=lang_model, prompt=map_prompt_template)
        combine_llm_chain = LLMChain(llm=lang_model, prompt=combine_prompt_template)
        document_variable_name = "context" 
        summarize_chain = StuffDocumentsChain(
            llm_chain=map_llm_chain,  # or combine_llm_chain depending on your use case
            document_prompt=map_prompt_template,
            document_variable_name=document_variable_name
        )

        # Load and prepare the data
        documents = url_loader.load()

        # Extract the page_content from each Document object
        texts = [doc.page_content for doc in documents]

        # Split the texts into chunks
        documents = text_splitter.create_documents(texts)

        # Generate the personalized outreach email
        email = summarize_chain.run({"input_documents": documents, "company": "Your Company", "company_information": company_information, "sales_rep": "Your Name", "prospect": "Prospect Name"})
        
        st.write(email)
    else:
        st.write("Please provide both the API key and source URL.")
