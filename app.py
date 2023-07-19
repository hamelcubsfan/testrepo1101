from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# Function to scrape data from a website
def pull_from_website(url):
    try:
        response = requests.get(url)
    except:
        print("Whoops, error")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
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

api_key = YOUR_OPENAI_API_KEY
source_url = YOUR_SOURCE_URL
candidate_name = YOUR_CANDIDATE_NAME

# Scrape data from the website
scraped_data = pull_from_website(source_url)
print("Scraped Data:", scraped_data)

# Initialize the necessary classes
lang_model = OpenAI(openai_api_key=api_key, model_name='gpt-3.5-turbo-16k', temperature=.25)
map_llm_chain = LLMChain(llm=lang_model, prompt=map_prompt_template)
combine_llm_chain = LLMChain(llm=lang_model, prompt=combine_prompt_template)

# Prepare the data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=2000)
documents = text_splitter.create_documents([scraped_data])

# Run the map stage
summaries = map_llm_chain.run({"input_documents": documents, "candidate": candidate_name})

# Convert summaries to a list of Document objects
summaries = [Document(page_content=summary) for summary in summaries]

# Run the combine stage
consolidated_summary = combine_llm_chain.run({"input_documents": summaries, "candidate": candidate_name})

print(consolidated_summary)
