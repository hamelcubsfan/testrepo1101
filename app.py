from langchain import OpenAI, PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.load import UnstructuredURLLoader
from langchain.docstore import Document
from langchain.chains import SimpleSequentialChain
from langchain.chains.mapreduce import MapReduceChain, CombineDocumentsChain
from langchain.prompts import PromptTemplate

# Parameters
api_key = "YOUR_API_KEY"
input_urls = ["https://www.example.com"]
prospect = "prospect"

# OpenAI
lang_model = OpenAI(openai_api_key=api_key)

# Prompt templates
input_variables = ["text", "prospect"]
map_prompt = "Write a concise summary about {prospect}. If the information is not about {prospect}, exclude it from your summary.\n\n{text}\n\n% CONCISE SUMMARY:"
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=input_variables)

combine_prompt = """Your goal is to write a personalized outbound email to {prospect}.\n\n% INFORMATION ABOUT {prospect}:\n{text}\n\n% YOUR RESPONSE:"""
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=input_variables)

# Loading the documents
url_loader = UnstructuredURLLoader(urls=input_urls)
documents = url_loader.load()

# Splitting the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20, length_function=len)
documents = [Document(page_content=text) for text in text_splitter.split_text(documents[0].page_content)]

# Creating the chain
map_reduce_chain = MapReduceChain(llm=lang_model, map_prompt=map_prompt_template)
combine_documents_chain = CombineDocumentsChain(llm=lang_model, combine_prompt=combine_prompt_template)
chain = SimpleSequentialChain(chains=[map_reduce_chain, combine_documents_chain], verbose=True)

# Running the chain
email = chain.run(documents)
