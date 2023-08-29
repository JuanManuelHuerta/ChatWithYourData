
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


import yaml



with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


openai.api_key  = config['KEYS']['OPENAI_API_KEY']


# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("docs/macro/Livro Macro.pdf"),
    PyPDFLoader("docs/macro/Macroeconomics_IntroReview.pdf"),
    PyPDFLoader("docs/macro/n.-gregory-mankiw-macroeconomics-7th-edition-2009.pdf")

]
docs = []
for loader in loaders:
    docs.extend(loader.load())




    # Define the Text Splitter 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 0,
    separators=["\n\n","\n","(?<=\. )"," ",""]
)

#Create a split of the document using the text splitter
splits = text_splitter.split_documents(docs)

print(len(splits))



embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)


#DONT FORGET TO DO THIS
#!rm -rf ./docs/chroma  # remove old database files if any


persist_directory = 'docs/chroma/'



vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)



print(vectordb._collection.count())


question = "What is the GDP?"


docs = vectordb.similarity_search(question,k=3) # k is number of documents

print("Q:",question)
len(docs)
print(docs[0].page_content)

##

question = "what is inflation?"

docs =vectordb.similarity_search(question,k=5)



print("Q:",question)
len(docs)
print(docs[0].page_content)





vectordb.persist()

