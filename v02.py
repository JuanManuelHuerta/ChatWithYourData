'''
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter


markdown_document = """# Title\n\n \
## Chapter 1\n\n \
Hi this is Jim\n\n Hi this is Joe\n\n \
### Section \n\n \
Hi this is Lance \n\n 
## Chapter 2\n\n \
Hi this is Molly"""


headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]




loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)

docs = text_splitter.split_documents(pages)

print(len(docs))
print(len(pages))


'''

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
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture02.pdf"),
    PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture03.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())




    # Define the Text Splitter 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

#Create a split of the document using the text splitter
splits = text_splitter.split_documents(docs)

print(len(splits))



embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)


sentence1 = "i like dogs"
sentence2 = "i like canines"
sentence3 = "the weather is ugly outside"

embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)


print(np.dot(embedding1, embedding2))



persist_directory = 'docs/chroma/'


#DONT FORGET TO DO THIS
#!rm -rf ./docs/chroma  # remove old database files if any




vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)



print(vectordb._collection.count())


question = "is there an email i can ask for help"


docs = vectordb.similarity_search(question,k=3) # k is number of documents


len(docs)


print(docs[0].page_content)


vectordb.persist()

