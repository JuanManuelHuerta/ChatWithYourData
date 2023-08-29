import os
import openai
import sys
import yaml
import datetime
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate



## This should run after v12
#  To do look at v03.py


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = config['KEYS']['OPENAI_API_KEY']

current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)

persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
print("Accessed Chroma database with ",vectordb._collection.count()," documents")

# Just to check all is well:
#question = "What are major topics for this class?"


#docs = vectordb.similarity_search(question,k=3)
#len(docs)


### RETREIVAL QA CHAIN & ALL DOCUMENTS ARE STUFFED INTO THE CONTEXT


llm = ChatOpenAI(model_name=llm_name, temperature=0,openai_api_key=openai.api_key)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="refine"
)

##  SET THE PROMPT:
question = "What is money and who invented it?"
question = "Who sets the interest rates? and What happens when interest rates rise?"


print("\n\n_________________QA CHAIN with REFINE_________________________")

result = qa_chain({"query":  question})
print("RERIEVAL QA Query:",question)
print("RETRIEVAL QA result:", result["result"])



### PROMPT CHAIN

print("\n\n_______________ QA CHAIN with WITH PROMPT TEMPLATE____________________")

# Build prompt

template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)


result = qa_chain({"query": question})
print(result["result"])

print("\n REFERENCES:")
print("result documents")
print(result["source_documents"][0])

print("\n\n_________________QA CHAIN with MAP REDUCEE_________________________")


## REtrieval QA chain types


qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)
result = qa_chain_mr({"query": question})
print(result["result"])




'''
#If you wish to experiment on the LangChain plus platform:

#Go to langchain plus platform and sign up
##Create an API key from your account's settings

#Use this API key in the code below
#uncomment the code
#Note, the endpoint in the video differs from the one below. Use the one below.

import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = "..." # replace dots with your api key
qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="map_reduce"
)
result = qa_chain_mr({"query": question})
result["result"]
qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type="refine"
)
result = qa_chain_mr({"query": question})
print(result["result"])
'''


