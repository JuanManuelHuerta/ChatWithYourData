import os
import openai
import sys
import yaml



with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


openai.api_key  = config['KEYS']['OPENAI_API_KEY']

import datetime
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)



from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)


print(vectordb._collection.count())


question = "What are major topics for this class?"
docs = vectordb.similarity_search(question,k=3)
len(docs)


### RETREIVAL QA CHAIN




from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0,openai_api_key=openai.api_key)


from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)
result = qa_chain({"query": question})
print(result["result"])




### PROMPT CHAIN



from langchain.prompts import PromptTemplate

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
question = "Is probability a class topic?"
result = qa_chain({"query": question})


print(result["result"])


print("result documents")


print(result["source_documents"][0])




## REtrieval QA chain types

print("Retrieval QA from chain_type")

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


