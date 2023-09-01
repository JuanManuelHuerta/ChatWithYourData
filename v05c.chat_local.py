import os
import openai
import sys
import yaml
import datetime


from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import GPT4All
#from gpt4all import GPT4All


## ToDo:
## 1. LocalEmbeddings
###        https://docs.gpt4all.io/index.html
###        https://python.langchain.com/docs/use_cases/question_answering/how_to/local_retrieval_qa
## 2. LocalLLm
##   https://python.langchain.com/docs/use_cases/question_answering/how_to/local_retrieval_qa
#    https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main

local = True

if local is True:
    model = GPT4All(model="orca-mini-3b.ggmlv3.q4_0.bin",max_tokens=2048)
else:
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    openai.api_key  = config['KEYS']['OPENAI_API_KEY']
    current_date = datetime.datetime.now().date()
    if current_date < datetime.date(2023, 9, 2):
        llm_name = "gpt-3.5-turbo-0301"
    else:
        llm_name = "gpt-3.5-turbo"
    print("LLM Name: ", llm_name)
    print(llm_name)
    model = ChatOpenAI(model_name=llm_name,openai_api_key=openai.api_key,  temperature=0)



def create_on_the_fly_KB():
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
        #chunk_size = 1500,                                                                                                                                                                                                                  
        chunk_size = 1500,
        chunk_overlap = 500,
        separators=["\n\n","\n","(?<=\. )"," ",""]
    )

    chunks = text_splitter.split_documents(docs)
    #chunks = text_splitter.split_text(docs)
    # Convert the chunks of text into embeddings to form a knowledge base
    #embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    embeddings=GPT4AllEmbeddings()
    knowledgeBase = FAISS.from_documents(chunks, embeddings)    
    return knowledgeBase


faiss_kb = create_on_the_fly_KB()





# This will initialize your database and retriever chain 
def load_db(chain_type, k):
    # load documents
    # define embedding
    #embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    # create vector database from data
    #db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    #db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    db=faiss_kb
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        #llm = ChatOpenAI(model_name=llm_name,openai_api_key=openai.api_key,  temperature=0),
        llm=model,
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa



import param

class cbfs(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query  = param.String("")
    db_response = param.List([])
    
    def __init__(self,  **params):
        super(cbfs, self).__init__( **params)
        self.panels = []
        #vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        self.qa = load_db("stuff", 4)

    
    def convchain(self, query):
        result = self.qa({"question": query, "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        self.db_query = result["generated_question"]
        self.db_response = result["source_documents"]
        self.answer = result['answer'] 
        print("\n\n")
        print('User: ',query)
        print('Chatbot: ',self.answer)
        
        print('\nReferences:')
        for a in self.db_response:
            print (" *** ",a)

        

    def clr_history(self,count=0):
        self.chat_history = []
        return 



cb = cbfs()
while True:
    query=input("\n\nEnter query: ")
    cb.convchain(query)
    

