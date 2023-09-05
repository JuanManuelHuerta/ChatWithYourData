import os
import openai
import sys
import yaml
import datetime
sys.path.append('../..')
from langchain.vectorstores import Chroma
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
import param



# LOAD config parameters
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
openai.api_key  = config['KEYS']['OPENAI_API_KEY']

# Load the LLM
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)

# Set the Chroma db directory
persist_directory = 'docs/chroma/'

embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
llm = ChatOpenAI(model_name=llm_name,openai_api_key=openai.api_key,  temperature=0)




# This will initialize your database and retriever chain 
def load_db(chain_type, k):
    # load documents
    # define embedding
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    # create vector database from data
    #db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm = ChatOpenAI(model_name=llm_name,openai_api_key=openai.api_key,  temperature=0),
        chain_type=chain_type, 
        retriever=retriever, 
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa




class cbfs(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query  = param.String("")
    db_response = param.List([])
    
    def __init__(self,  **params):
        super(cbfs, self).__init__( **params)
        self.panels = []
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
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
    

