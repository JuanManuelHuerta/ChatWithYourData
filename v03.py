import yaml
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file




with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


openai.api_key  = config['KEYS']['OPENAI_API_KEY']




from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'


from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


    

embedding = OpenAIEmbeddings(openai_api_key=openai.api_key)
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)


print(vectordb._collection.count())

texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
]



smalldb = Chroma.from_texts(texts, embedding=embedding)

print ("SIMILARITY SEARCH ================================================")
question = "Tell me about all-white mushrooms with large fruiting bodies"
print("QUERY:",question)
print(smalldb.similarity_search(question, k=2))

#
#  MAXIMUM MARGINAL RELEVANCE (MMR): Increases diversity in responses
#


print ("MAXIMUM MARGIN RELEVANCE ================================================")
question = "what did they say about matlab?"
print("QUERY:",question)

#docs_ss = vectordb.similarity_search(question,k=3)
docs_mmr = vectordb.max_marginal_relevance_search(question,k=3)

print("MMR A1:",docs_mmr[0].page_content[:100])
print("MMR A2",docs_mmr[1].page_content[:100])

##
##  SELF QUERY: Uses and LLM (OpenAI) to set up description of the filters that the query provides
##


print ("SELF QUERY ================================================")
print("QUERY:",question)


metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The lecture the chunk is from, should be one of `docs/cs229_lectures/MachineLearning-Lecture01.pdf`, `docs/cs229_lectures/MachineLearning-Lecture02.pdf`, or `docs/cs229_lectures/MachineLearning-Lecture03.pdf`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The page from the lecture",
        type="integer",
    ),
]
document_content_description = "Lecture notes"
llm = OpenAI(temperature=0,openai_api_key=openai.api_key)
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)
question = "what did they say about regression in the third lecture?"
docs = retriever.get_relevant_documents(question)


for d in docs:
    print("SelfQueryRetriever: ",d.metadata)



##
## CONTEXTUAL COMPRESSION: EXTRACTS ONLY THE RELEVANT BITS
##


print ("COMPRESSED RETRIEVER ================================================")
question = "what did they say about matlab?"
print("QUERY:",question)
    
# Wrap our vectorstore
llm = OpenAI(temperature=0,openai_api_key=openai.api_key)
compressor = LLMChainExtractor.from_llm(llm)

#compression_retriever = ContextualCompressionRetriever(
#    base_compressor=compressor,
#    base_retriever=vectordb.as_retriever()
#)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(search_type = "mmr")
)

compressed_docs = compression_retriever.get_relevant_documents(question)
pretty_print_docs(compressed_docs)
