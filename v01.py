from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")

#Load the document by calling loader.load()
pages = loader.load()

print(len(pages))
print(pages[0].page_content[0:500])

print(pages[0].metadata)
# {'source': 'docs/cs229_lectures/MachineLearning-Lecture01.pdf', 'page': 0}

