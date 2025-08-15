from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = r"C:\Users\2437874\Downloads\langchain\document_loaders\books",
    glob = "*.pdf",
    loader_cls = PyPDFLoader
)
docs = loader.load()
print(docs[0].page_content)
print(docs[0].metadata)
print(len(docs))  