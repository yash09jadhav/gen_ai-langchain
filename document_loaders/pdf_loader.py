from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"C:\Users\2437874\Downloads\langchain\document_loaders\resume_yash_jadhav.pdf")

docs = loader.load()

print(docs[0].page_content)
print(docs[0].metadata)