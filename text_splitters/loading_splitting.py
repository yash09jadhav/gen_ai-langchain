from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("C:\\Users\\2437874\\Downloads\\langchain\\text_splitters\\resume_yash_jadhav.pdf")
docs = loader.load()

splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=100,
    chunk_overlap=0,
)
result = splitter.split_documents(docs)

print("Page 1 Content:", result[0].page_content)
print("Page 1 Metadata:", result[0].metadata)
print("Page 2 Content:", result[1].page_content)
print("Page 2 Metadata:", result[1].metadata)
