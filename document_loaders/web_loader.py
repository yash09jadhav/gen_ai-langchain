from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader('https://www.youtube.com/watch?v=bL92ALSZ2Cg&t=641s')

docs = loader.load()

print(docs)