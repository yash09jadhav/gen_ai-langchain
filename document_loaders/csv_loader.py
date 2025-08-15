from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path = r"C:\Users\LENOVO\OneDrive\Desktop\ds\datasets\membership.csv",
)

docs = loader.load()
print(docs[0])
print(len(docs)) # <-- these are number of rows in the CSV file