from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    google_api_key="your-google-api-key-here"
)

text_splitter = SemanticChunker(
    chunk_size=200,
    embeddings=embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)
load_dotenv()

text = """
# Project Name: Smart Student Tracker   
A simple Python-based project to manage and track student data, including their grades, age, and academic status.
## Features
- Add new students with relevant info
- View student details
- Check if a student is passing
- Easily extendable class-based design
## 🛠 Tech Stack
- Python 3.10+
- No external dependencies
## Getting Started
1. Clone the repo
   bash
   git clone https://github.com/your-username/student-tracker.git
   """
docs = text_splitter.create_documents([text])
print(docs[0])
print("---------------")
print(docs[1])