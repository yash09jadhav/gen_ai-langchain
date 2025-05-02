from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_token"

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',# which model 
    task = 'text-generation'
)
model = ChatHuggingFace(llm=llm)

result = model.invoke('what is the capital of india ?')

print(result.content)