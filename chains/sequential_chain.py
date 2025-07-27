from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

prompt1 = PromptTemplate(
    template= "generate a detailed report on {topic}",
    input_variables= ["topic"]
)

prompt2 = PromptTemplate(
    template= "generate a 5 point summary from {text}",
    input_variables=["text"]
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=2048
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic" : "best career in software"})

print(result)