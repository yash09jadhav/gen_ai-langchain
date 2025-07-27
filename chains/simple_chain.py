from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

prompt = PromptTemplate(
    template = "generate 5 facts about {topic}",
    input_variables={"topic"}
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=2048
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic":"virat kohli vs sachin tendulkar who is better batsman"})

print(result)

# to visualize chains
chain.get_graph().print_ascii()