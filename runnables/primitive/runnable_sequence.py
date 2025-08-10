from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv 
from langchain.schema.runnable import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template = 'write a joke about {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template = 'explain the joke about {text}',
    input_variables=['text']
)
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

print(chain.invoke({'topic':'AI'}))