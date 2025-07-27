from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash"
)

model2 = HuggingFaceEndpoint(
    repo_id ="HuggingFaceH4/zephyr-7b-beta",
    task="conversational",
)

prompt1 = PromptTemplate(
    template = "generate notes on topic {text}",
    input_variables = ['text']
)

prompt2 = PromptTemplate(
    template = "generate questions on topic {text}",
    input_variables = ['text']
)

prompt3 = PromptTemplate(
    template = "merge both the notes -> {notes} and questions -> {questions}",
    input_variables = ['notes', 'questions']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'questions': prompt2 | model2 | parser,
})

merge = prompt3 | model1 | parser

chain = parallel_chain | merge

text = "Jobs, businesses, and startups each offer distinct career paths shaped by risk, reward, and personal ambition. A job typically provides financial stability, structured work hours, and a clear hierarchy, making it a safe and reliable choice for many. In contrast, running a business offers greater control and independence, allowing individuals to build something of their own, but it also demands continuous effort, capital, and the ability to manage uncertainty. Startups, often considered the most dynamic of the three, focus on innovation and rapid scaling, usually fueled by external funding and a high tolerance for failure. While jobs reward consistency and performance within set roles, businesses reward strategic thinking and leadership over time. Startups, however, thrive on creativity, adaptability, and solving unsolved problems, often disrupting traditional models. The level of risk also differs—jobs carry the least, businesses moderate, and startups the highest. Financial returns from a job are generally fixed, whereas businesses and startups can scale exponentially if successful. The work-life balance varies as well—jobs often allow for more predictable routines, while startups and businesses can blur boundaries between personal and professional life. Ultimately, the choice depends on an individual’s goals, risk appetite, and willingness to take initiative beyond the security of a paycheck"

result = chain.invoke({'text' : text})
print(result)