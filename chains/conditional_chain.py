from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash"
)

parser1 = StrOutputParser()

class Feedback(BaseModel):
    sentiment : Literal["positive", "negative"]

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="classify the given feedback as positive or negative {feedback} {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

'''
result = classifier_chain.invoke({"feedback" : "this is a good smartphone"}).sentiment
print(result)
'''
prompt2 = PromptTemplate(
    template="give a response to this positive feedback {feedback}",
    input_variables=['feedback'],
)
prompt3 = PromptTemplate(
    template="give a response to this negative feedback {feedback}",
    input_variables=['feedback'],
)

branch_chain = RunnableBranch(
    (lambda x : x.sentiment == 'positive',  prompt2 | model | parser1),
    (lambda x : x.sentiment == 'negative',  prompt3 | model | parser1),
    RunnableLambda (lambda x: "could not find sentimnt")
)

chain = classifier_chain | branch_chain

print(chain.invoke({"feedback" : "this is a terrible phone"}))

chain.get_graph().print_ascii()
