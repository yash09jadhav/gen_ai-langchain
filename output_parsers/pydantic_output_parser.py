from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

class Person(BaseModel):
    name:str = Field(description = "name of the person"),
    age: int = Field(description = "age"),
    city: str = Field(description="city of person")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = "generate name, age, city of a fictional {place} person {format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction":parser.get_format_instructions()}
)

prompt = template.invoke({"place":"indian"})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)