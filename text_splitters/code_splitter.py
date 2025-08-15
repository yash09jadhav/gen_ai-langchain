from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def get_details(self):
        return f"Name: {self.name}, Age: {self.age}"
class Course:
    def __init__(self, title, duration):
        self.title = title
        self.duration = duration
    def get_details(self):
        return f"Title: {self.title}, Duration: {self.duration} hours"
"""
splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=200,
    chunk_overlap=0,
    language=Language.PYTHON
)
chunks = splitter.split_text(text)

print(chunks[0])
print("---------------")
print(chunks[1])