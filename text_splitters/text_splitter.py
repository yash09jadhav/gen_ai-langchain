text = """the Tesla Cybertruck is a battery-electric full-size pickup truck manufactured by Tesla, 
Inc. since 2023. It was first unveiled as a prototype in November 2019, featuring a distinctive 
angular design composed of flat, unpainted stainless steel body panels, drawing comparisons to 
low-polygon computer models.

Originally scheduled for production in late 2021, the vehicle faced multiple delays before entering 
limited production at Gigafactory Texas in November 2023, with initial customer deliveries occurring
later that month. As of 2025, three variants are available: a tri-motor all-wheel drive (AWD) model
marketed as the "Cyberbeast", a dual-motor AWD model, and a single-motor rear-wheel drive (RWD) 
"Long Range" model. EPA range estimates vary by configuration, from 320 to 350 miles (515 to 565 km).
The Cybertruck is sold exclusively in the United States and Canada. The Cybertruck has been 
criticized for its production quality and safety concerns while its sales have been described as
 disappointing."""

from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=100,
    chunk_overlap=0,
)
print(splitter.split_text(text))