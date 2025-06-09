from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate


# --------------------------------------------------------------------------------------------------------------- #
# Basic Info model and prompts
class BasicInfo(BaseModel):
    name: str = Field(description="name of the candidate")
    bio: str = Field(description="Bio, profile, introduction or summary of the candidate")
    job_title: str = Field(description="current or latest job title of the candidate")


basic_info_parser = PydanticOutputParser(pydantic_object=BasicInfo)

basic_details_prompt = PromptTemplate(
    template="{resume}\n{format_instructions}\n",
    input_variables=["resume"],
    partial_variables={"format_instructions": basic_info_parser.get_format_instructions()},
)


fallback_basic_info_prompt = PromptTemplate(
    template="What is the {query}?\nRESUME:\n{resume}\nANSWER:\n",
    input_variables=["query", "resume"]
)


# --------------------------------------------------------------------------------------------------------------- #
# Skills model and prompts
class Skills(BaseModel):
    skills: List[str] = Field(description="list of skills, programming languages, IT tools, software skills")


skills_parser = PydanticOutputParser(pydantic_object=Skills)

skills_template = """
Extract the technical skills, programming languages, IT tools, and software skills from this resume.
Only extract answers from the resume, do not make up answers.
RESUME:\n{resume}\n{format_instructions}\n
"""

skills_prompt = PromptTemplate(
    template=skills_template,
    input_variables=["resume"],
    partial_variables={"format_instructions": skills_parser.get_format_instructions()},
)

fallback_skills_prompt = PromptTemplate(
    template="""What are the skills in this resume?\nRESUME:\n{resume}\n
        Answer with a comma separated list.""",
    input_variables=["resume"]
)


# --------------------------------------------------------------------------------------------------------------- #
# Education model and prompts
class Education(BaseModel):
    """Education qualification"""
    qualification: str = Field(description="university or high-school education qualification or degree")
    establishment: Optional[str] = Field(description="establishment where the qualification was obtained")
    year: Optional[str] = Field(description="year when the qualification was obtained")


# Prompt Template to extract education degrees in a structured output
fallback_education_prompt = PromptTemplate(
    template="""
    You are a profession in finding Education degree, your job is to tell what are the university degrees mentioned in the Resume/CV
    Use the template to format the answer. Only use the resume to answer, do not make up answers. Provided answer must have these words: "University" or "College".
    If your answer don't have that word, exclude it.
    If there is no education mentioned in the resume, just answer with 'None'

    The template is provided to you as below, you can answer as much as you want, number of degrees is not constrained:

    "Qualification, Name of establishment, Year \n"
    "Qualification, Name of establishment, Year \n"

    DO NOT return answer that does not contain any name of establishment.
    The Qualification should be an answer that focused on the major, for example: "Bachelor in Artificial Intelligent"
    
    RESUME:\n{resume}\n
    ANSWER:
    """,
    input_variables=["resume"],
)