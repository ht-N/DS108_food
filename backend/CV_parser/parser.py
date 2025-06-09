import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import docx
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_groq import ChatGroq
from groq import Groq

from pydantic_models_prompts import Education
from pydantic_models_prompts import (
    basic_details_prompt, fallback_basic_info_prompt,
    skills_prompt, fallback_skills_prompt,
    fallback_education_prompt
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)

logger.addHandler(stream_handler)

load_dotenv()

# Simplified output template
output_template = {
    'candidate_name': '',
    'job_title': '',
    'bio': '',
    'skills': [],
    'education': []
}


class ResumeManager:
    def __init__(self, resume_f, model_name, extension=None):
        self.output = output_template.copy()
        self.resume = get_resume_content(resume_f, extension)
        self.model_name = model_name
        # Use ChatGroq from langchain_groq instead of OpenAI
        self.model = ChatGroq(model=model_name, request_timeout=10, max_retries=1)
        # Direct Groq client for non-langchain calls
        self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def process_file(self):
        self.extract_basic_info()
        self.extract_skills()
        self.extract_education()

    def extract_pydantic(self, target):
        start = time.time()
        chain = create_extraction_chain_pydantic(target, self.model)

        result = chain.invoke({"input": self.resume})
        end = time.time()
        seconds = end - start
        return result, seconds

    def query_model(self, query, json_mode=True):
        start = time.time()

        if json_mode:
            completion = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user",
                           "content": query}],
                response_format={'type': 'json_object'},
                timeout=8,
            )

        else:
            completion = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user",
                           "content": query}],
                timeout=8,
            )

        end = time.time()
        seconds = end - start
        result = completion.choices[0].message.content
        return result, seconds

    def extract_basic_info(self):
        query = basic_details_prompt.format(resume=self.resume)
        output, seconds = self.query_model(query)
        output = json.loads(output)
        logger.debug(f"# Basic Info Extract:\n{output}")
        logger.info(f"# Basic Info Extraction took {seconds} seconds")

        try:
            self.output['candidate_name'] = output['name']
        except KeyError:
            query = fallback_basic_info_prompt.format(query='name', resume=self.resume)
            name, _ = self.query_model(query, json_mode=False)
            self.output['candidate_name'] = name

        try:
            self.output['job_title'] = output['job_title']
        except KeyError:
            query = fallback_basic_info_prompt.format(query='current or last job title', resume=self.resume)
            title, _ = self.query_model(query, json_mode=False)
            self.output['job_title'] = title

        try:
            self.output['bio'] = output['bio']
        except KeyError:
            query = fallback_basic_info_prompt.format(query='bio or profile summary', resume=self.resume)
            bio, _ = self.query_model(query, json_mode=False)
            self.output['bio'] = bio

    def extract_skills(self):
        try:
            query = skills_prompt.format(resume=self.resume)
            output, seconds = self.query_model(query)
            output = json.loads(output)
            logger.debug(f"# Skills Extract:\n{output}")
            logger.info(f"# Skills Extraction took {seconds} seconds")
            self.output['skills'] = output['skills']

        except Exception as e:
            logger.warning(f"Skills extraction error: {e}")
            query = fallback_skills_prompt.format(resume=self.resume)
            output, seconds = self.query_model(query, json_mode=False)
            logger.debug(f"# Skills Extract:\n{output}")
            logger.info(f"# Skills Extraction took {seconds} seconds")
            self.output['skills'] = [skill.strip() for skill in output.split(',')]

    def extract_education(self):
        try:
            output, seconds = self.extract_pydantic(Education)
            logger.debug(f"# Education Extract:\n{output}")
            logger.info(f"# Education Extraction took {seconds} seconds")
            self.output['education'] = [json.loads(x.json().encode('utf-8')) for x in output]

        except Exception as e:
            logger.warning(f"Education extraction error: {e}")
            query = fallback_education_prompt.format(resume=self.resume)
            output, seconds = self.query_model(query, json_mode=False)
            logger.debug(f"# Education Extract:\n{output}")
            logger.info(f"# Education Extraction took {seconds} seconds")
            self.output['education'] = output


def get_resume_content(file_path, extension=None):
    if not extension:
        extension = os.path.splitext(file_path)[1]
    if extension == '.pdf':
        pdf_reader = PdfReader(file_path)
        content = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            for line in text.split('\n'):
                line = line.rstrip()
                if line:
                    content += line
                    content += '\n'
    elif extension in ['.docx', '.doc']:
        doc = docx.Document(file_path)
        content = ""
        for paragraph in doc.paragraphs:
            content += paragraph.text + "\n"

    else:
        sys.exit(f"Unsupported file type {extension}")
    return content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a Resume with Groq LLM models")
    parser.add_argument("file_path", help="Path to the resume, accepted types .pdf or .docx")
    parser.add_argument("--model_name", default='deepseek-r1-distill-llama-70b',
                        help="Name of the model, default to llama-3.1-8b-instant")

    args = parser.parse_args()
    logging.info(f"Processing {args.file_path}")

    resume_manager = ResumeManager(args.file_path, args.model_name)

    start_time = time.time()
    resume_manager.process_file()
    end_time = time.time()

    resume_name = Path(args.file_path).stem
    output_file_path = f"{resume_name}_output.json"
    with open(output_file_path, 'w') as file:
        json.dump(resume_manager.output, file, indent=2)

    print(json.dumps(resume_manager.output, indent=2))

    seconds = end_time - start_time
    m, s = divmod(seconds, 60)
    logger.info(f"Total time {int(m)} min {int(s)} seconds")