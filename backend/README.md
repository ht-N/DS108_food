# RAG Job Finder System

A Retrieval-Augmented Generation (RAG) system that helps match your CV with suitable job listings. This application uses natural language processing and semantic search to find the most relevant job opportunities based on your resume.

## Overview

The RAG Job Finder system combines several key technologies:

1. **Web Crawler**: Automatically extracts job listings from TopCV website
2. **CV Parser**: Utilizes LLMs to extract structured information from your resume
3. **RAG Implementation**: Applies semantic search to match your skills and experience with relevant job opportunities

## How It Works

### Crawler from TopCV Website

The system includes a dedicated crawler that:
- Scrapes job listings from TopCV
- Extracts key information such as field, experience requirements, location, company size, salary, and detailed job requirements
- Creates a structured database of job opportunities

**Note: The current data I'm using is very limited, you can modify the crawler to crawl more jobs data.**

### CV Parser

The CV parser can process resume (PDF format) and extract:
- Basic information (name, job title, bio)
- Skills and technical expertise
- Educational background and qualifications

The parser utilizes large language models (like Llama, Deepseek from Groq) to accurately understand and extract information from your resume.

### RAG Implementation

The RAG (Retrieval-Augmented Generation) system:
1. Converts resume data and job listings into vector embeddings (using thenlper/gte-large from HuggingFace)
2. Uses semantic search to find the best matches between your profile and available jobs
3. Ranks job opportunities based on relevance to your experience and skills
4. Provides detailed information about each match to help you make informed decisions

## How to Use

### Clone the respository
```bash
git clone https://github.com/ht-N/Jobsearch_RAG.git
cd Jobsearch_RAG
```

### Install requirements

```bash
# First, you need to install a virtual env, you can use Anaconda (Miniconda)
conda create -n <your_vir_env_name> python=3.11
conda activate <your_vir_env_name>
```

```bash
# These are the required library, it will be in the requirements.txt folder:
pandas
numpy
torch
faiss-cpu
sentence-transformers
pdfplumber
streamlit
selenium
msedge-selenium-tools
webdriver-manager
requests
beautifulsoup4
lxml
```

```bash
pip install -r requirements.txt
```

### Setting up the API key
You need to set up the API key for the chatGroq to use the DeepSeek model for CV parser, you need to create a .env file and set up the API key in this format:
```bash
GROQ_API_KEY = "your_groq_api_key"
```

### Running the Application
1. Start the Streamlit application:
   ```
   streamlit run app.py
   ```
2. Upload your CV (PDF format)
3. View your CV summary and matching job recommendations

## Acknowledgement
Thanks to @Sajjad Amjad for the CV Parser!
- [Sajjad Amjad's Github](https://github.com/Sajjad-Amjad/Resume-Parser#)
- [His CV_Parser](https://github.com/Sajjad-Amjad/Resume-Parser#)

Also because of a cute person asked me for help, so this project existed.
