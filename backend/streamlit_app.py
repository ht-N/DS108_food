import streamlit as st
import pandas as pd
import os
import json
import tempfile
from pathlib import Path
import torch
from langchain_community.document_loaders import CSVLoader
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from CV_parser.parser import ResumeManager, get_resume_content

# Set page configuration
st.set_page_config(
    page_title="CV Job Matcher",
    page_icon="📄",
    layout="wide"
)

@st.cache_resource
def load_embedding_model():
    """Load the embedding model once and cache it"""
    # EMBEDDING_MODEL_NAME = "dangvantuan/vietnamese-document-embedding"
    EMBEDDING_MODEL_NAME = "thenlper/gte-large"
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
    )
    return embedding_model

@st.cache_resource
def create_job_vectorstore():
    """Create and cache the job vectorstore from CSV data"""
    csv_file_path = fr"data\job_details_it.csv"

    # Function to clean text by replacing newlines with spaces
    def clean_text(text):
        if text is None:
            return ""
        return text.replace('\n', ' ').strip()

    # Load CSV file directly as pandas DataFrame
    import pandas as pd
    df = pd.read_csv(csv_file_path)

    # Create a document for each row
    docs_list = []
    for index, row in df.iterrows():
        # Create metadata from DataFrame columns
        metadata = {
            'source': csv_file_path,
            'row': index,
            'field': clean_text(str(row.get('Field', ''))),
            'experience': clean_text(str(row.get('Experience', ''))),
            'location': clean_text(str(row.get('Location', ''))),
            'company_size': clean_text(str(row.get('Company Size', ''))),
            'salary': clean_text(str(row.get('Salary', ''))),
            'job_requirements': clean_text(str(row.get('Job Requirements', ''))),
            'url': clean_text(str(row.get('URL', ''))),
        }
        
        # Extract additional numeric fields if they exist
        if 'Experience_year' in row:
            metadata['experience_year'] = str(row['Experience_year'])
        if 'minSalary' in row:
            metadata['min_salary'] = str(row['minSalary'])
        if 'maxSalary' in row:
            metadata['max_salary'] = str(row['maxSalary'])
        
        # Create page content with only Field, Location, and Job Requirements
        page_content = f"Field: {metadata['field']} Location: {metadata['location']} Job Requirements: {metadata['job_requirements']}"
        
        # Create document with clean metadata
        new_doc = Document(
            page_content=clean_text(page_content),
            metadata=metadata
        )
        docs_list.append(new_doc)
    
    # Create vector store
    embedding_model = load_embedding_model()
    vectorstore = FAISS.from_documents(documents=docs_list, embedding=embedding_model)
    
    return vectorstore

def process_cv(file, model_name="deepseek-r1-distill-llama-70b"):
    """Process the uploaded CV file"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
        tmp_file.write(file.getvalue())
        file_path = tmp_file.name
        
    try:
        # Use your existing ResumeManager to process the CV
        resume_manager = ResumeManager(file_path, model_name)
        resume_manager.process_file()
        return resume_manager.output
    finally:
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

def find_matching_jobs(cv_data, vectorstore, top_k=5):
    """Find jobs matching the CV profile"""
    # Create a query string from CV data
    query = f"Field: {cv_data.get('job_title', '')} Job Requirements: {' '.join(cv_data.get('skills', []))}"
    
    # Search for similar jobs
    embedding_model = load_embedding_model()
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    
    return results

# Main Streamlit app
def main():
    st.title("CV Job Matcher")
    
    # Sidebar
    with st.sidebar:
        st.header("Upload CV")
        uploaded_file = st.file_uploader("Choose a CV file", type=["pdf", "docx"])
        model_option = st.selectbox(
            "Select LLM model",
            ["deepseek-r1-distill-llama-70b", "llama-3.1-8b-instant"]
        )
        process_button = st.button("Find Matching Jobs")
    
    # Main content area
    if not process_button or uploaded_file is None:
        st.info("Upload your CV and click 'Find Matching Jobs' to see job recommendations.")
        
        # Load and preprocess job data in the background
        with st.spinner("Preparing job database..."):
            job_vectorstore = create_job_vectorstore()
        st.success("Job database loaded and ready!")
        
    elif process_button and uploaded_file is not None:
        # Process CV
        with st.spinner("Processing CV..."):
            cv_data = process_cv(uploaded_file, model_option)
        
        # Display CV data
        st.header("Your CV Summary")
        st.subheader(f"{cv_data.get('candidate_name', 'Unknown')} - {cv_data.get('job_title', 'Unknown')}")
        st.write(f"**Bio:** {cv_data.get('bio', 'Not available')}")
        
        st.subheader("Skills")
        st.write(", ".join(cv_data.get('skills', ['Not available'])))
        
        st.subheader("Education")
        for edu in cv_data.get('education', []):
            if isinstance(edu, dict):
                st.write(f"• {edu.get('qualification', '')} from {edu.get('establishment', '')} ({edu.get('year', '')})")
            else:
                st.write(f"• {edu}")
        
        # Find matching jobs
        with st.spinner("Finding matching jobs..."):
            job_vectorstore = create_job_vectorstore()
            matching_jobs = find_matching_jobs(cv_data, job_vectorstore)
        
        # Display matching jobs
        st.header("Top Job Matches")
        
        # Sort matching jobs by score (highest first)
        sorted_jobs = sorted(matching_jobs, key=lambda x: x[1], reverse=True)
        
        # Display sorted jobs
        for i, (job, score) in enumerate(sorted_jobs, 1):
            st.subheader(f"#{i}: {job.metadata.get('field')} - Score: {score:.2f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Field:** {job.metadata.get('field', 'Not specified')}")
                st.write(f"**Location:** {job.metadata.get('location', 'Not specified')}")
                st.write(f"**Experience Required:** {job.metadata.get('experience', 'Not specified')}")
            
            with col2:
                st.write(f"**Salary:** {job.metadata.get('salary', 'Not specified')}")
                st.write(f"**Company Size:** {job.metadata.get('company_size', 'Not specified')}")
                if job.metadata.get('url'):
                    st.markdown(f"**[Apply Here]({job.metadata.get('url')})**")
            
            st.write(f"**Job Requirements:**")
            st.write(job.page_content.split("Job Requirements: ")[-1] if "Job Requirements: " in job.page_content else "Not specified")
            st.divider()

if __name__ == "__main__":
    main()