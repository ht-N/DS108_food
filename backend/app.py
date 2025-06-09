from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os
from pathlib import Path
import re
import math
from collections import Counter
import unicodedata

# Pydantic models for job data
class JobModel(BaseModel):
    id: int
    jobTitle: str
    companyName: str
    field: str
    experience: str
    location: str
    companySize: str
    salary: str
    jobDescription: str
    jobRequirements: str
    companyLogoUrl: Optional[str] = None
    url: Optional[str] = None
    experienceYear: Optional[int] = None
    minSalary: Optional[float] = None
    maxSalary: Optional[float] = None

class JobListResponse(BaseModel):
    jobs: List[JobModel]
    total: int
    page: int
    pageSize: int

# Initialize FastAPI app
app = FastAPI(
    title="Job Board API",
    description="API for serving job listings from CSV data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store job data
jobs_data = []

class BM25:
    """BM25 algorithm implementation for Vietnamese text search"""
    
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        
        # Preprocess and index the corpus
        self._preprocess_corpus()
        self._calculate_idf()
    
    def _preprocess_text(self, text):
        """Preprocess Vietnamese text"""
        if not text:
            return []
        
        # Convert to lowercase and normalize unicode
        text = unicodedata.normalize('NFC', str(text).lower())
        
        # Remove special characters, keep Vietnamese characters and numbers
        text = re.sub(r'[^\w\s\u00C0-\u017F\u1EA0-\u1EF9]', ' ', text)
        
        # Split into words and filter out empty strings
        words = [word.strip() for word in text.split() if word.strip() and len(word.strip()) > 1]
        
        return words
    
    def _preprocess_corpus(self):
        """Preprocess the entire corpus"""
        for doc in self.corpus:
            # Combine job title, description, requirements, and company name for search
            combined_text = f"{doc.get('jobTitle', '')} {doc.get('jobDescription', '')} {doc.get('jobRequirements', '')} {doc.get('companyName', '')} {doc.get('field', '')}"
            words = self._preprocess_text(combined_text)
            
            self.doc_freqs.append(Counter(words))
            self.doc_len.append(len(words))
        
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
    
    def _calculate_idf(self):
        """Calculate IDF for all terms"""
        n = len(self.corpus)
        all_words = set()
        
        for doc_freq in self.doc_freqs:
            all_words.update(doc_freq.keys())
        
        for word in all_words:
            containing_docs = sum(1 for doc_freq in self.doc_freqs if word in doc_freq)
            self.idf[word] = math.log((n - containing_docs + 0.5) / (containing_docs + 0.5) + 1)
    
    def get_scores(self, query):
        """Get BM25 scores for a query"""
        query_words = self._preprocess_text(query)
        scores = []
        
        for i, doc_freq in enumerate(self.doc_freqs):
            score = 0
            doc_len = self.doc_len[i]
            
            for word in query_words:
                if word in doc_freq:
                    freq = doc_freq[word]
                    idf = self.idf.get(word, 0)
                    
                    # BM25 formula
                    score += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            
            scores.append(score)
        
        return scores
    
    def search(self, query, top_k=None):
        """Search and return ranked results"""
        scores = self.get_scores(query)
        
        # Create list of (index, score) and sort by score
        scored_docs = [(i, score) for i, score in enumerate(scores) if score > 0]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            scored_docs = scored_docs[:top_k]
        
        return scored_docs

# Global BM25 instance
bm25_index = None

def load_jobs_from_csv():
    """Load jobs from CSV file and convert to proper format"""
    global jobs_data, bm25_index
    
    # Path to your CSV file
    csv_path = Path("crawler/job_details_all_processed.csv")
    
    if not csv_path.exists():
        # Try alternative paths
        alternative_paths = [
            "data/job_details_all.csv",
            "job_details_all.csv",
            "../crawler/job_details_all.csv"
        ]
        
        for alt_path in alternative_paths:
            if Path(alt_path).exists():
                csv_path = Path(alt_path)
                break
        else:
            raise FileNotFoundError(f"CSV file not found. Tried: {csv_path}, {alternative_paths}")
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Convert to list of dictionaries
        jobs_data = []
        for index, row in df.iterrows():
            # Handle experience year
            exp_year = row.get("Experience_year", "")
            try:
                experience_year = int(float(exp_year)) if exp_year and str(exp_year).lower() != 'nan' else None
            except:
                experience_year = None
            
            # Handle salary values
            min_sal = row.get("minSalary", "")
            max_sal = row.get("maxSalary", "")
            try:
                min_salary = float(min_sal) if min_sal and str(min_sal).lower() not in ['nan', 'tbd'] else None
            except:
                min_salary = None
            try:
                max_salary = float(max_sal) if max_sal and str(max_sal).lower() not in ['nan', 'tbd'] else None
            except:
                max_salary = None
            
            job = {
                "id": index + 1,
                "jobTitle": str(row.get("Job Title", "")).strip(),
                "companyName": str(row.get("Company Name", "")).strip(),
                "field": str(row.get("Field", "")).strip(),
                "experience": str(row.get("Experience", "")).strip(),
                "location": str(row.get("Location", "")).strip(),
                "companySize": str(row.get("Company Size", "")).strip(),
                "salary": str(row.get("Salary", "")).strip(),
                "jobDescription": str(row.get("Job Description", "")).strip(),
                "jobRequirements": str(row.get("Job Requirements", "")).strip(),
                "companyLogoUrl": str(row.get("Company Logo URL", "")).strip() or None,
                "url": str(row.get("URL", "")).strip() or None,
                "experienceYear": experience_year,
                "minSalary": min_salary,
                "maxSalary": max_salary,
            }
            jobs_data.append(job)
            
        # Initialize BM25 index
        if jobs_data:
            print(f"Loaded {len(jobs_data)} jobs from CSV")
            print("Initializing BM25 search index...")
            bm25_index = BM25(jobs_data)
            print("BM25 index initialized successfully")
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        jobs_data = []
        bm25_index = None

# Load jobs on startup
@app.on_event("startup")
async def startup_event():
    load_jobs_from_csv()

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Job Board API",
        "version": "1.0.0",
        "total_jobs": len(jobs_data),
        "endpoints": {
            "jobs": "/api/jobs",
            "job_by_id": "/api/jobs/{id}",
            "docs": "/docs"
        }
    }

# Get all jobs with pagination and filtering
@app.get("/api/jobs", response_model=List[JobModel])
async def get_jobs(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=1000, description="Number of jobs per page"),
    field: Optional[str] = Query(None, description="Filter by field/industry"),
    location: Optional[str] = Query(None, description="Filter by location"),
    experience: Optional[str] = Query(None, description="Filter by experience years (comma-separated)"),
    salary: Optional[str] = Query(None, description="Filter by salary ranges (comma-separated)"),
    search: Optional[str] = Query(None, description="Search in job titles and descriptions"),
    sort_by: Optional[str] = Query("id", description="Sort by: id, salary, companyName")
):
    """Get jobs with optional filtering, pagination and sorting"""
    
    if not jobs_data:
        raise HTTPException(status_code=503, detail="Job data not available")
    
    # Start with all jobs
    filtered_jobs = jobs_data.copy()
    
    # Apply filters
    if field:
        filtered_jobs = [job for job in filtered_jobs if field.lower() in job["field"].lower()]
    
    if location:
        filtered_jobs = [job for job in filtered_jobs if location.lower() in job["location"].lower()]
    
    # Experience filter (by experience years)
    if experience:
        experience_values = experience.split(',')
        experience_filtered = []
        for job in filtered_jobs:
            job_exp = int(job.get('experienceYear', 0) or 0)
            for exp_val in experience_values:
                exp_val = exp_val.strip()
                if exp_val == '5' and job_exp >= 5:
                    experience_filtered.append(job)
                    break
                elif str(job_exp) == exp_val:
                    experience_filtered.append(job)
                    break
        filtered_jobs = experience_filtered
    
    # Salary filter (by salary ranges)
    if salary:
        salary_ranges = salary.split(',')
        salary_filtered = []
        for job in filtered_jobs:
            job_salary = job.get('salary', '').lower()
            job_min = float(job.get('minSalary', 0) or 0)
            job_max = float(job.get('maxSalary', 0) or 0)
            
            for salary_range in salary_ranges:
                salary_range = salary_range.strip()
                if salary_range == 'negotiate' and 'thỏa thuận' in job_salary:
                    salary_filtered.append(job)
                    break
                elif '-' in salary_range or salary_range.endswith('+'):
                    try:
                        if salary_range.endswith('+'):
                            min_sal = float(salary_range.replace('+', ''))
                            if job_min >= min_sal or job_max >= min_sal:
                                salary_filtered.append(job)
                                break
                        else:
                            min_sal, max_sal = map(float, salary_range.split('-'))
                            # Check for overlap between job salary range and filter range
                            if ((job_min >= min_sal and job_min <= max_sal) or 
                                (job_max >= min_sal and job_max <= max_sal) or
                                (job_min <= min_sal and job_max >= max_sal)):
                                salary_filtered.append(job)
                                break
                    except ValueError:
                        continue
        filtered_jobs = salary_filtered
    
    if search:
        # Use BM25 for search if available
        if bm25_index and len(search.strip()) > 1:
            try:
                # Get BM25 search results  
                search_results = bm25_index.search(search, top_k=1000)
                
                if search_results:
                    # Get job IDs from BM25 results
                    bm25_job_ids = {jobs_data[idx]["id"] for idx, score in search_results}
                    
                    # Apply BM25 filter to already filtered jobs
                    filtered_jobs = [job for job in filtered_jobs if job["id"] in bm25_job_ids]
                    
                    # Sort by BM25 score
                    job_to_score = {jobs_data[idx]["id"]: score for idx, score in search_results}
                    filtered_jobs.sort(key=lambda job: job_to_score.get(job["id"], 0), reverse=True)
                    
                    print(f"BM25 search applied: {len(filtered_jobs)} jobs match both search and filters")
                else:
                    # No BM25 results, return empty
                    filtered_jobs = []
                    print("No BM25 search results found")
                    
            except Exception as e:
                print(f"BM25 search error: {e}")
                # Fallback to simple search
                search_lower = search.lower()
                filtered_jobs = [
                    job for job in filtered_jobs 
                    if search_lower in job["jobTitle"].lower()
                    or search_lower in job["field"].lower() 
                    or search_lower in job["jobDescription"].lower()
                    or search_lower in job["companyName"].lower()
                    or search_lower in job["jobRequirements"].lower()
                ]
                print(f"Fallback simple search: {len(filtered_jobs)} jobs found")
        else:
            # Fallback to simple search for short queries
            search_lower = search.lower()
            filtered_jobs = [
                job for job in filtered_jobs 
                if search_lower in job["jobTitle"].lower()
                or search_lower in job["field"].lower() 
                or search_lower in job["jobDescription"].lower()
                or search_lower in job["companyName"].lower()
                    or search_lower in job["jobRequirements"].lower()
            ]
            print(f"Simple search for short query: {len(filtered_jobs)} jobs found")
    
    # Sort jobs
    if sort_by == "salary":
        # Simple salary sorting - extract numbers from salary string
        def extract_salary_number(salary_str):
            import re
            numbers = re.findall(r'\d+', salary_str)
            return int(numbers[-1]) if numbers else 0
        
        filtered_jobs.sort(key=lambda x: extract_salary_number(x["salary"]), reverse=True)
    elif sort_by == "companyName":
        filtered_jobs.sort(key=lambda x: x["companyName"])
    # Default is already sorted by id
    
    # Apply pagination
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    paginated_jobs = filtered_jobs[start_index:end_index]
    
    return paginated_jobs

# Get job by ID
@app.get("/api/jobs/{job_id}", response_model=JobModel)
async def get_job_by_id(job_id: int):
    """Get a specific job by ID"""
    
    if not jobs_data:
        raise HTTPException(status_code=503, detail="Job data not available")
    
    # Find job by ID
    job = next((job for job in jobs_data if job["id"] == job_id), None)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job

# Get unique locations for filter
@app.get("/api/locations")
async def get_unique_locations():
    """Get unique job locations for filter"""
    
    if not jobs_data:
        raise HTTPException(status_code=503, detail="Job data not available")
    
    # Extract unique locations
    locations = set()
    for job in jobs_data:
        location = job["location"].strip()
        location = location.split('&')[0].strip()
        if location:
            # Split multiple locations (e.g., "Hà Nội, Hưng Yên")
            location_parts = [loc.strip() for loc in location.split(",")]
            locations.update(location_parts)
    
    # Sort locations alphabetically
    return sorted(list(locations))

# Get job statistics
@app.get("/api/jobs/stats")
async def get_job_stats():
    """Get statistics about jobs"""
    
    if not jobs_data:
        raise HTTPException(status_code=503, detail="Job data not available")
    
    # Count by field
    field_counts = {}
    location_counts = {}
    experience_counts = {}
    
    for job in jobs_data:
        # Count fields
        field = job["field"]
        field_counts[field] = field_counts.get(field, 0) + 1
        
        # Count locations
        location = job["location"]
        location_counts[location] = location_counts.get(location, 0) + 1
        
        # Count experience levels
        experience = job["experience"]
        experience_counts[experience] = experience_counts.get(experience, 0) + 1
    
    return {
        "total_jobs": len(jobs_data),
        "fields": dict(sorted(field_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        "locations": dict(sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        "experience_levels": dict(sorted(experience_counts.items(), key=lambda x: x[1], reverse=True))
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "jobs_loaded": len(jobs_data) > 0,
        "total_jobs": len(jobs_data)
    }

# Reload jobs endpoint (useful for development)
@app.post("/api/reload-jobs")
async def reload_jobs():
    """Reload jobs from CSV file"""
    try:
        load_jobs_from_csv()
        return {
            "message": "Jobs reloaded successfully",
            "total_jobs": len(jobs_data),
            "bm25_index_ready": bm25_index is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading jobs: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 