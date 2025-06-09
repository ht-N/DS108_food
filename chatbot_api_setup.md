# Hướng dẫn Setup API Chatbot trên Google Colab

## 🚀 Cài đặt API Chatbot cho RAG System

### 1. Tạo notebook mới trên Google Colab

```python
# Install required packages
!pip install fastapi uvicorn pyngrok python-multipart
!pip install transformers torch
!pip install langchain sentence-transformers
!pip install faiss-cpu  # or faiss-gpu if using GPU
!pip install PyPDF2 python-docx

# Import libraries
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pyngrok import ngrok
import nest_asyncio
import json
import os
from datetime import datetime
```

### 2. Setup RAG System (Basic Template)

```python
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2
from docx import Document
import io

class JobRAGSystem:
    def __init__(self):
        # Load embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize FAISS index (will be populated with job data)
        self.index = None
        self.job_data = []
        
        # Load job data (replace with your CSV data)
        self.load_job_data()
    
    def load_job_data(self):
        # Load your job data here
        # This should match the CSV data structure
        pass
    
    def extract_text_from_pdf(self, file_bytes):
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    def extract_text_from_docx(self, file_bytes):
        doc = Document(io.BytesIO(file_bytes))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def analyze_cv(self, cv_text):
        # Implement CV analysis logic
        return "CV analysis result here"
    
    def search_jobs(self, query, top_k=5):
        # Implement job search using RAG
        return []
    
    def generate_response(self, query, cv_text=None):
        # Implement response generation
        return "Generated response here"

# Initialize RAG system
rag_system = JobRAGSystem()
```

### 3. Setup FastAPI Application

```python
# Create FastAPI app
app = FastAPI(title="Job Chatbot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat_endpoint(
    message: str = Form(...),
    timestamp: str = Form(...),
    cv_file: UploadFile = File(None)
):
    try:
        response_data = {
            "response": "",
            "cv_analysis": None,
            "job_recommendations": []
        }
        
        cv_text = None
        
        # Process uploaded CV if provided
        if cv_file:
            file_bytes = await cv_file.read()
            
            if cv_file.content_type == "application/pdf":
                cv_text = rag_system.extract_text_from_pdf(file_bytes)
            elif cv_file.content_type in ["application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                cv_text = rag_system.extract_text_from_docx(file_bytes)
            
            if cv_text:
                response_data["cv_analysis"] = rag_system.analyze_cv(cv_text)
        
        # Generate response based on message and CV
        response_data["response"] = rag_system.generate_response(message, cv_text)
        
        # Search for relevant jobs
        if any(keyword in message.lower() for keyword in ["việc làm", "job", "tuyển dụng"]):
            response_data["job_recommendations"] = rag_system.search_jobs(message)
        
        return response_data
        
    except Exception as e:
        return {
            "response": f"Xin lỗi, có lỗi xảy ra: {str(e)}",
            "cv_analysis": None,
            "job_recommendations": []
        }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
```

### 4. Start Server với Ngrok

```python
# Enable nested asyncio for Jupyter
nest_asyncio.apply()

# Setup ngrok authentication (get your token from ngrok.com)
ngrok.set_auth_token("YOUR_NGROK_TOKEN")  # Replace with your token

# Start ngrok tunnel
public_url = ngrok.connect(8000)
print(f"🚀 Public URL: {public_url}")
print(f"📋 Use this URL in your frontend: {public_url}/chat")

# Start the server
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 5. Cập nhật URL trong Frontend

Sau khi chạy Colab và có public URL từ ngrok, cập nhật trong `job_listing.html`:

```javascript
// Thay thế URL này bằng URL từ ngrok
const CHATBOT_API_URL = 'https://abc123.ngrok.io/chat';
```

### 6. Test API

```python
# Test endpoint manually
import requests

url = "YOUR_NGROK_URL/chat"
data = {
    "message": "Tôi đang tìm việc làm lập trình",
    "timestamp": "2024-01-01T00:00:00Z"
}

response = requests.post(url, data=data)
print(response.json())
```

## 🔧 Tùy chỉnh nâng cao

### 1. Thêm Vietnamese Language Model

```python
# Use Vietnamese-specific models
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "vinai/phobert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
```

### 2. Implement CV Parsing

```python
def advanced_cv_analysis(self, cv_text):
    # Extract skills, experience, education
    skills = self.extract_skills(cv_text)
    experience = self.extract_experience(cv_text)
    education = self.extract_education(cv_text)
    
    analysis = {
        "skills": skills,
        "experience_years": experience,
        "education": education,
        "recommendations": self.generate_recommendations(skills, experience)
    }
    
    return analysis
```

### 3. Job Matching Algorithm

```python
def match_jobs_to_cv(self, cv_analysis):
    # Use embedding similarity to match CV with jobs
    cv_embedding = self.embedding_model.encode(cv_analysis["skills"])
    
    similarities = []
    for job in self.job_data:
        job_embedding = self.embedding_model.encode(job["requirements"])
        similarity = np.dot(cv_embedding, job_embedding)
        similarities.append((job, similarity))
    
    # Return top matches
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [job for job, score in similarities[:5]]
```

## 📝 Notes

- Nhớ thay thế `YOUR_NGROK_TOKEN` bằng token thực tế từ ngrok.com
- URL ngrok sẽ thay đổi mỗi khi restart, cần cập nhật trong frontend
- Để production, nên deploy lên server thật thay vì dùng ngrok
- Có thể sử dụng Google Colab Pro để có GPU và tăng performance

## 🚀 Next Steps

1. Implement advanced NLP models cho tiếng Việt
2. Tích hợp với database jobs thực tế
3. Thêm conversation memory
4. Implement user authentication
5. Add logging và monitoring 