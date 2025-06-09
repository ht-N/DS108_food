#!/usr/bin/env python3
"""
Startup script for the Job Board FastAPI server
"""

import uvicorn
import os
from pathlib import Path

def start_server():
    """Start the FastAPI server with proper configuration"""
    
    # Change to the backend directory if not already there
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    
    # Configuration
    config = {
        "app": "fastapi_app:app",
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,  # Enable auto-reload for development
        "log_level": "info"
    }
    
    print("🚀 Starting Job Board API Server...")
    print(f"📍 Server will be available at: http://localhost:8000")
    print(f"📚 API Documentation: http://localhost:8000/docs")
    print(f"🔄 Auto-reload: {'Enabled' if config['reload'] else 'Disabled'}")
    print("-" * 50)
    
    # Start the server
    uvicorn.run(**config)

if __name__ == "__main__":
    start_server() 