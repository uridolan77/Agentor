from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from typing import List, Dict, Any, Optional

# Import modules
from db.database import init_db
from db.init_data import init_data
from auth.router import router as auth_router

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("agentor-backoffice")
logger.setLevel(logging.DEBUG)

# Create FastAPI app
app = FastAPI(
    title="Agentor BackOffice API",
    description="API for managing Agentor agents, tools, and workflows",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Frontend origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

# Import security utilities
from auth.security import oauth2_scheme

# Import API routers
from api import api_router
from auth.router import router as auth_router

# Include routers
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(api_router)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized successfully")

    # Initialize database data
    logger.info("Initializing database data...")
    init_data()
    logger.info("Database data initialized successfully")

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint returning API information."""
    return {
        "name": "Agentor BackOffice API",
        "version": "0.1.0",
        "status": "running",
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
