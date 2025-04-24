"""API module for the Agentor BackOffice."""

from fastapi import APIRouter

# Import routers
from .training import router as training_router
from .reporting import router as reporting_router  # Import our modular reporting router

# Create main API router
api_router = APIRouter()

# Include training router
api_router.include_router(training_router)

# Include reporting router
api_router.include_router(reporting_router)
