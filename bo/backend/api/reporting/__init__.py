"""Reporting API module."""

from fastapi import APIRouter

# Import routers from each module
from .datasources import router as datasources_router
from .reports import router as reports_router
from .dimensions import router as dimensions_router
from .metrics import router as metrics_router
from .datamodels import router as datamodels_router

# Create main reporting router
router = APIRouter(
    prefix="/reporting",
    tags=["reporting"],
    responses={404: {"description": "Not found"}},
)

# Include sub-routers
router.include_router(datasources_router)
router.include_router(reports_router)
router.include_router(dimensions_router)
router.include_router(metrics_router)
router.include_router(datamodels_router)