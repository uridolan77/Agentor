"""Dimension API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import uuid
from typing import List, Optional

from db import database
from db.reporting import Dimension, DataSource
from auth.dependencies import get_current_active_user
from db.models import User
from .models import DimensionCreate, DimensionUpdate, DimensionResponse

router = APIRouter(
    prefix="",  # No prefix here since it's added in the main router
    tags=["reporting"],
)

@router.post("/dimensions/", response_model=DimensionResponse)
def create_dimension(
    dimension: DimensionCreate,
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new dimension"""
    try:
        # Check if data source exists and user has access
        db_data_source = db.query(DataSource).filter(DataSource.id == dimension.data_source_id).first()
        if db_data_source is None:
            raise HTTPException(status_code=404, detail="Data source not found")

        if current_user.role != "admin" and db_data_source.creator_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to create dimensions for this data source")

        db_dimension = Dimension(
            id=str(uuid.uuid4()),
            **dimension.dict()
        )
        db.add(db_dimension)
        db.commit()
        db.refresh(db_dimension)
        return db_dimension
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error creating dimension: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in create_dimension: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/dimensions/", response_model=List[DimensionResponse])
def list_dimensions(
    data_source_id: Optional[str] = Query(None, title="Filter dimensions by data source ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List dimensions"""
    try:
        query = db.query(Dimension)

        # Apply filters
        if data_source_id:
            query = query.filter(Dimension.data_source_id == data_source_id)

        # Apply pagination
        query = query.offset(skip).limit(limit)

        return query.all()
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in list_dimensions: {str(e)}")
        print(f"Error details: {error_details}")
        return []

@router.get("/dimensions/{dimension_id}", response_model=DimensionResponse)
def get_dimension(
    dimension_id: str = Path(..., title="The ID of the dimension to get"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific dimension by ID"""
    try:
        db_dimension = db.query(Dimension).filter(Dimension.id == dimension_id).first()
        if db_dimension is None:
            raise HTTPException(status_code=404, detail="Dimension not found")
        
        return db_dimension
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in get_dimension: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/dimensions/{dimension_id}", response_model=DimensionResponse)
def update_dimension(
    dimension_update: DimensionUpdate,
    dimension_id: str = Path(..., title="The ID of the dimension to update"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update a dimension"""
    try:
        db_dimension = db.query(Dimension).filter(Dimension.id == dimension_id).first()
        if db_dimension is None:
            raise HTTPException(status_code=404, detail="Dimension not found")

        # Check if user has access to the data source
        db_data_source = db.query(DataSource).filter(DataSource.id == db_dimension.data_source_id).first()
        if db_data_source is None:
            raise HTTPException(status_code=404, detail="Data source not found")

        if current_user.role != "admin" and db_data_source.creator_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to update dimensions for this data source")

        # Update fields
        update_data = dimension_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_dimension, field, value)

        db.commit()
        db.refresh(db_dimension)
        return db_dimension
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error updating dimension: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in update_dimension: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/dimensions/{dimension_id}", response_model=dict)
def delete_dimension(
    dimension_id: str = Path(..., title="The ID of the dimension to delete"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a dimension"""
    try:
        db_dimension = db.query(Dimension).filter(Dimension.id == dimension_id).first()
        if db_dimension is None:
            raise HTTPException(status_code=404, detail="Dimension not found")

        # Check if user has access to the data source
        db_data_source = db.query(DataSource).filter(DataSource.id == db_dimension.data_source_id).first()
        if db_data_source is None:
            raise HTTPException(status_code=404, detail="Data source not found")

        if current_user.role != "admin" and db_data_source.creator_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to delete dimensions for this data source")

        db.delete(db_dimension)
        db.commit()
        return {"message": "Dimension deleted successfully"}
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error deleting dimension: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in delete_dimension: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")