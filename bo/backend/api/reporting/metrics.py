"""Metric API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import uuid
from typing import List, Optional

from db import database
from db.reporting import Metric, CalculatedMetric, DataSource
from auth.dependencies import get_current_active_user
from db.models import User
from .models import (
    MetricCreate, MetricUpdate, MetricResponse,
    CalculatedMetricCreate, CalculatedMetricUpdate, CalculatedMetricResponse
)

router = APIRouter(
    prefix="",  # No prefix here since it's added in the main router
    tags=["reporting"],
)

# Regular Metrics endpoints
@router.post("/metrics/", response_model=MetricResponse)
def create_metric(
    metric: MetricCreate,
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new metric"""
    try:
        # Check if data source exists and user has access
        db_data_source = db.query(DataSource).filter(DataSource.id == metric.data_source_id).first()
        if db_data_source is None:
            raise HTTPException(status_code=404, detail="Data source not found")

        if current_user.role != "admin" and db_data_source.creator_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to create metrics for this data source")

        db_metric = Metric(
            id=str(uuid.uuid4()),
            **metric.dict()
        )
        db.add(db_metric)
        db.commit()
        db.refresh(db_metric)
        return db_metric
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error creating metric: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in create_metric: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/metrics/", response_model=List[MetricResponse])
def list_metrics(
    data_source_id: Optional[str] = Query(None, title="Filter metrics by data source ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List metrics"""
    try:
        query = db.query(Metric)

        # Apply filters
        if data_source_id:
            query = query.filter(Metric.data_source_id == data_source_id)

        # Apply pagination
        query = query.offset(skip).limit(limit)

        return query.all()
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in list_metrics: {str(e)}")
        print(f"Error details: {error_details}")
        return []

@router.get("/metrics/{metric_id}", response_model=MetricResponse)
def get_metric(
    metric_id: str = Path(..., title="The ID of the metric to get"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific metric by ID"""
    try:
        db_metric = db.query(Metric).filter(Metric.id == metric_id).first()
        if db_metric is None:
            raise HTTPException(status_code=404, detail="Metric not found")
        
        return db_metric
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in get_metric: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/metrics/{metric_id}", response_model=MetricResponse)
def update_metric(
    metric_update: MetricUpdate,
    metric_id: str = Path(..., title="The ID of the metric to update"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update a metric"""
    try:
        db_metric = db.query(Metric).filter(Metric.id == metric_id).first()
        if db_metric is None:
            raise HTTPException(status_code=404, detail="Metric not found")

        # Check if user has access to the data source
        db_data_source = db.query(DataSource).filter(DataSource.id == db_metric.data_source_id).first()
        if db_data_source is None:
            raise HTTPException(status_code=404, detail="Data source not found")

        if current_user.role != "admin" and db_data_source.creator_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to update metrics for this data source")

        # Update fields
        update_data = metric_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_metric, field, value)

        db.commit()
        db.refresh(db_metric)
        return db_metric
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error updating metric: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in update_metric: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/metrics/{metric_id}", response_model=dict)
def delete_metric(
    metric_id: str = Path(..., title="The ID of the metric to delete"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a metric"""
    try:
        db_metric = db.query(Metric).filter(Metric.id == metric_id).first()
        if db_metric is None:
            raise HTTPException(status_code=404, detail="Metric not found")

        # Check if user has access to the data source
        db_data_source = db.query(DataSource).filter(DataSource.id == db_metric.data_source_id).first()
        if db_data_source is None:
            raise HTTPException(status_code=404, detail="Data source not found")

        if current_user.role != "admin" and db_data_source.creator_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to delete metrics for this data source")

        db.delete(db_metric)
        db.commit()
        return {"message": "Metric deleted successfully"}
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error deleting metric: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in delete_metric: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Calculated Metrics endpoints
@router.post("/calculated-metrics/", response_model=CalculatedMetricResponse)
def create_calculated_metric(
    calculated_metric: CalculatedMetricCreate,
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new calculated metric"""
    try:
        # Check if data source exists and user has access
        db_data_source = db.query(DataSource).filter(DataSource.id == calculated_metric.data_source_id).first()
        if db_data_source is None:
            raise HTTPException(status_code=404, detail="Data source not found")

        if current_user.role != "admin" and db_data_source.creator_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to create calculated metrics for this data source")

        db_calculated_metric = CalculatedMetric(
            id=str(uuid.uuid4()),
            **calculated_metric.dict()
        )
        db.add(db_calculated_metric)
        db.commit()
        db.refresh(db_calculated_metric)
        return db_calculated_metric
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error creating calculated metric: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in create_calculated_metric: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/calculated-metrics/", response_model=List[CalculatedMetricResponse])
def list_calculated_metrics(
    data_source_id: Optional[str] = Query(None, title="Filter calculated metrics by data source ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List calculated metrics"""
    try:
        query = db.query(CalculatedMetric)

        # Apply filters
        if data_source_id:
            query = query.filter(CalculatedMetric.data_source_id == data_source_id)

        # Apply pagination
        query = query.offset(skip).limit(limit)

        return query.all()
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in list_calculated_metrics: {str(e)}")
        print(f"Error details: {error_details}")
        return []

@router.get("/calculated-metrics/{calculated_metric_id}", response_model=CalculatedMetricResponse)
def get_calculated_metric(
    calculated_metric_id: str = Path(..., title="The ID of the calculated metric to get"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific calculated metric by ID"""
    try:
        db_calculated_metric = db.query(CalculatedMetric).filter(CalculatedMetric.id == calculated_metric_id).first()
        if db_calculated_metric is None:
            raise HTTPException(status_code=404, detail="Calculated metric not found")
        
        return db_calculated_metric
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in get_calculated_metric: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/calculated-metrics/{calculated_metric_id}", response_model=CalculatedMetricResponse)
def update_calculated_metric(
    calculated_metric_update: CalculatedMetricUpdate,
    calculated_metric_id: str = Path(..., title="The ID of the calculated metric to update"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update a calculated metric"""
    try:
        db_calculated_metric = db.query(CalculatedMetric).filter(CalculatedMetric.id == calculated_metric_id).first()
        if db_calculated_metric is None:
            raise HTTPException(status_code=404, detail="Calculated metric not found")

        # Check if user has access to the data source
        db_data_source = db.query(DataSource).filter(DataSource.id == db_calculated_metric.data_source_id).first()
        if db_data_source is None:
            raise HTTPException(status_code=404, detail="Data source not found")

        if current_user.role != "admin" and db_data_source.creator_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to update calculated metrics for this data source")

        # Update fields
        update_data = calculated_metric_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_calculated_metric, field, value)

        db.commit()
        db.refresh(db_calculated_metric)
        return db_calculated_metric
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error updating calculated metric: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in update_calculated_metric: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/calculated-metrics/{calculated_metric_id}", response_model=dict)
def delete_calculated_metric(
    calculated_metric_id: str = Path(..., title="The ID of the calculated metric to delete"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a calculated metric"""
    try:
        db_calculated_metric = db.query(CalculatedMetric).filter(CalculatedMetric.id == calculated_metric_id).first()
        if db_calculated_metric is None:
            raise HTTPException(status_code=404, detail="Calculated metric not found")

        # Check if user has access to the data source
        db_data_source = db.query(DataSource).filter(DataSource.id == db_calculated_metric.data_source_id).first()
        if db_data_source is None:
            raise HTTPException(status_code=404, detail="Data source not found")

        if current_user.role != "admin" and db_data_source.creator_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to delete calculated metrics for this data source")

        db.delete(db_calculated_metric)
        db.commit()
        return {"message": "Calculated metric deleted successfully"}
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error deleting calculated metric: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in delete_calculated_metric: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")