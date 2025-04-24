from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

from db import database
from db.reporting import DataSource, Dimension, Metric, CalculatedMetric, Report, ReportPermission
from auth.dependencies import get_current_active_user
from db.models import User

router = APIRouter(
    prefix="/reporting",
    tags=["reporting"],
    responses={404: {"description": "Not found"}},
)

# Pydantic models for request and response
from pydantic import BaseModel, Field

class DataSourceBase(BaseModel):
    name: str
    description: Optional[str] = None
    connection_type: str
    connection_config: Dict[str, Any]

class DataSourceCreate(DataSourceBase):
    pass

class DataSourceUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    connection_type: Optional[str] = None
    connection_config: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class DataSourceResponse(DataSourceBase):
    id: str
    creator_id: int
    created_at: datetime
    updated_at: datetime
    is_active: bool

    class Config:
        from_attributes = True

class DimensionBase(BaseModel):
    name: str
    display_name: str
    description: Optional[str] = None
    data_source_id: str
    data_type: str
    table_name: str
    column_name: str
    formatting: Optional[Dict[str, Any]] = None

class DimensionCreate(DimensionBase):
    pass

class DimensionUpdate(BaseModel):
    name: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    data_type: Optional[str] = None
    table_name: Optional[str] = None
    column_name: Optional[str] = None
    formatting: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class DimensionResponse(DimensionBase):
    id: str
    created_at: datetime
    updated_at: datetime
    is_active: bool

    class Config:
        from_attributes = True

class MetricBase(BaseModel):
    name: str
    display_name: str
    description: Optional[str] = None
    data_source_id: str
    data_type: str
    calculation_type: str
    table_name: str
    column_name: str
    formatting: Optional[Dict[str, Any]] = None

class MetricCreate(MetricBase):
    pass

class MetricUpdate(BaseModel):
    name: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    data_type: Optional[str] = None
    calculation_type: Optional[str] = None
    table_name: Optional[str] = None
    column_name: Optional[str] = None
    formatting: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class MetricResponse(MetricBase):
    id: str
    created_at: datetime
    updated_at: datetime
    is_active: bool

    class Config:
        from_attributes = True

class CalculatedMetricBase(BaseModel):
    name: str
    display_name: str
    description: Optional[str] = None
    data_source_id: str
    data_type: str
    formula: str
    dependencies: List[str]  # List of metric IDs
    formatting: Optional[Dict[str, Any]] = None

class CalculatedMetricCreate(CalculatedMetricBase):
    pass

class CalculatedMetricUpdate(BaseModel):
    name: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    data_type: Optional[str] = None
    formula: Optional[str] = None
    dependencies: Optional[List[str]] = None
    formatting: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class CalculatedMetricResponse(CalculatedMetricBase):
    id: str
    created_at: datetime
    updated_at: datetime
    is_active: bool

    class Config:
        from_attributes = True

class ReportBase(BaseModel):
    name: str
    description: Optional[str] = None
    data_source_id: str
    configuration: Dict[str, Any]
    is_public: bool = False
    is_favorite: bool = False
    thumbnail: Optional[str] = None

class ReportCreate(ReportBase):
    pass

class ReportUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    configuration: Optional[Dict[str, Any]] = None
    is_public: Optional[bool] = None
    is_favorite: Optional[bool] = None
    last_run_at: Optional[datetime] = None
    thumbnail: Optional[str] = None

class ReportResponse(ReportBase):
    id: str
    creator_id: int
    created_at: datetime
    updated_at: datetime
    last_run_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class ReportPermissionBase(BaseModel):
    report_id: str
    user_id: Optional[int] = None
    team_id: Optional[int] = None
    permission_type: str = "view"  # view, edit, admin

class ReportPermissionCreate(ReportPermissionBase):
    pass

class ReportPermissionResponse(ReportPermissionBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

# API endpoints for DataSource
@router.post("/datasources/", response_model=DataSourceResponse)
def create_data_source(
    data_source: DataSourceCreate,
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new data source"""
    db_data_source = DataSource(
        id=str(uuid.uuid4()),
        **data_source.dict(),
        creator_id=current_user.id
    )
    db.add(db_data_source)
    try:
        db.commit()
        db.refresh(db_data_source)
        return db_data_source
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error creating data source: {str(e)}")

@router.get("/datasources/", response_model=List[DataSourceResponse])
def list_data_sources(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List data sources"""
    # If admin, return all data sources, otherwise filter by creator
    if current_user.role == "admin":
        db_data_sources = db.query(DataSource).offset(skip).limit(limit).all()
    else:
        db_data_sources = db.query(DataSource).filter(DataSource.creator_id == current_user.id).offset(skip).limit(limit).all()
    return db_data_sources

@router.get("/datasources/{data_source_id}", response_model=DataSourceResponse)
def get_data_source(
    data_source_id: str = Path(..., title="The ID of the data source to get"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific data source by ID"""
    db_data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
    if db_data_source is None:
        raise HTTPException(status_code=404, detail="Data source not found")

    # Check permissions (admin or creator)
    if current_user.role != "admin" and db_data_source.creator_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this data source")

    return db_data_source

@router.put("/datasources/{data_source_id}", response_model=DataSourceResponse)
def update_data_source(
    data_source_update: DataSourceUpdate,
    data_source_id: str = Path(..., title="The ID of the data source to update"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update a data source"""
    db_data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
    if db_data_source is None:
        raise HTTPException(status_code=404, detail="Data source not found")

    # Check permissions (admin or creator)
    if current_user.role != "admin" and db_data_source.creator_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to update this data source")

    # Update fields
    update_data = data_source_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_data_source, field, value)

    try:
        db.commit()
        db.refresh(db_data_source)
        return db_data_source
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error updating data source: {str(e)}")

@router.delete("/datasources/{data_source_id}", response_model=dict)
def delete_data_source(
    data_source_id: str = Path(..., title="The ID of the data source to delete"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a data source"""
    db_data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
    if db_data_source is None:
        raise HTTPException(status_code=404, detail="Data source not found")

    # Check permissions (admin or creator)
    if current_user.role != "admin" and db_data_source.creator_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this data source")

    db.delete(db_data_source)
    try:
        db.commit()
        return {"message": "Data source deleted successfully"}
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error deleting data source: {str(e)}")

# API endpoints for Dimension
@router.post("/dimensions/", response_model=DimensionResponse)
def create_dimension(
    dimension: DimensionCreate,
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new dimension"""
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
    try:
        db.commit()
        db.refresh(db_dimension)
        return db_dimension
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error creating dimension: {str(e)}")

@router.get("/dimensions/", response_model=List[DimensionResponse])
def list_dimensions(
    data_source_id: Optional[str] = Query(None, title="Filter dimensions by data source ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List dimensions"""
    query = db.query(Dimension)

    # Apply filters
    if data_source_id:
        query = query.filter(Dimension.data_source_id == data_source_id)

    # Apply pagination
    query = query.offset(skip).limit(limit)

    return query.all()

# Similar endpoints for Metrics, CalculatedMetrics, and Reports...
# For brevity, I'll implement the Reports endpoints and you can follow the same pattern for others

# API endpoints for Report
@router.post("/reports/", response_model=ReportResponse)
def create_report(
    report: ReportCreate,
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new report"""
    # Check if data source exists
    db_data_source = db.query(DataSource).filter(DataSource.id == report.data_source_id).first()
    if db_data_source is None:
        raise HTTPException(status_code=404, detail="Data source not found")

    db_report = Report(
        id=str(uuid.uuid4()),
        **report.dict(),
        creator_id=current_user.id
    )
    db.add(db_report)
    try:
        db.commit()
        db.refresh(db_report)
        return db_report
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error creating report: {str(e)}")

@router.get("/reports/", response_model=List[ReportResponse])
def list_reports(
    data_source_id: Optional[str] = Query(None, title="Filter reports by data source ID"),
    is_public: Optional[bool] = Query(None, title="Filter reports by public status"),
    is_favorite: Optional[bool] = Query(None, title="Filter reports by favorite status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List reports"""
    try:
        # Base query for reports the user can access
        base_query = db.query(Report.id)
        
        # User's own reports
        user_reports = base_query.filter(Report.creator_id == current_user.id).subquery()
        
        # Public reports
        public_reports = base_query.filter(Report.is_public == True).subquery()
        
        # Reports shared directly with the user through permissions
        user_permission_reports = db.query(ReportPermission.report_id).filter(
            ReportPermission.user_id == current_user.id
        ).subquery()
        
        # Reports shared with user's teams - handle case where user has no teams
        team_permission_reports = None
        if hasattr(current_user, 'teams') and current_user.teams:
            # Get all team IDs the user belongs to
            team_ids = [team.id for team in current_user.teams]
            if team_ids:
                team_permission_reports = db.query(ReportPermission.report_id).filter(
                    ReportPermission.team_id.in_(team_ids)
                ).subquery()
        
        # Combine all accessible report IDs
        accessible_report_ids = []
        accessible_report_ids.extend([r[0] for r in db.query(user_reports).all()])
        accessible_report_ids.extend([r[0] for r in db.query(public_reports).all()])
        accessible_report_ids.extend([r[0] for r in db.query(user_permission_reports).all()])
        
        if team_permission_reports:
            accessible_report_ids.extend([r[0] for r in db.query(team_permission_reports).all()])
        
        # Remove duplicates
        accessible_report_ids = list(set(accessible_report_ids))
        
        # Query for the final reports with filters
        query = db.query(Report).filter(Report.id.in_(accessible_report_ids))

        # Apply additional filters
        if data_source_id:
            query = query.filter(Report.data_source_id == data_source_id)

        if is_public is not None:
            query = query.filter(Report.is_public == is_public)

        if is_favorite is not None:
            query = query.filter(Report.is_favorite == is_favorite)

        # Apply pagination
        query = query.offset(skip).limit(limit)

        return query.all()
        
    except Exception as e:
        # Log the error for debugging
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in list_reports: {str(e)}")
        print(f"Error details: {error_details}")
        
        # Return an empty list instead of failing
        return []

@router.get("/reports/{report_id}", response_model=ReportResponse)
def get_report(
    report_id: str = Path(..., title="The ID of the report to get"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific report by ID"""
    db_report = db.query(Report).filter(Report.id == report_id).first()
    if db_report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    # Check permissions
    if (
        db_report.creator_id != current_user.id and
        not db_report.is_public and
        not db.query(ReportPermission).filter(
            ReportPermission.report_id == report_id,
            ReportPermission.user_id == current_user.id
        ).first()
    ):
        raise HTTPException(status_code=403, detail="Not authorized to access this report")

    return db_report

@router.put("/reports/{report_id}", response_model=ReportResponse)
def update_report(
    report_update: ReportUpdate,
    report_id: str = Path(..., title="The ID of the report to update"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update a report"""
    db_report = db.query(Report).filter(Report.id == report_id).first()
    if db_report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    # Check permissions
    if (
        db_report.creator_id != current_user.id and
        not db.query(ReportPermission).filter(
            ReportPermission.report_id == report_id,
            ReportPermission.user_id == current_user.id,
            ReportPermission.permission_type.in_(["edit", "admin"])
        ).first()
    ):
        raise HTTPException(status_code=403, detail="Not authorized to update this report")

    # Update fields
    update_data = report_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_report, field, value)

    try:
        db.commit()
        db.refresh(db_report)
        return db_report
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error updating report: {str(e)}")

@router.delete("/reports/{report_id}", response_model=dict)
def delete_report(
    report_id: str = Path(..., title="The ID of the report to delete"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a report"""
    db_report = db.query(Report).filter(Report.id == report_id).first()
    if db_report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    # Check permissions (admin, creator, or user with admin permission)
    if (
        current_user.role != "admin" and
        db_report.creator_id != current_user.id and
        not db.query(ReportPermission).filter(
            ReportPermission.report_id == report_id,
            ReportPermission.user_id == current_user.id,
            ReportPermission.permission_type == "admin"
        ).first()
    ):
        raise HTTPException(status_code=403, detail="Not authorized to delete this report")

    db.delete(db_report)
    try:
        db.commit()
        return {"message": "Report deleted successfully"}
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error deleting report: {str(e)}")

# Endpoint for running a report (this would be implemented in a separate service)
@router.post("/reports/{report_id}/run", response_model=dict)
def run_report(
    report_id: str = Path(..., title="The ID of the report to run"),
    params: Optional[Dict[str, Any]] = Body(None, title="Parameters for the report"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Run a report and return results"""
    db_report = db.query(Report).filter(Report.id == report_id).first()
    if db_report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    # Check permissions
    if (
        db_report.creator_id != current_user.id and
        not db_report.is_public and
        not db.query(ReportPermission).filter(
            ReportPermission.report_id == report_id,
            ReportPermission.user_id == current_user.id
        ).first()
    ):
        raise HTTPException(status_code=403, detail="Not authorized to run this report")

    # Update last run time
    db_report.last_run_at = datetime.now()
    db.commit()

    # Here you would call your report execution service
    # This is a placeholder implementation
    return {
        "report_id": report_id,
        "status": "success",
        "message": "Report execution started",
        "execution_id": str(uuid.uuid4())
    }

# Additional endpoints for report sharing
@router.post("/reports/{report_id}/permissions", response_model=ReportPermissionResponse)
def add_report_permission(
    permission: ReportPermissionCreate,
    report_id: str = Path(..., title="The ID of the report to share"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Share a report with a user or team"""
    db_report = db.query(Report).filter(Report.id == report_id).first()
    if db_report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    # Check if user is authorized to share (admin, creator, or has admin permission)
    if (
        current_user.role != "admin" and
        db_report.creator_id != current_user.id and
        not db.query(ReportPermission).filter(
            ReportPermission.report_id == report_id,
            ReportPermission.user_id == current_user.id,
            ReportPermission.permission_type == "admin"
        ).first()
    ):
        raise HTTPException(status_code=403, detail="Not authorized to share this report")

    # Check if at least one of user_id or team_id is provided
    if permission.user_id is None and permission.team_id is None:
        raise HTTPException(status_code=400, detail="Either user_id or team_id must be provided")

    # Create permission
    db_permission = ReportPermission(**permission.dict())
    db.add(db_permission)
    try:
        db.commit()
        db.refresh(db_permission)
        return db_permission
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error creating permission: {str(e)}")

@router.get("/reports/{report_id}/permissions", response_model=List[ReportPermissionResponse])
def list_report_permissions(
    report_id: str = Path(..., title="The ID of the report"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List all permissions for a report"""
    db_report = db.query(Report).filter(Report.id == report_id).first()
    if db_report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    # Check if user is authorized to view permissions
    if (
        current_user.role != "admin" and
        db_report.creator_id != current_user.id and
        not db.query(ReportPermission).filter(
            ReportPermission.report_id == report_id,
            ReportPermission.user_id == current_user.id,
            ReportPermission.permission_type == "admin"
        ).first()
    ):
        raise HTTPException(status_code=403, detail="Not authorized to view permissions for this report")

    return db.query(ReportPermission).filter(ReportPermission.report_id == report_id).all()

@router.delete("/reports/{report_id}/permissions/{permission_id}", response_model=dict)
def remove_report_permission(
    report_id: str = Path(..., title="The ID of the report"),
    permission_id: int = Path(..., title="The ID of the permission to remove"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Remove a permission for a report"""
    db_report = db.query(Report).filter(Report.id == report_id).first()
    if db_report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    db_permission = db.query(ReportPermission).filter(
        ReportPermission.id == permission_id,
        ReportPermission.report_id == report_id
    ).first()
    if db_permission is None:
        raise HTTPException(status_code=404, detail="Permission not found")

    # Check if user is authorized to remove permissions
    if (
        current_user.role != "admin" and
        db_report.creator_id != current_user.id and
        not db.query(ReportPermission).filter(
            ReportPermission.report_id == report_id,
            ReportPermission.user_id == current_user.id,
            ReportPermission.permission_type == "admin"
        ).first()
    ):
        raise HTTPException(status_code=403, detail="Not authorized to remove permissions for this report")

    db.delete(db_permission)
    try:
        db.commit()
        return {"message": "Permission removed successfully"}
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error removing permission: {str(e)}")