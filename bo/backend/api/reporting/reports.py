"""Report API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from db import database
from db.reporting import Report, ReportPermission, DataSource
from auth.dependencies import get_current_active_user
from db.models import User
from .models import ReportCreate, ReportUpdate, ReportResponse, ReportPermissionCreate, ReportPermissionResponse

router = APIRouter(
    prefix="",  # No prefix here since it's added in the main router
    tags=["reporting"],
)

@router.post("/reports/", response_model=ReportResponse)
def create_report(
    report: ReportCreate,
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new report"""
    try:
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
        db.commit()
        db.refresh(db_report)
        return db_report
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error creating report: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in create_report: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")

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
    try:
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
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in get_report: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/reports/{report_id}", response_model=ReportResponse)
def update_report(
    report_update: ReportUpdate,
    report_id: str = Path(..., title="The ID of the report to update"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update a report"""
    try:
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

        db.commit()
        db.refresh(db_report)
        return db_report
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error updating report: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in update_report: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/reports/{report_id}", response_model=dict)
def delete_report(
    report_id: str = Path(..., title="The ID of the report to delete"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a report"""
    try:
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
        db.commit()
        return {"message": "Report deleted successfully"}
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error deleting report: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in delete_report: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Endpoint for running a report (this would be implemented in a separate service)
@router.post("/reports/{report_id}/run", response_model=dict)
def run_report(
    report_id: str = Path(..., title="The ID of the report to run"),
    params: Optional[Dict[str, Any]] = Body(None, title="Parameters for the report"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Run a report and return results"""
    try:
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
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in run_report: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")

# API endpoints for ReportPermissions
@router.post("/reports/{report_id}/permissions", response_model=ReportPermissionResponse)
def add_report_permission(
    permission: ReportPermissionCreate,
    report_id: str = Path(..., title="The ID of the report to share"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Share a report with a user or team"""
    try:
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
        db.commit()
        db.refresh(db_permission)
        return db_permission
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error creating permission: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in add_report_permission: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/reports/{report_id}/permissions", response_model=List[ReportPermissionResponse])
def list_report_permissions(
    report_id: str = Path(..., title="The ID of the report"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List all permissions for a report"""
    try:
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
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in list_report_permissions: {str(e)}")
        print(f"Error details: {error_details}")
        return []

@router.delete("/reports/{report_id}/permissions/{permission_id}", response_model=dict)
def remove_report_permission(
    report_id: str = Path(..., title="The ID of the report"),
    permission_id: int = Path(..., title="The ID of the permission to remove"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Remove a permission for a report"""
    try:
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
        db.commit()
        return {"message": "Permission removed successfully"}
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error removing permission: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in remove_report_permission: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")