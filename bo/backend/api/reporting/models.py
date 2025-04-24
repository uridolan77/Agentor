"""Pydantic models for the reporting API."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# DataSource models
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

# Dimension models
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

# Metric models
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

# CalculatedMetric models
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

# Report models
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

# ReportPermission models
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