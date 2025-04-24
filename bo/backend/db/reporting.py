from sqlalchemy import Boolean, Column, Integer, String, DateTime, ForeignKey, JSON, Table
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from .database import Base

class DataSource(Base):
    """Data source model for connecting to external data"""
    __tablename__ = "data_sources"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    connection_type = Column(String, nullable=False)  # 'database', 'api', 'file', etc.
    connection_config = Column(JSON, nullable=False)
    creator_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    is_active = Column(Boolean, default=True)

    # Relationships
    creator = relationship("User", backref="data_sources")
    dimensions = relationship("Dimension", back_populates="data_source", cascade="all, delete-orphan")
    metrics = relationship("Metric", back_populates="data_source", cascade="all, delete-orphan")
    calculated_metrics = relationship("CalculatedMetric", back_populates="data_source", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="data_source")
    data_models = relationship("DataModel", back_populates="data_source", cascade="all, delete-orphan")

class Dimension(Base):
    """Dimension model for report fields representing categories or groupings"""
    __tablename__ = "dimensions"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    display_name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    data_source_id = Column(String, ForeignKey("data_sources.id"))
    data_type = Column(String, nullable=False)  # 'string', 'number', 'date', etc.
    table_name = Column(String, nullable=False)
    column_name = Column(String, nullable=False)
    formatting = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    is_active = Column(Boolean, default=True)

    # Relationships
    data_source = relationship("DataSource", back_populates="dimensions")

class Metric(Base):
    """Metric model for report fields representing numerical values"""
    __tablename__ = "metrics"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    display_name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    data_source_id = Column(String, ForeignKey("data_sources.id"))
    data_type = Column(String, nullable=False)  # 'number', 'percentage', etc.
    calculation_type = Column(String, nullable=False)  # 'sum', 'average', 'count', etc.
    table_name = Column(String, nullable=False)
    column_name = Column(String, nullable=False)
    formatting = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    is_active = Column(Boolean, default=True)

    # Relationships
    data_source = relationship("DataSource", back_populates="metrics")

class CalculatedMetric(Base):
    """Calculated metric model for custom formulas based on other metrics"""
    __tablename__ = "calculated_metrics"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    display_name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    data_source_id = Column(String, ForeignKey("data_sources.id"))
    data_type = Column(String, nullable=False)  # 'number', 'percentage', etc.
    formula = Column(String, nullable=False)  # SQL or expression to calculate the metric
    dependencies = Column(JSON, nullable=False)  # List of metric IDs this calculation depends on
    formatting = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    is_active = Column(Boolean, default=True)

    # Relationships
    data_source = relationship("DataSource", back_populates="calculated_metrics")

class Report(Base):
    """Report model storing report configurations"""
    __tablename__ = "reports"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    data_source_id = Column(String, ForeignKey("data_sources.id"))
    configuration = Column(JSON, nullable=False)  # JSON representing the report configuration
    creator_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    is_public = Column(Boolean, default=False)
    is_favorite = Column(Boolean, default=False)
    last_run_at = Column(DateTime(timezone=True), nullable=True)
    thumbnail = Column(String, nullable=True)  # URL or base64 thumbnail of the report

    # Relationships
    creator = relationship("User", backref="reports")
    data_source = relationship("DataSource", back_populates="reports")

# Association table for sharing reports with teams
report_team = Table(
    "report_team",
    Base.metadata,
    Column("report_id", String, ForeignKey("reports.id"), primary_key=True),
    Column("team_id", Integer, ForeignKey("teams.id"), primary_key=True),
)

# Report sharing permissions
class ReportPermission(Base):
    """Report sharing permissions"""
    __tablename__ = "report_permissions"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String, ForeignKey("reports.id"))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)
    permission_type = Column(String, default="view")  # view, edit, admin
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    report = relationship("Report", backref="permissions")
    user = relationship("User", backref="report_permissions")
    team = relationship("Team", backref="report_permissions")


class DataModel(Base):
    """Data model for storing database schema and relationships"""
    __tablename__ = "data_models"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    data_source_id = Column(String, ForeignKey("data_sources.id"))
    creator_id = Column(Integer, ForeignKey("users.id"))
    layout = Column(String, nullable=True)  # JSON string for storing canvas layout
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

    # Relationships
    data_source = relationship("DataSource", back_populates="data_models")
    creator = relationship("User", backref="data_models")
    tables = relationship("DataModelTable", back_populates="data_model", cascade="all, delete-orphan")
    relationships = relationship("DataModelRelationship", back_populates="data_model", cascade="all, delete-orphan")


class DataModelTable(Base):
    """Table in a data model"""
    __tablename__ = "data_model_tables"

    id = Column(String, primary_key=True)
    data_model_id = Column(String, ForeignKey("data_models.id"))
    table_name = Column(String, nullable=False)

    # Relationships
    data_model = relationship("DataModel", back_populates="tables")


class DataModelRelationship(Base):
    """Relationship between tables in a data model"""
    __tablename__ = "data_model_relationships"

    id = Column(String, primary_key=True)
    data_model_id = Column(String, ForeignKey("data_models.id"))
    source_table = Column(String, nullable=False)
    source_column = Column(String, nullable=False)
    target_table = Column(String, nullable=False)
    target_column = Column(String, nullable=False)
    relationship_type = Column(String, nullable=False)  # 'one-to-one', 'one-to-many', 'many-to-one', 'many-to-many'

    # Relationships
    data_model = relationship("DataModel", back_populates="relationships")