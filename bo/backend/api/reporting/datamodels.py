"""DataModel API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import uuid
from typing import List, Optional, Dict, Any
import json
import sqlalchemy
from pydantic import BaseModel

from db import database
from db.reporting import DataSource, DataModel, DataModelTable, DataModelRelationship
from auth.dependencies import get_current_active_user
from db.models import User
from db.connections import get_connection_for_data_source

router = APIRouter(
    prefix="",  # No prefix here since it's added in the main router
    tags=["reporting"],
)

# Pydantic models for API responses
class ColumnSchemaResponse(BaseModel):
    name: str
    type: str
    isPrimaryKey: bool = False
    isForeignKey: bool = False
    references: Optional[Dict[str, str]] = None

class TableSchemaResponse(BaseModel):
    name: str
    columns: List[ColumnSchemaResponse]

class RelationshipResponse(BaseModel):
    id: str
    sourceTable: str
    sourceColumn: str
    targetTable: str
    targetColumn: str
    type: str  # 'one-to-one', 'one-to-many', 'many-to-one', 'many-to-many'

class DataModelResponse(BaseModel):
    id: str
    name: str
    dataSourceId: str
    tables: List[str]
    relationships: List[RelationshipResponse]
    layout: Optional[str] = None  # JSON string for canvas layout
    createdAt: str
    updatedAt: str

class DataModelCreate(BaseModel):
    name: str
    dataSourceId: str
    tables: List[str]
    relationships: List[Dict[str, Any]]
    layout: Optional[str] = None  # JSON string for canvas layout

class DataModelUpdate(BaseModel):
    name: Optional[str] = None
    tables: Optional[List[str]] = None
    relationships: Optional[List[Dict[str, Any]]] = None
    layout: Optional[str] = None  # JSON string for canvas layout

# Helper function to get data model response
async def get_data_model_response(model_id: str, db: Session) -> DataModelResponse:
    """Get a data model response by ID"""
    db_model = db.query(DataModel).filter(DataModel.id == model_id).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="Data model not found")

    # Get tables
    db_tables = db.query(DataModelTable).filter(DataModelTable.data_model_id == model_id).all()
    tables = [table.table_name for table in db_tables]

    # Get relationships
    db_relationships = db.query(DataModelRelationship).filter(DataModelRelationship.data_model_id == model_id).all()
    relationships = [
        RelationshipResponse(
            id=rel.id,
            sourceTable=rel.source_table,
            sourceColumn=rel.source_column,
            targetTable=rel.target_table,
            targetColumn=rel.target_column,
            type=rel.relationship_type
        )
        for rel in db_relationships
    ]

    return DataModelResponse(
        id=db_model.id,
        name=db_model.name,
        dataSourceId=db_model.data_source_id,
        tables=tables,
        relationships=relationships,
        layout=db_model.layout,
        createdAt=db_model.created_at.isoformat(),
        updatedAt=db_model.updated_at.isoformat()
    )

@router.post("/datamodels", response_model=DataModelResponse)
async def create_data_model(
    model: DataModelCreate,
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new data model"""
    try:
        # Check if the data source exists and user has access
        data_source = db.query(DataSource).filter(DataSource.id == model.dataSourceId).first()
        if not data_source:
            raise HTTPException(status_code=404, detail="Data source not found")

        if current_user.role != "admin" and data_source.creator_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to create data models for this data source")

        # Create the data model
        model_id = str(uuid.uuid4())
        now = sqlalchemy.func.now()

        db_model = DataModel(
            id=model_id,
            name=model.name,
            data_source_id=model.dataSourceId,
            creator_id=current_user.id,
            layout=model.layout,
            created_at=now,
            updated_at=now
        )
        db.add(db_model)

        # Add tables
        for table_name in model.tables:
            db_table = DataModelTable(
                id=str(uuid.uuid4()),
                data_model_id=model_id,
                table_name=table_name
            )
            db.add(db_table)

        # Add relationships
        for rel in model.relationships:
            db_rel = DataModelRelationship(
                id=rel.get('id', str(uuid.uuid4())),
                data_model_id=model_id,
                source_table=rel['sourceTable'],
                source_column=rel['sourceColumn'],
                target_table=rel['targetTable'],
                target_column=rel['targetColumn'],
                relationship_type=rel['type']
            )
            db.add(db_rel)

        db.commit()

        # Return the created model
        return await get_data_model_response(db_model.id, db)

    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error creating data model: {str(e)}")
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in create_data_model: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to create data model: {str(e)}")

@router.get("/datamodels", response_model=List[DataModelResponse])
async def list_data_models(
    dataSourceId: Optional[str] = Query(None, title="Filter models by data source ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List data models"""
    try:
        query = db.query(DataModel)

        # Apply filters
        if dataSourceId:
            query = query.filter(DataModel.data_source_id == dataSourceId)

        # Filter by creator if not admin
        if current_user.role != "admin":
            query = query.filter(DataModel.creator_id == current_user.id)

        # Apply pagination
        query = query.offset(skip).limit(limit)

        # Get models
        db_models = query.all()

        # Convert to response models
        models = []
        for db_model in db_models:
            models.append(await get_data_model_response(db_model.id, db))

        return models

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in list_data_models: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to list data models: {str(e)}")

@router.get("/datamodels/{model_id}", response_model=DataModelResponse)
async def get_data_model(
    model_id: str = Path(..., title="The ID of the data model to get"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific data model by ID"""
    try:
        db_model = db.query(DataModel).filter(DataModel.id == model_id).first()
        if not db_model:
            raise HTTPException(status_code=404, detail="Data model not found")

        # Check permissions
        if current_user.role != "admin" and db_model.creator_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to access this data model")

        return await get_data_model_response(model_id, db)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in get_data_model: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to get data model: {str(e)}")

@router.put("/datamodels/{model_id}", response_model=DataModelResponse)
async def update_data_model(
    model_update: DataModelUpdate,
    model_id: str = Path(..., title="The ID of the data model to update"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update a data model"""
    try:
        db_model = db.query(DataModel).filter(DataModel.id == model_id).first()
        if not db_model:
            raise HTTPException(status_code=404, detail="Data model not found")

        # Check permissions
        if current_user.role != "admin" and db_model.creator_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to update this data model")

        # Update fields
        if model_update.name is not None:
            db_model.name = model_update.name

        # Update layout if provided
        if model_update.layout is not None:
            db_model.layout = model_update.layout

        db_model.updated_at = sqlalchemy.func.now()

        # Update tables if provided
        if model_update.tables is not None:
            # Delete existing tables
            db.query(DataModelTable).filter(DataModelTable.data_model_id == model_id).delete()

            # Add new tables
            for table_name in model_update.tables:
                db_table = DataModelTable(
                    id=str(uuid.uuid4()),
                    data_model_id=model_id,
                    table_name=table_name
                )
                db.add(db_table)

        # Update relationships if provided
        if model_update.relationships is not None:
            # Delete existing relationships
            db.query(DataModelRelationship).filter(DataModelRelationship.data_model_id == model_id).delete()

            # Add new relationships
            for rel in model_update.relationships:
                db_rel = DataModelRelationship(
                    id=rel.get('id', str(uuid.uuid4())),
                    data_model_id=model_id,
                    source_table=rel['sourceTable'],
                    source_column=rel['sourceColumn'],
                    target_table=rel['targetTable'],
                    target_column=rel['targetColumn'],
                    relationship_type=rel['type']
                )
                db.add(db_rel)

        db.commit()

        return await get_data_model_response(model_id, db)

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in update_data_model: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to update data model: {str(e)}")

@router.delete("/datamodels/{model_id}")
async def delete_data_model(
    model_id: str = Path(..., title="The ID of the data model to delete"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a data model"""
    try:
        db_model = db.query(DataModel).filter(DataModel.id == model_id).first()
        if not db_model:
            raise HTTPException(status_code=404, detail="Data model not found")

        # Check permissions
        if current_user.role != "admin" and db_model.creator_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to delete this data model")

        # Delete relationships
        db.query(DataModelRelationship).filter(DataModelRelationship.data_model_id == model_id).delete()

        # Delete tables
        db.query(DataModelTable).filter(DataModelTable.data_model_id == model_id).delete()

        # Delete model
        db.delete(db_model)

        db.commit()

        return {"message": "Data model deleted successfully"}

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in delete_data_model: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to delete data model: {str(e)}")

@router.post("/datamodels/{model_id}/generate-sql", response_model=Dict[str, str])
async def generate_sql_from_model(
    model_id: str = Path(..., title="The ID of the data model to generate SQL for"),
    options: Dict[str, Any] = Body(..., title="SQL generation options"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Generate SQL from a data model"""
    try:
        db_model = db.query(DataModel).filter(DataModel.id == model_id).first()
        if not db_model:
            raise HTTPException(status_code=404, detail="Data model not found")

        # Check permissions
        if current_user.role != "admin" and db_model.creator_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to access this data model")

        # Get data source
        data_source = db.query(DataSource).filter(DataSource.id == db_model.data_source_id).first()
        if not data_source:
            raise HTTPException(status_code=404, detail="Data source not found")

        # Get tables
        db_tables = db.query(DataModelTable).filter(DataModelTable.data_model_id == model_id).all()
        tables = [table.table_name for table in db_tables]

        # Get relationships
        db_relationships = db.query(DataModelRelationship).filter(DataModelRelationship.data_model_id == model_id).all()
        relationships = [
            {
                "id": rel.id,
                "sourceTable": rel.source_table,
                "sourceColumn": rel.source_column,
                "targetTable": rel.target_table,
                "targetColumn": rel.target_column,
                "type": rel.relationship_type
            }
            for rel in db_relationships
        ]

        # Get database connection
        conn = get_connection_for_data_source(data_source)

        # Generate SQL based on dialect
        dialect = options.get("dialect", "mysql")
        include_drop_table = options.get("includeDropTable", False)
        include_foreign_keys = options.get("includeForeignKeys", True)

        # Get table schemas
        table_schemas = {}
        for table_name in tables:
            try:
                # Get columns for the table
                if dialect == "mysql":
                    query = f"""
                        SELECT
                            COLUMN_NAME as name,
                            DATA_TYPE as type,
                            CASE WHEN COLUMN_KEY = 'PRI' THEN 1 ELSE 0 END as isPrimaryKey,
                            CASE WHEN COLUMN_KEY = 'MUL' THEN 1 ELSE 0 END as isForeignKey
                        FROM
                            INFORMATION_SCHEMA.COLUMNS
                        WHERE
                            TABLE_SCHEMA = '{conn.database}' AND
                            TABLE_NAME = '{table_name}'
                    """
                    cursor = conn.cursor()
                    cursor.execute(query)
                    columns = []
                    for row in cursor.fetchall():
                        columns.append({
                            "name": row[0],
                            "type": row[1],
                            "isPrimaryKey": bool(row[2]),
                            "isForeignKey": bool(row[3])
                        })
                    table_schemas[table_name] = {"name": table_name, "columns": columns}
            except Exception as e:
                print(f"Error getting schema for table {table_name}: {str(e)}")
                # Continue with next table

        # Generate SQL
        sql = []

        # Add DROP TABLE statements if requested
        if include_drop_table:
            for table_name in tables:
                sql.append(f"DROP TABLE IF EXISTS `{table_name}`;")
            sql.append("")  # Empty line

        # Add CREATE TABLE statements
        for table_name, schema in table_schemas.items():
            create_table = f"CREATE TABLE `{table_name}` (\n"

            # Add columns
            column_defs = []
            primary_keys = []

            for column in schema["columns"]:
                col_name = column["name"]
                col_type = column["type"]

                # Map MySQL types to the selected dialect if needed
                if dialect != "mysql":
                    # Add type mapping here if needed
                    pass

                col_def = f"  `{col_name}` {col_type.upper()}"

                # Add primary key constraint inline
                if column["isPrimaryKey"]:
                    col_def += " PRIMARY KEY"
                    primary_keys.append(col_name)

                column_defs.append(col_def)

            # Add primary key constraint if multiple columns
            if len(primary_keys) > 1:
                pk_constraint = f"  PRIMARY KEY (`{'`, `'.join(primary_keys)}`)"
                column_defs.append(pk_constraint)

            # Add foreign key constraints if requested
            if include_foreign_keys:
                for rel in relationships:
                    if rel["sourceTable"] == table_name:
                        fk_constraint = f"  FOREIGN KEY (`{rel['sourceColumn']}`) REFERENCES `{rel['targetTable']}` (`{rel['targetColumn']}`)"
                        column_defs.append(fk_constraint)

            create_table += ",\n".join(column_defs)
            create_table += "\n);"

            sql.append(create_table)
            sql.append("")  # Empty line

        # Return the generated SQL
        return {"sql": "\n".join(sql)}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in generate_sql_from_model: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to generate SQL: {str(e)}")
