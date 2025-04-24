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
    createdAt: str
    updatedAt: str

class DataModelCreate(BaseModel):
    name: str
    dataSourceId: str
    tables: List[str]
    relationships: List[Dict[str, Any]]

class DataModelUpdate(BaseModel):
    name: Optional[str] = None
    tables: Optional[List[str]] = None
    relationships: Optional[List[Dict[str, Any]]] = None

@router.get("/datasources/{data_source_id}/schema", response_model=List[TableSchemaResponse])
async def get_database_schema(
    data_source_id: str = Path(..., title="The ID of the data source"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get the database schema (tables and columns) for a data source"""
    try:
        # Get the data source
        data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
        if not data_source:
            raise HTTPException(status_code=404, detail="Data source not found")
        
        # Check permissions
        if current_user.role != "admin" and data_source.creator_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to access this data source")
        
        # Get connection for the data source
        connection = get_connection_for_data_source(data_source)
        
        # Fetch schema information based on data source type
        if data_source.connection_type in ['mysql', 'postgresql']:
            result = await fetch_sql_schema(connection, data_source.connection_type)
        elif data_source.connection_type == 'sqlite':
            result = await fetch_sqlite_schema(connection)
        else:
            raise HTTPException(status_code=400, detail=f"Schema introspection not supported for {data_source.connection_type}")
        
        return result
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in get_database_schema: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to get database schema: {str(e)}")

async def fetch_sql_schema(connection, db_type):
    """Fetch schema for MySQL or PostgreSQL databases"""
    tables = []
    
    try:
        # Get list of tables
        if db_type == 'mysql':
            table_query = "SHOW TABLES"
            table_names = [row[0] for row in connection.execute(table_query).fetchall()]
        else:  # PostgreSQL
            table_query = "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public'"
            table_names = [row[0] for row in connection.execute(table_query).fetchall()]
        
        # For each table, get columns and keys
        for table_name in table_names:
            columns = []
            
            # Get columns
            if db_type == 'mysql':
                column_query = f"SHOW COLUMNS FROM `{table_name}`"
                column_data = connection.execute(column_query).fetchall()
                
                # Get primary keys
                pk_query = f"""
                    SELECT k.COLUMN_NAME
                    FROM information_schema.table_constraints t
                    JOIN information_schema.key_column_usage k
                    USING(constraint_name,table_schema,table_name)
                    WHERE t.constraint_type='PRIMARY KEY'
                    AND t.table_name='{table_name}'
                """
                primary_keys = [row[0] for row in connection.execute(pk_query).fetchall()]
                
                # Get foreign keys
                fk_query = f"""
                    SELECT 
                        k.COLUMN_NAME, 
                        k.REFERENCED_TABLE_NAME, 
                        k.REFERENCED_COLUMN_NAME
                    FROM information_schema.table_constraints t
                    JOIN information_schema.key_column_usage k
                    USING(constraint_name,table_schema,table_name)
                    WHERE t.constraint_type='FOREIGN KEY'
                    AND t.table_name='{table_name}'
                """
                foreign_keys = {row[0]: {"table": row[1], "column": row[2]} 
                                for row in connection.execute(fk_query).fetchall()}
                
                for col in column_data:
                    column_name = col[0]
                    column_type = col[1]
                    is_primary = column_name in primary_keys
                    is_foreign = column_name in foreign_keys
                    references = foreign_keys.get(column_name) if is_foreign else None
                    
                    columns.append(ColumnSchemaResponse(
                        name=column_name,
                        type=column_type,
                        isPrimaryKey=is_primary,
                        isForeignKey=is_foreign,
                        references=references
                    ))
            
            else:  # PostgreSQL
                column_query = f"""
                    SELECT 
                        column_name, 
                        data_type
                    FROM information_schema.columns
                    WHERE table_name = '{table_name}'
                """
                column_data = connection.execute(column_query).fetchall()
                
                # Get primary keys
                pk_query = f"""
                    SELECT a.attname
                    FROM pg_index i
                    JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                    WHERE i.indrelid = '{table_name}'::regclass AND i.indisprimary
                """
                primary_keys = [row[0] for row in connection.execute(pk_query).fetchall()]
                
                # Get foreign keys
                fk_query = f"""
                    SELECT
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM
                        information_schema.table_constraints AS tc
                        JOIN information_schema.key_column_usage AS kcu
                          ON tc.constraint_name = kcu.constraint_name
                        JOIN information_schema.constraint_column_usage AS ccu
                          ON ccu.constraint_name = tc.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = '{table_name}'
                """
                foreign_keys = {row[0]: {"table": row[1], "column": row[2]} 
                                for row in connection.execute(fk_query).fetchall()}
                
                for col in column_data:
                    column_name = col[0]
                    column_type = col[1]
                    is_primary = column_name in primary_keys
                    is_foreign = column_name in foreign_keys
                    references = foreign_keys.get(column_name) if is_foreign else None
                    
                    columns.append(ColumnSchemaResponse(
                        name=column_name,
                        type=column_type,
                        isPrimaryKey=is_primary,
                        isForeignKey=is_foreign,
                        references=references
                    ))
            
            tables.append(TableSchemaResponse(
                name=table_name,
                columns=columns
            ))
        
        return tables
    
    except Exception as e:
        print(f"Error fetching SQL schema: {str(e)}")
        raise

async def fetch_sqlite_schema(connection):
    """Fetch schema for SQLite databases"""
    tables = []
    
    try:
        # Get list of tables
        table_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        table_names = [row[0] for row in connection.execute(table_query).fetchall()]
        
        # For each table, get columns and keys
        for table_name in table_names:
            columns = []
            
            # Get columns and their types
            pragma_query = f"PRAGMA table_info('{table_name}')"
            column_data = connection.execute(pragma_query).fetchall()
            
            # Get foreign keys
            fk_query = f"PRAGMA foreign_key_list('{table_name}')"
            foreign_keys = {row[3]: {"table": row[2], "column": row[4]} 
                           for row in connection.execute(fk_query).fetchall()}
            
            for col in column_data:
                column_name = col[1]
                column_type = col[2]
                is_primary = bool(col[5])  # The primary key flag (1 if it's part of the PK)
                is_foreign = column_name in foreign_keys
                references = foreign_keys.get(column_name) if is_foreign else None
                
                columns.append(ColumnSchemaResponse(
                    name=column_name,
                    type=column_type,
                    isPrimaryKey=is_primary,
                    isForeignKey=is_foreign,
                    references=references
                ))
            
            tables.append(TableSchemaResponse(
                name=table_name,
                columns=columns
            ))
        
        return tables
    
    except Exception as e:
        print(f"Error fetching SQLite schema: {str(e)}")
        raise

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
        models = query.all()
        
        # Convert to response format
        result = []
        for model in models:
            model_response = await get_data_model_response(model.id, db)
            result.append(model_response)
        
        return result
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in list_data_models: {str(e)}")
        print(f"Error details: {error_details}")
        return []

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
    
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error updating data model: {str(e)}")
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

@router.delete("/datamodels/{model_id}", response_model=dict)
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
        
        # Delete related tables and relationships
        db.query(DataModelTable).filter(DataModelTable.data_model_id == model_id).delete()
        db.query(DataModelRelationship).filter(DataModelRelationship.data_model_id == model_id).delete()
        
        # Delete the model
        db.delete(db_model)
        db.commit()
        
        return {"message": "Data model deleted successfully"}
    
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error deleting data model: {str(e)}")
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

@router.post("/datasources/{data_source_id}/test-query", response_model=Dict[str, Any])
async def test_query(
    data_source_id: str = Path(..., title="The ID of the data source"),
    query_data: Dict[str, str] = Body(..., title="The query to test"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Test a SQL query on a data source"""
    try:
        # Get the data source
        data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
        if not data_source:
            raise HTTPException(status_code=404, detail="Data source not found")
        
        # Check permissions
        if current_user.role != "admin" and data_source.creator_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to access this data source")
        
        # Get the query
        query = query_data.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Validate query (only SELECT queries allowed for testing)
        if not query.strip().upper().startswith("SELECT"):
            raise HTTPException(status_code=400, detail="Only SELECT queries are allowed for testing")
        
        # Get connection for the data source
        connection = get_connection_for_data_source(data_source)
        
        # Execute the query
        result = connection.execute(query)
        rows = result.fetchall()
        columns = result.keys()
        
        # Convert to JSON-serializable format
        result_data = []
        for row in rows:
            result_data.append(dict(zip(columns, row)))
        
        return {
            "columns": list(columns),
            "rows": result_data,
            "rowCount": len(rows)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in test_query: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail=f"Failed to execute query: {str(e)}")

async def get_data_model_response(model_id: str, db: Session) -> DataModelResponse:
    """Helper function to convert a data model to the response format"""
    try:
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
            createdAt=db_model.created_at.isoformat() if db_model.created_at else None,
            updatedAt=db_model.updated_at.isoformat() if db_model.updated_at else None
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in get_data_model_response: {str(e)}")
        print(f"Error details: {error_details}")
        raise