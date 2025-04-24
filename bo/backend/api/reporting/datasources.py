"""DataSource API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import uuid
from typing import List, Optional, Dict, Any
import json
import pymysql
import sqlalchemy
import logging
from pydantic import BaseModel

from db import database
from db.reporting import DataSource
from auth.dependencies import get_current_active_user
from db.models import User
from .models import DataSourceCreate, DataSourceUpdate, DataSourceResponse

# Schema response models
class ColumnSchema(BaseModel):
    name: str
    type: str
    isPrimaryKey: bool = False
    isForeignKey: bool = False
    references: Optional[Dict[str, str]] = None

class TableSchema(BaseModel):
    name: str
    columns: List[ColumnSchema]

# Database connection utilities
def get_mysql_schema(config: Dict[str, Any]) -> List[TableSchema]:
    """Get schema from MySQL database"""
    try:
        # Extract connection parameters
        host = config.get("host", "localhost")
        port = config.get("port")
        port = int(port) if port else 3306
        database_name = config.get("database", "")
        username = config.get("username", "")
        password = config.get("password", "")
        ssl_enabled = config.get("ssl_enabled", False)

        # Validate required parameters
        if not database_name:
            raise ValueError("Database name is required")

        # Log connection attempt (without password)
        logger = logging.getLogger("agentor-backoffice.api.reporting.datasources")
        logger.info(f"Attempting to connect to MySQL database: {host}:{port}/{database_name} as {username}")

        # Create connection
        try:
            connection = pymysql.connect(
                host=host,
                user=username,
                password=password,
                database=database_name,
                port=port,
                ssl=ssl_enabled,
                cursorclass=pymysql.cursors.DictCursor,
                connect_timeout=5
            )
            logger.info(f"Successfully connected to MySQL database: {host}:{port}/{database_name}")
        except pymysql.MySQLError as e:
            error_code = e.args[0]
            error_message = e.args[1] if len(e.args) > 1 else str(e)

            if error_code == 1045:  # Access denied
                raise HTTPException(
                    status_code=500,
                    detail=f"Access denied for user '{username}'. Please check your username and password."
                )
            elif error_code == 1049:  # Unknown database
                raise HTTPException(
                    status_code=500,
                    detail=f"Database '{database_name}' does not exist."
                )
            elif error_code == 2003:  # Connection refused
                raise HTTPException(
                    status_code=500,
                    detail=f"Could not connect to MySQL server at {host}:{port}. Please check if the server is running and accessible."
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"MySQL connection error ({error_code}): {error_message}"
                )

        # List tables and their columns
        tables = []
        with connection.cursor() as cursor:
            # Get all tables
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s
                ORDER BY table_name
            """, (database_name,))
            table_rows = cursor.fetchall()

            # Get primary keys info
            cursor.execute("""
                SELECT k.table_name, k.column_name
                FROM information_schema.table_constraints t
                JOIN information_schema.key_column_usage k
                USING(constraint_name,table_schema,table_name)
                WHERE t.constraint_type='PRIMARY KEY' AND t.table_schema=%s
            """, (database_name,))
            primary_keys = {}
            for pk_row in cursor.fetchall():
                # Debug the actual keys in the result
                logger.debug(f"Primary key row keys: {pk_row.keys()}")

                # Handle different column name formats (MySQL might return TABLE_NAME instead of table_name)
                if 'table_name' in pk_row:
                    table_name = pk_row['table_name']
                    column_name = pk_row['column_name']
                elif 'TABLE_NAME' in pk_row:
                    table_name = pk_row['TABLE_NAME']
                    column_name = pk_row['COLUMN_NAME']
                else:
                    # If we can't find the expected keys, log the issue and skip this row
                    logger.error(f"Unexpected primary key row format: {pk_row}")
                    continue

                if table_name not in primary_keys:
                    primary_keys[table_name] = []
                primary_keys[table_name].append(column_name)

            # Get foreign keys info
            cursor.execute("""
                SELECT
                    k.table_name,
                    k.column_name,
                    k.referenced_table_name,
                    k.referenced_column_name
                FROM information_schema.key_column_usage k
                JOIN information_schema.table_constraints t
                ON k.constraint_name = t.constraint_name
                WHERE t.constraint_type = 'FOREIGN KEY'
                AND k.table_schema = %s
                AND k.referenced_table_name IS NOT NULL
            """, (database_name,))
            foreign_keys = {}
            for fk_row in cursor.fetchall():
                # Debug the actual keys in the result
                logger.debug(f"Foreign key row keys: {fk_row.keys()}")

                # Handle different column name formats
                if 'table_name' in fk_row and 'column_name' in fk_row:
                    table_name = fk_row['table_name']
                    column_name = fk_row['column_name']
                    ref_table = fk_row['referenced_table_name']
                    ref_column = fk_row['referenced_column_name']
                elif 'TABLE_NAME' in fk_row and 'COLUMN_NAME' in fk_row:
                    table_name = fk_row['TABLE_NAME']
                    column_name = fk_row['COLUMN_NAME']
                    ref_table = fk_row['REFERENCED_TABLE_NAME']
                    ref_column = fk_row['REFERENCED_COLUMN_NAME']
                else:
                    # If we can't find the expected keys, log the issue and skip this row
                    logger.error(f"Unexpected foreign key row format: {fk_row}")
                    continue

                if table_name not in foreign_keys:
                    foreign_keys[table_name] = {}
                foreign_keys[table_name][column_name] = {
                    'table': ref_table,
                    'column': ref_column
                }

            # Process each table
            for table_row in table_rows:
                # Handle different column name formats
                if 'table_name' in table_row:
                    table_name = table_row['table_name']
                elif 'TABLE_NAME' in table_row:
                    table_name = table_row['TABLE_NAME']
                else:
                    # If we can't find the expected keys, log the issue and skip this row
                    logger.error(f"Unexpected table row format: {table_row}")
                    continue

                # Get columns for this table
                cursor.execute("""
                    SELECT column_name, data_type, column_key
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                """, (database_name, table_name))

                # Debug the column query results
                logger.debug(f"Executing column query for table: {table_name}")
                columns = []

                for column_row in cursor.fetchall():
                    # Debug the actual keys in the result
                    logger.debug(f"Column row keys: {column_row.keys()}")

                    # Handle different column name formats
                    if 'column_name' in column_row and 'data_type' in column_row:
                        column_name = column_row['column_name']
                        data_type = column_row['data_type']
                    elif 'COLUMN_NAME' in column_row and 'DATA_TYPE' in column_row:
                        column_name = column_row['COLUMN_NAME']
                        data_type = column_row['DATA_TYPE']
                    else:
                        # If we can't find the expected keys, log the issue and skip this row
                        logger.error(f"Unexpected column row format: {column_row}")
                        continue

                    # Check if primary key
                    is_primary_key = table_name in primary_keys and column_name in primary_keys[table_name]

                    # Check if foreign key and get references
                    is_foreign_key = table_name in foreign_keys and column_name in foreign_keys[table_name]
                    references = None
                    if is_foreign_key:
                        references = foreign_keys[table_name][column_name]

                    # Add column to list
                    columns.append(ColumnSchema(
                        name=column_name,
                        type=data_type,
                        isPrimaryKey=is_primary_key,
                        isForeignKey=is_foreign_key,
                        references=references
                    ))

                # Add table with its columns to the list
                tables.append(TableSchema(name=table_name, columns=columns))

        connection.close()
        return tables

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error getting MySQL schema: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error connecting to MySQL database: {str(e)}")

def get_database_schema(data_source: DataSource) -> List[TableSchema]:
    """Get schema from a data source based on its connection type"""
    try:
        conn_type = data_source.connection_type
        conn_config = data_source.connection_config

        if isinstance(conn_config, str):
            # Handle case where config might be stored as a string
            conn_config = json.loads(conn_config)

        if conn_type.lower() == "mysql":
            return get_mysql_schema(conn_config)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported connection type: {conn_type}. Currently only MySQL is supported."
            )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error getting database schema: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error getting database schema: {str(e)}")

router = APIRouter(
    prefix="",  # No prefix here since it's added in the main router
    tags=["reporting"],
)

@router.post("/datasources/", response_model=DataSourceResponse)
def create_data_source(
    data_source: DataSourceCreate,
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new data source"""
    try:
        db_data_source = DataSource(
            id=str(uuid.uuid4()),
            **data_source.model_dump(),
            creator_id=current_user.id
        )
        db.add(db_data_source)
        db.commit()
        db.refresh(db_data_source)
        return db_data_source
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error creating data source: {str(e)}")
    except Exception as e:
        db.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in create_data_source: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/datasources/", response_model=List[DataSourceResponse])
def list_data_sources(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List data sources"""
    try:
        # If admin, return all data sources, otherwise filter by creator
        if current_user.role == "admin":
            db_data_sources = db.query(DataSource).offset(skip).limit(limit).all()
        else:
            db_data_sources = db.query(DataSource).filter(DataSource.creator_id == current_user.id).offset(skip).limit(limit).all()
        return db_data_sources
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in list_data_sources: {str(e)}")
        print(f"Error details: {error_details}")
        return []

@router.get("/datasources/{data_source_id}", response_model=DataSourceResponse)
def get_data_source(
    data_source_id: str = Path(..., title="The ID of the data source to get"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific data source by ID"""
    try:
        db_data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
        if db_data_source is None:
            raise HTTPException(status_code=404, detail="Data source not found")

        # Check permissions (admin or creator)
        if current_user.role != "admin" and db_data_source.creator_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to access this data source")

        return db_data_source
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in get_data_source: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/datasources/{data_source_id}", response_model=DataSourceResponse)
def update_data_source(
    data_source_update: DataSourceUpdate,
    data_source_id: str = Path(..., title="The ID of the data source to update"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update a data source"""
    try:
        db_data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
        if db_data_source is None:
            raise HTTPException(status_code=404, detail="Data source not found")

        # Check permissions (admin or creator)
        if current_user.role != "admin" and db_data_source.creator_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to update this data source")

        # Update fields
        update_data = data_source_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_data_source, field, value)

        db.commit()
        db.refresh(db_data_source)
        return db_data_source
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error updating data source: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in update_data_source: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/datasources/{data_source_id}", response_model=dict)
def delete_data_source(
    data_source_id: str = Path(..., title="The ID of the data source to delete"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a data source"""
    try:
        db_data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
        if db_data_source is None:
            raise HTTPException(status_code=404, detail="Data source not found")

        # Check permissions (admin or creator)
        if current_user.role != "admin" and db_data_source.creator_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to delete this data source")

        db.delete(db_data_source)
        db.commit()
        return {"message": "Data source deleted successfully"}
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error deleting data source: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in delete_data_source: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/datasources/{data_source_id}/schema", response_model=List[TableSchema])
def get_data_source_schema(
    data_source_id: str = Path(..., title="The ID of the data source to get schema for"),
    db: Session = Depends(database.get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get the database schema (tables and columns) for a specific data source"""
    try:
        # Get the data source
        db_data_source = db.query(DataSource).filter(DataSource.id == data_source_id).first()
        if db_data_source is None:
            raise HTTPException(status_code=404, detail="Data source not found")

        # Check permissions (admin or creator)
        if current_user.role != "admin" and db_data_source.creator_id != current_user.id:
            raise HTTPException(status_code=403, detail="Not authorized to access this data source")

        # Get schema from the data source
        return get_database_schema(db_data_source)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in get_data_source_schema: {str(e)}")
        print(f"Error details: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error getting database schema: {str(e)}")
