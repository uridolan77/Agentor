"""
Document store interfaces for the Agentor framework.

This module provides interfaces for interacting with document-oriented NoSQL databases,
such as MongoDB, Firestore, and CouchDB.
"""

from typing import Any, Dict, List, Optional, Union, TypeVar
import time
import json
import re

from .base import (
    DatabaseConnection, DatabaseResult,
    ConnectionError, QueryError, TransactionError
)
from .nosql import NoSqlType, NoSqlConnection
from agentor.utils.logging import get_logger

logger = get_logger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Document type


class DocumentStore(NoSqlConnection):
    """Connection to a document-oriented NoSQL database."""

    def __init__(
        self,
        name: str,
        connection_string: str,
        database_name: str,
        **kwargs
    ):
        super().__init__(name=name, connection_string=connection_string, db_type=NoSqlType.DOCUMENT, **kwargs)
        self.database_name = database_name
        self.client = None
        self.db = None

    async def connect(self) -> DatabaseResult:
        """Connect to the document store."""
        try:
            # Determine the document store type from the connection string
            if self.connection_string.startswith("mongodb"):
                return await self._connect_mongodb()
            elif self.connection_string.startswith("firestore"):
                return await self._connect_firestore()
            else:
                return DatabaseResult.error_result(f"Unsupported document store type: {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to connect to document store: {self.name}, error: {e}")
            return DatabaseResult.error_result(ConnectionError(f"Failed to connect to document store: {e}"))

    async def _connect_mongodb(self) -> DatabaseResult:
        """Connect to a MongoDB database."""
        try:
            import motor.motor_asyncio
            self.client = motor.motor_asyncio.AsyncIOMotorClient(self.connection_string)
            self.db = self.client[self.database_name]
            
            # Test the connection
            await self.db.command("ping")
            
            self.connected = True
            self.last_activity = time.time()
            logger.info(f"Connected to MongoDB document store: {self.name}")
            return DatabaseResult.success_result()
        except ImportError:
            return DatabaseResult.error_result(
                "motor package is not installed. Install it with 'pip install motor'"
            )
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {self.name}, error: {e}")
            return DatabaseResult.error_result(ConnectionError(f"Failed to connect to MongoDB: {e}"))

    async def _connect_firestore(self) -> DatabaseResult:
        """Connect to a Firestore database."""
        try:
            from google.cloud import firestore
            
            # Parse the connection string to extract project ID
            # Format: firestore://project_id
            match = re.match(r"firestore://([^/]+)", self.connection_string)
            if not match:
                return DatabaseResult.error_result("Invalid Firestore connection string")
            
            project_id = match.group(1)
            self.client = firestore.AsyncClient(project=project_id)
            self.db = self.client
            
            self.connected = True
            self.last_activity = time.time()
            logger.info(f"Connected to Firestore document store: {self.name}")
            return DatabaseResult.success_result()
        except ImportError:
            return DatabaseResult.error_result(
                "google-cloud-firestore package is not installed. Install it with 'pip install google-cloud-firestore'"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Firestore: {self.name}, error: {e}")
            return DatabaseResult.error_result(ConnectionError(f"Failed to connect to Firestore: {e}"))

    async def disconnect(self) -> DatabaseResult:
        """Disconnect from the document store."""
        try:
            if self.client:
                if self.connection_string.startswith("mongodb"):
                    self.client.close()
                elif self.connection_string.startswith("firestore"):
                    await self.client.close()
            
            self.connected = False
            self.client = None
            self.db = None
            logger.info(f"Disconnected from document store: {self.name}")
            return DatabaseResult.success_result()
        except Exception as e:
            logger.error(f"Failed to disconnect from document store: {self.name}, error: {e}")
            return DatabaseResult.error_result(ConnectionError(f"Failed to disconnect from document store: {e}"))

    async def get_collection(self, collection_name: str) -> Any:
        """Get a collection from the document store."""
        if not self.connected:
            raise ConnectionError("Not connected to document store")
        
        if self.connection_string.startswith("mongodb"):
            return self.db[collection_name]
        elif self.connection_string.startswith("firestore"):
            return self.db.collection(collection_name)
        else:
            raise ValueError(f"Unsupported document store type: {self.connection_string}")

    async def insert_one(self, collection_name: str, document: Dict[str, Any]) -> DatabaseResult:
        """Insert a single document into the document store."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to document store"))
        
        try:
            self.last_activity = time.time()
            
            if self.connection_string.startswith("mongodb"):
                collection = await self.get_collection(collection_name)
                result = await collection.insert_one(document)
                return DatabaseResult.success_result(
                    data=str(result.inserted_id),
                    affected_rows=1,
                    last_insert_id=str(result.inserted_id)
                )
            elif self.connection_string.startswith("firestore"):
                collection = await self.get_collection(collection_name)
                doc_ref = collection.document()
                await doc_ref.set(document)
                return DatabaseResult.success_result(
                    data=doc_ref.id,
                    affected_rows=1,
                    last_insert_id=doc_ref.id
                )
            else:
                return DatabaseResult.error_result(f"Unsupported document store type: {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to insert document into {collection_name}: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to insert document: {e}"))

    async def insert_many(self, collection_name: str, documents: List[Dict[str, Any]]) -> DatabaseResult:
        """Insert multiple documents into the document store."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to document store"))
        
        try:
            self.last_activity = time.time()
            
            if self.connection_string.startswith("mongodb"):
                collection = await self.get_collection(collection_name)
                result = await collection.insert_many(documents)
                return DatabaseResult.success_result(
                    data=[str(id) for id in result.inserted_ids],
                    affected_rows=len(result.inserted_ids)
                )
            elif self.connection_string.startswith("firestore"):
                collection = await self.get_collection(collection_name)
                batch = self.db.batch()
                doc_refs = []
                
                for document in documents:
                    doc_ref = collection.document()
                    batch.set(doc_ref, document)
                    doc_refs.append(doc_ref)
                
                await batch.commit()
                return DatabaseResult.success_result(
                    data=[ref.id for ref in doc_refs],
                    affected_rows=len(doc_refs)
                )
            else:
                return DatabaseResult.error_result(f"Unsupported document store type: {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to insert documents into {collection_name}: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to insert documents: {e}"))

    async def find_one(self, collection_name: str, filter_dict: Dict[str, Any]) -> DatabaseResult:
        """Find a single document in the document store."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to document store"))
        
        try:
            self.last_activity = time.time()
            
            if self.connection_string.startswith("mongodb"):
                collection = await self.get_collection(collection_name)
                document = await collection.find_one(filter_dict)
                if document:
                    # Convert ObjectId to string for JSON serialization
                    document["_id"] = str(document["_id"])
                return DatabaseResult.success_result(data=document)
            elif self.connection_string.startswith("firestore"):
                collection = await self.get_collection(collection_name)
                query = collection
                
                for field, value in filter_dict.items():
                    query = query.where(field, "==", value)
                
                docs = await query.limit(1).get()
                if docs:
                    doc = docs[0]
                    document = doc.to_dict()
                    document["id"] = doc.id
                    return DatabaseResult.success_result(data=document)
                else:
                    return DatabaseResult.success_result(data=None)
            else:
                return DatabaseResult.error_result(f"Unsupported document store type: {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to find document in {collection_name}: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to find document: {e}"))

    async def find_many(self, collection_name: str, filter_dict: Dict[str, Any], limit: int = 0, skip: int = 0) -> DatabaseResult:
        """Find multiple documents in the document store."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to document store"))
        
        try:
            self.last_activity = time.time()
            
            if self.connection_string.startswith("mongodb"):
                collection = await self.get_collection(collection_name)
                cursor = collection.find(filter_dict)
                
                if skip > 0:
                    cursor = cursor.skip(skip)
                if limit > 0:
                    cursor = cursor.limit(limit)
                
                documents = await cursor.to_list(length=None)
                
                # Convert ObjectId to string for JSON serialization
                for document in documents:
                    document["_id"] = str(document["_id"])
                
                return DatabaseResult.success_result(data=documents)
            elif self.connection_string.startswith("firestore"):
                collection = await self.get_collection(collection_name)
                query = collection
                
                for field, value in filter_dict.items():
                    query = query.where(field, "==", value)
                
                if limit > 0:
                    query = query.limit(limit)
                
                # Firestore doesn't have a native skip, so we need to implement it manually
                docs = await query.get()
                
                if skip > 0:
                    docs = docs[skip:]
                
                documents = []
                for doc in docs:
                    document = doc.to_dict()
                    document["id"] = doc.id
                    documents.append(document)
                
                return DatabaseResult.success_result(data=documents)
            else:
                return DatabaseResult.error_result(f"Unsupported document store type: {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to find documents in {collection_name}: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to find documents: {e}"))

    async def update_one(self, collection_name: str, filter_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> DatabaseResult:
        """Update a single document in the document store."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to document store"))
        
        try:
            self.last_activity = time.time()
            
            if self.connection_string.startswith("mongodb"):
                collection = await self.get_collection(collection_name)
                result = await collection.update_one(filter_dict, {"$set": update_dict})
                return DatabaseResult.success_result(
                    affected_rows=result.modified_count,
                    metadata={"matched_count": result.matched_count}
                )
            elif self.connection_string.startswith("firestore"):
                collection = await self.get_collection(collection_name)
                query = collection
                
                for field, value in filter_dict.items():
                    query = query.where(field, "==", value)
                
                docs = await query.limit(1).get()
                if docs:
                    doc = docs[0]
                    await doc.reference.update(update_dict)
                    return DatabaseResult.success_result(affected_rows=1)
                else:
                    return DatabaseResult.success_result(affected_rows=0)
            else:
                return DatabaseResult.error_result(f"Unsupported document store type: {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to update document in {collection_name}: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to update document: {e}"))

    async def update_many(self, collection_name: str, filter_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> DatabaseResult:
        """Update multiple documents in the document store."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to document store"))
        
        try:
            self.last_activity = time.time()
            
            if self.connection_string.startswith("mongodb"):
                collection = await self.get_collection(collection_name)
                result = await collection.update_many(filter_dict, {"$set": update_dict})
                return DatabaseResult.success_result(
                    affected_rows=result.modified_count,
                    metadata={"matched_count": result.matched_count}
                )
            elif self.connection_string.startswith("firestore"):
                collection = await self.get_collection(collection_name)
                query = collection
                
                for field, value in filter_dict.items():
                    query = query.where(field, "==", value)
                
                docs = await query.get()
                batch = self.db.batch()
                
                for doc in docs:
                    batch.update(doc.reference, update_dict)
                
                await batch.commit()
                return DatabaseResult.success_result(affected_rows=len(docs))
            else:
                return DatabaseResult.error_result(f"Unsupported document store type: {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to update documents in {collection_name}: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to update documents: {e}"))

    async def delete_one(self, collection_name: str, filter_dict: Dict[str, Any]) -> DatabaseResult:
        """Delete a single document from the document store."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to document store"))
        
        try:
            self.last_activity = time.time()
            
            if self.connection_string.startswith("mongodb"):
                collection = await self.get_collection(collection_name)
                result = await collection.delete_one(filter_dict)
                return DatabaseResult.success_result(affected_rows=result.deleted_count)
            elif self.connection_string.startswith("firestore"):
                collection = await self.get_collection(collection_name)
                query = collection
                
                for field, value in filter_dict.items():
                    query = query.where(field, "==", value)
                
                docs = await query.limit(1).get()
                if docs:
                    doc = docs[0]
                    await doc.reference.delete()
                    return DatabaseResult.success_result(affected_rows=1)
                else:
                    return DatabaseResult.success_result(affected_rows=0)
            else:
                return DatabaseResult.error_result(f"Unsupported document store type: {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to delete document from {collection_name}: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to delete document: {e}"))

    async def delete_many(self, collection_name: str, filter_dict: Dict[str, Any]) -> DatabaseResult:
        """Delete multiple documents from the document store."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to document store"))
        
        try:
            self.last_activity = time.time()
            
            if self.connection_string.startswith("mongodb"):
                collection = await self.get_collection(collection_name)
                result = await collection.delete_many(filter_dict)
                return DatabaseResult.success_result(affected_rows=result.deleted_count)
            elif self.connection_string.startswith("firestore"):
                collection = await self.get_collection(collection_name)
                query = collection
                
                for field, value in filter_dict.items():
                    query = query.where(field, "==", value)
                
                docs = await query.get()
                batch = self.db.batch()
                
                for doc in docs:
                    batch.delete(doc.reference)
                
                await batch.commit()
                return DatabaseResult.success_result(affected_rows=len(docs))
            else:
                return DatabaseResult.error_result(f"Unsupported document store type: {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to delete documents from {collection_name}: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to delete documents: {e}"))

    # Implement the abstract methods from DatabaseConnection
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Execute a query on the document store.
        
        For document stores, this is typically a JSON query language like MongoDB's query language.
        """
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to document store"))
        
        try:
            self.last_activity = time.time()
            params = params or {}
            
            # Parse the query to determine the operation and collection
            query_data = json.loads(query)
            operation = query_data.get("operation")
            collection_name = query_data.get("collection")
            
            if not operation or not collection_name:
                return DatabaseResult.error_result("Query must include 'operation' and 'collection'")
            
            if operation == "find_one":
                return await self.find_one(collection_name, query_data.get("filter", {}))
            elif operation == "find_many":
                return await self.find_many(
                    collection_name,
                    query_data.get("filter", {}),
                    query_data.get("limit", 0),
                    query_data.get("skip", 0)
                )
            elif operation == "insert_one":
                return await self.insert_one(collection_name, query_data.get("document", {}))
            elif operation == "insert_many":
                return await self.insert_many(collection_name, query_data.get("documents", []))
            elif operation == "update_one":
                return await self.update_one(
                    collection_name,
                    query_data.get("filter", {}),
                    query_data.get("update", {})
                )
            elif operation == "update_many":
                return await self.update_many(
                    collection_name,
                    query_data.get("filter", {}),
                    query_data.get("update", {})
                )
            elif operation == "delete_one":
                return await self.delete_one(collection_name, query_data.get("filter", {}))
            elif operation == "delete_many":
                return await self.delete_many(collection_name, query_data.get("filter", {}))
            else:
                return DatabaseResult.error_result(f"Unsupported operation: {operation}")
        except json.JSONDecodeError:
            return DatabaseResult.error_result("Invalid JSON query")
        except Exception as e:
            logger.error(f"Failed to execute query on document store: {self.name}, error: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to execute query: {e}"))

    async def execute_many(self, query: str, params_list: List[Dict[str, Any]]) -> DatabaseResult:
        """Execute a query multiple times with different parameters."""
        results = []
        total_affected = 0
        
        for params in params_list:
            result = await self.execute(query, params)
            results.append(result)
            if result.success:
                total_affected += result.affected_rows
        
        return DatabaseResult.success_result(
            data=results,
            affected_rows=total_affected
        )

    async def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch a single document from the document store."""
        try:
            # Parse the query to determine the collection and filter
            query_data = json.loads(query)
            collection_name = query_data.get("collection")
            filter_dict = query_data.get("filter", {})
            
            if not collection_name:
                return DatabaseResult.error_result("Query must include 'collection'")
            
            return await self.find_one(collection_name, filter_dict)
        except json.JSONDecodeError:
            return DatabaseResult.error_result("Invalid JSON query")
        except Exception as e:
            logger.error(f"Failed to fetch one document: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to fetch one document: {e}"))

    async def fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch all documents from the document store."""
        try:
            # Parse the query to determine the collection and filter
            query_data = json.loads(query)
            collection_name = query_data.get("collection")
            filter_dict = query_data.get("filter", {})
            limit = query_data.get("limit", 0)
            skip = query_data.get("skip", 0)
            
            if not collection_name:
                return DatabaseResult.error_result("Query must include 'collection'")
            
            return await self.find_many(collection_name, filter_dict, limit, skip)
        except json.JSONDecodeError:
            return DatabaseResult.error_result("Invalid JSON query")
        except Exception as e:
            logger.error(f"Failed to fetch all documents: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to fetch all documents: {e}"))

    async def fetch_value(self, query: str, params: Optional[Dict[str, Any]] = None) -> DatabaseResult:
        """Fetch a single value from the document store."""
        result = await self.fetch_one(query, params)
        
        if not result.success or result.data is None:
            return result
        
        # Parse the query to determine the field to extract
        try:
            query_data = json.loads(query)
            field = query_data.get("field")
            
            if not field:
                return DatabaseResult.error_result("Query must include 'field' to extract a single value")
            
            if field in result.data:
                return DatabaseResult.success_result(data=result.data[field])
            else:
                return DatabaseResult.error_result(f"Field '{field}' not found in document")
        except json.JSONDecodeError:
            return DatabaseResult.error_result("Invalid JSON query")
        except Exception as e:
            logger.error(f"Failed to fetch value: {e}")
            return DatabaseResult.error_result(QueryError(f"Failed to fetch value: {e}"))

    async def begin_transaction(self) -> DatabaseResult:
        """Begin a transaction.
        
        Note: Not all document stores support transactions.
        """
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to document store"))
        
        try:
            self.last_activity = time.time()
            
            if self.connection_string.startswith("mongodb"):
                self.session = await self.client.start_session()
                self.session.start_transaction()
                self.in_transaction = True
                return DatabaseResult.success_result()
            elif self.connection_string.startswith("firestore"):
                self.transaction = self.db.transaction()
                await self.transaction.begin()
                self.in_transaction = True
                return DatabaseResult.success_result()
            else:
                return DatabaseResult.error_result(f"Unsupported document store type: {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to begin transaction on document store: {self.name}, error: {e}")
            return DatabaseResult.error_result(TransactionError(f"Failed to begin transaction: {e}"))

    async def commit_transaction(self) -> DatabaseResult:
        """Commit a transaction."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to document store"))
        
        if not self.in_transaction:
            return DatabaseResult.error_result(TransactionError("No transaction in progress"))
        
        try:
            self.last_activity = time.time()
            
            if self.connection_string.startswith("mongodb"):
                await self.session.commit_transaction()
                await self.session.end_session()
                self.session = None
                self.in_transaction = False
                return DatabaseResult.success_result()
            elif self.connection_string.startswith("firestore"):
                await self.transaction.commit()
                self.transaction = None
                self.in_transaction = False
                return DatabaseResult.success_result()
            else:
                return DatabaseResult.error_result(f"Unsupported document store type: {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to commit transaction on document store: {self.name}, error: {e}")
            return DatabaseResult.error_result(TransactionError(f"Failed to commit transaction: {e}"))

    async def rollback_transaction(self) -> DatabaseResult:
        """Rollback a transaction."""
        if not self.connected:
            return DatabaseResult.error_result(ConnectionError("Not connected to document store"))
        
        if not self.in_transaction:
            return DatabaseResult.error_result(TransactionError("No transaction in progress"))
        
        try:
            self.last_activity = time.time()
            
            if self.connection_string.startswith("mongodb"):
                await self.session.abort_transaction()
                await self.session.end_session()
                self.session = None
                self.in_transaction = False
                return DatabaseResult.success_result()
            elif self.connection_string.startswith("firestore"):
                await self.transaction.rollback()
                self.transaction = None
                self.in_transaction = False
                return DatabaseResult.success_result()
            else:
                return DatabaseResult.error_result(f"Unsupported document store type: {self.connection_string}")
        except Exception as e:
            logger.error(f"Failed to rollback transaction on document store: {self.name}, error: {e}")
            return DatabaseResult.error_result(TransactionError(f"Failed to rollback transaction: {e}"))
