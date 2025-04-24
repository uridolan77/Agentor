"""
Procedural Memory implementation for the Agentor framework.

Procedural memory stores learned behaviors, skills, and procedures that the agent can execute.
It allows for storing and retrieving action sequences and learned policies.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import time
import json
import numpy as np
import asyncio
from dataclasses import dataclass, field
import logging
import pickle
import base64

from agentor.components.memory import Memory
from agentor.llm_gateway.utils.metrics import track_memory_operation

logger = logging.getLogger(__name__)


@dataclass
class Procedure:
    """A procedure or skill stored in procedural memory."""
    
    id: str
    """Unique identifier for the procedure."""
    
    name: str
    """Human-readable name of the procedure."""
    
    description: str
    """Description of what the procedure does."""
    
    created_at: float = field(default_factory=time.time)
    """When the procedure was created."""
    
    updated_at: float = field(default_factory=time.time)
    """When the procedure was last updated."""
    
    execution_count: int = 0
    """Number of times this procedure has been executed."""
    
    success_count: int = 0
    """Number of successful executions."""
    
    average_duration: float = 0.0
    """Average execution duration in seconds."""
    
    parameters: Dict[str, Any] = field(default_factory=dict)
    """Parameters for the procedure."""
    
    tags: List[str] = field(default_factory=list)
    """Tags for categorizing the procedure."""
    
    embedding: Optional[List[float]] = None
    """Vector embedding of the procedure for semantic search."""
    
    # The actual procedure implementation can be stored in different ways
    code: Optional[str] = None
    """Python code implementing the procedure."""
    
    function: Optional[bytes] = None
    """Serialized function implementing the procedure."""
    
    steps: Optional[List[Dict[str, Any]]] = None
    """Step-by-step instructions for the procedure."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the procedure to a dictionary.
        
        Returns:
            Dictionary representation of the procedure
        """
        result = {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'average_duration': self.average_duration,
            'parameters': self.parameters,
            'tags': self.tags
        }
        
        if self.code:
            result['code'] = self.code
        
        if self.function:
            # Encode the serialized function as base64
            result['function'] = base64.b64encode(self.function).decode('utf-8')
        
        if self.steps:
            result['steps'] = self.steps
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Procedure':
        """Create a procedure from a dictionary.
        
        Args:
            data: Dictionary representation of a procedure
            
        Returns:
            A Procedure object
        """
        # Create a copy of the data to avoid modifying the original
        data_copy = data.copy()
        
        # Handle function serialization
        function_data = data_copy.pop('function', None)
        if function_data:
            # Decode the base64-encoded function
            function_bytes = base64.b64decode(function_data)
            data_copy['function'] = function_bytes
        
        return cls(**data_copy)
    
    def record_execution(self, success: bool, duration: float):
        """Record an execution of this procedure.
        
        Args:
            success: Whether the execution was successful
            duration: The duration of the execution in seconds
        """
        self.execution_count += 1
        if success:
            self.success_count += 1
        
        # Update the average duration using a weighted average
        if self.execution_count == 1:
            self.average_duration = duration
        else:
            self.average_duration = (
                (self.average_duration * (self.execution_count - 1) + duration) / 
                self.execution_count
            )
        
        self.updated_at = time.time()
    
    def get_success_rate(self) -> float:
        """Get the success rate of this procedure.
        
        Returns:
            The success rate as a float between 0.0 and 1.0
        """
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count
    
    async def execute(self, *args, **kwargs) -> Any:
        """Execute this procedure.
        
        Args:
            *args: Positional arguments for the procedure
            **kwargs: Keyword arguments for the procedure
            
        Returns:
            The result of the procedure
            
        Raises:
            ValueError: If the procedure cannot be executed
        """
        start_time = time.time()
        success = False
        result = None
        
        try:
            if self.function:
                # Deserialize and execute the function
                func = pickle.loads(self.function)
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                success = True
            
            elif self.code:
                # Execute the code
                # This is potentially unsafe and should be used with caution
                local_vars = {'args': args, 'kwargs': kwargs, 'result': None}
                exec(self.code, {}, local_vars)
                result = local_vars.get('result')
                success = True
            
            elif self.steps:
                # For step-by-step procedures, just return the steps
                result = self.steps
                success = True
            
            else:
                raise ValueError(f"Procedure {self.id} has no executable content")
        
        except Exception as e:
            logger.error(f"Error executing procedure {self.id}: {str(e)}")
            raise
        
        finally:
            # Record the execution
            duration = time.time() - start_time
            self.record_execution(success, duration)
        
        return result


class ProceduralMemory(Memory):
    """Procedural memory implementation that stores learned behaviors and skills."""
    
    def __init__(
        self,
        embedding_provider=None,
        max_procedures: int = 1000,
        retention_threshold: float = 0.2,
    ):
        """Initialize the procedural memory.
        
        Args:
            embedding_provider: Provider for generating embeddings
            max_procedures: Maximum number of procedures to store
            retention_threshold: Success rate threshold below which procedures may be forgotten
        """
        self.procedures: Dict[str, Procedure] = {}
        self.embedding_provider = embedding_provider
        self.max_procedures = max_procedures
        self.retention_threshold = retention_threshold
        self.lock = asyncio.Lock()
    
    @track_memory_operation("add", "procedural")
    async def add(self, item: Dict[str, Any]):
        """Add a procedure to procedural memory.
        
        Args:
            item: The procedure to add
        """
        async with self.lock:
            # Generate a procedure ID if not provided
            proc_id = item.get('id')
            if proc_id is None:
                proc_id = f"proc-{int(time.time())}-{hash(str(time.time()))}"[:16]
                item['id'] = proc_id
            
            # Check if this is an update to an existing procedure
            if proc_id in self.procedures:
                # Update the existing procedure
                existing_proc = self.procedures[proc_id]
                
                # Preserve execution statistics
                execution_count = existing_proc.execution_count
                success_count = existing_proc.success_count
                average_duration = existing_proc.average_duration
                
                # Create a new procedure from the item
                procedure = Procedure.from_dict(item)
                
                # Restore execution statistics
                procedure.execution_count = execution_count
                procedure.success_count = success_count
                procedure.average_duration = average_duration
                
                self.procedures[proc_id] = procedure
                logger.info(f"Updated procedure {proc_id}")
            else:
                # Create a new procedure
                procedure = Procedure.from_dict(item)
                self.procedures[proc_id] = procedure
                logger.info(f"Added new procedure {proc_id}")
            
            # Generate embedding if we have a provider
            if self.embedding_provider and not procedure.embedding:
                # Create a text representation of the procedure
                proc_text = self._procedure_to_text(procedure)
                
                # Generate the embedding
                procedure.embedding = await self.embedding_provider.get_embedding(proc_text)
            
            # Check if we need to prune procedures
            await self._prune_if_needed()
    
    @track_memory_operation("get", "procedural")
    async def get(self, query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Get procedures that match a query.
        
        Args:
            query: The query to match
            limit: Maximum number of procedures to return
            
        Returns:
            A list of matching procedures as dictionaries
        """
        results = []
        
        # If querying by procedure ID
        if 'id' in query:
            procedure = self.procedures.get(query['id'])
            if procedure:
                results.append(procedure.to_dict())
            return results
        
        # If querying by name
        elif 'name' in query:
            name = query['name'].lower()
            for procedure in self.procedures.values():
                if name in procedure.name.lower():
                    results.append(procedure.to_dict())
                    if len(results) >= limit:
                        break
        
        # If querying by tags
        elif 'tags' in query:
            tags = query['tags']
            if not isinstance(tags, list):
                tags = [tags]
            
            for procedure in self.procedures.values():
                if any(tag in procedure.tags for tag in tags):
                    results.append(procedure.to_dict())
                    if len(results) >= limit:
                        break
        
        # If querying by semantic similarity
        elif 'text' in query and self.embedding_provider:
            query_embedding = await self.embedding_provider.get_embedding(query['text'])
            
            # Get procedures with embeddings
            procs_with_embeddings = [
                (proc, proc.embedding)
                for proc in self.procedures.values()
                if proc.embedding is not None
            ]
            
            if procs_with_embeddings:
                # Calculate similarities
                similarities = [
                    (self._cosine_similarity(query_embedding, embedding), proc)
                    for proc, embedding in procs_with_embeddings
                ]
                
                # Sort by similarity
                similarities.sort(reverse=True, key=lambda x: x[0])
                
                # Get top results
                for similarity, proc in similarities[:limit]:
                    if similarity >= query.get('threshold', 0.0):
                        result = proc.to_dict()
                        result['similarity'] = similarity
                        results.append(result)
        
        # Otherwise, return procedures sorted by success rate
        else:
            sorted_procs = sorted(
                self.procedures.values(),
                key=lambda p: p.get_success_rate(),
                reverse=True
            )
            
            for proc in sorted_procs[:limit]:
                results.append(proc.to_dict())
        
        return results
    
    @track_memory_operation("clear", "procedural")
    async def clear(self):
        """Clear all procedures from memory."""
        async with self.lock:
            self.procedures = {}
    
    async def get_procedure(self, proc_id: str) -> Optional[Procedure]:
        """Get a specific procedure by ID.
        
        Args:
            proc_id: The ID of the procedure to get
            
        Returns:
            The procedure, or None if not found
        """
        return self.procedures.get(proc_id)
    
    async def execute_procedure(self, proc_id: str, *args, **kwargs) -> Any:
        """Execute a procedure by ID.
        
        Args:
            proc_id: The ID of the procedure to execute
            *args: Positional arguments for the procedure
            **kwargs: Keyword arguments for the procedure
            
        Returns:
            The result of the procedure
            
        Raises:
            ValueError: If the procedure is not found
        """
        procedure = self.procedures.get(proc_id)
        if procedure is None:
            raise ValueError(f"Procedure {proc_id} not found")
        
        return await procedure.execute(*args, **kwargs)
    
    async def store_function(
        self, 
        name: str, 
        description: str, 
        func: Callable, 
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Store a Python function as a procedure.
        
        Args:
            name: Name of the procedure
            description: Description of the procedure
            func: The function to store
            parameters: Optional parameters for the function
            tags: Optional tags for categorizing the procedure
            
        Returns:
            The ID of the stored procedure
        """
        # Serialize the function
        serialized_func = pickle.dumps(func)
        
        # Create a procedure
        proc_id = f"proc-func-{int(time.time())}-{hash(name)}"[:16]
        
        procedure = Procedure(
            id=proc_id,
            name=name,
            description=description,
            function=serialized_func,
            parameters=parameters or {},
            tags=tags or []
        )
        
        # Store the procedure
        async with self.lock:
            self.procedures[proc_id] = procedure
            
            # Generate embedding if we have a provider
            if self.embedding_provider:
                proc_text = self._procedure_to_text(procedure)
                procedure.embedding = await self.embedding_provider.get_embedding(proc_text)
            
            # Check if we need to prune procedures
            await self._prune_if_needed()
        
        return proc_id
    
    async def store_code(
        self, 
        name: str, 
        description: str, 
        code: str, 
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Store Python code as a procedure.
        
        Args:
            name: Name of the procedure
            description: Description of the procedure
            code: The Python code to store
            parameters: Optional parameters for the code
            tags: Optional tags for categorizing the procedure
            
        Returns:
            The ID of the stored procedure
        """
        # Create a procedure
        proc_id = f"proc-code-{int(time.time())}-{hash(name)}"[:16]
        
        procedure = Procedure(
            id=proc_id,
            name=name,
            description=description,
            code=code,
            parameters=parameters or {},
            tags=tags or []
        )
        
        # Store the procedure
        async with self.lock:
            self.procedures[proc_id] = procedure
            
            # Generate embedding if we have a provider
            if self.embedding_provider:
                proc_text = self._procedure_to_text(procedure)
                procedure.embedding = await self.embedding_provider.get_embedding(proc_text)
            
            # Check if we need to prune procedures
            await self._prune_if_needed()
        
        return proc_id
    
    async def store_steps(
        self, 
        name: str, 
        description: str, 
        steps: List[Dict[str, Any]], 
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Store step-by-step instructions as a procedure.
        
        Args:
            name: Name of the procedure
            description: Description of the procedure
            steps: The step-by-step instructions
            parameters: Optional parameters for the steps
            tags: Optional tags for categorizing the procedure
            
        Returns:
            The ID of the stored procedure
        """
        # Create a procedure
        proc_id = f"proc-steps-{int(time.time())}-{hash(name)}"[:16]
        
        procedure = Procedure(
            id=proc_id,
            name=name,
            description=description,
            steps=steps,
            parameters=parameters or {},
            tags=tags or []
        )
        
        # Store the procedure
        async with self.lock:
            self.procedures[proc_id] = procedure
            
            # Generate embedding if we have a provider
            if self.embedding_provider:
                proc_text = self._procedure_to_text(procedure)
                procedure.embedding = await self.embedding_provider.get_embedding(proc_text)
            
            # Check if we need to prune procedures
            await self._prune_if_needed()
        
        return proc_id
    
    async def delete_procedure(self, proc_id: str) -> bool:
        """Delete a procedure.
        
        Args:
            proc_id: The ID of the procedure to delete
            
        Returns:
            True if the procedure was deleted, False otherwise
        """
        async with self.lock:
            if proc_id in self.procedures:
                del self.procedures[proc_id]
                return True
            return False
    
    async def _prune_if_needed(self):
        """Prune procedures if we have more than the maximum."""
        if len(self.procedures) <= self.max_procedures:
            return
        
        # Sort procedures by success rate
        sorted_procs = sorted(
            self.procedures.values(),
            key=lambda p: p.get_success_rate()
        )
        
        # Remove the least successful procedures
        procs_to_remove = len(self.procedures) - self.max_procedures
        for proc in sorted_procs[:procs_to_remove]:
            # Only remove procedures below the retention threshold
            if proc.get_success_rate() <= self.retention_threshold:
                del self.procedures[proc.id]
    
    def _procedure_to_text(self, procedure: Procedure) -> str:
        """Convert a procedure to a text representation for embedding.
        
        Args:
            procedure: The procedure to convert
            
        Returns:
            A text representation of the procedure
        """
        parts = [
            f"Name: {procedure.name}",
            f"Description: {procedure.description}"
        ]
        
        if procedure.tags:
            parts.append(f"Tags: {', '.join(procedure.tags)}")
        
        if procedure.parameters:
            param_parts = []
            for key, value in procedure.parameters.items():
                param_parts.append(f"{key}: {value}")
            parts.append(f"Parameters: {', '.join(param_parts)}")
        
        if procedure.code:
            # Include a snippet of the code
            code_snippet = procedure.code[:200] + "..." if len(procedure.code) > 200 else procedure.code
            parts.append(f"Code: {code_snippet}")
        
        if procedure.steps:
            # Include step descriptions
            step_parts = []
            for i, step in enumerate(procedure.steps[:5]):  # Limit to first 5 steps
                if 'description' in step:
                    step_parts.append(f"Step {i+1}: {step['description']}")
            
            if step_parts:
                parts.append("Steps: " + " ".join(step_parts))
        
        return "\n".join(parts)
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate the cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            The cosine similarity (between -1 and 1)
        """
        a_np = np.array(a)
        b_np = np.array(b)
        return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))
