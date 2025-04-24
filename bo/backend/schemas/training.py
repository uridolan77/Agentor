"""
Schemas for training-related data structures.
"""

from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator


class TrainingMetrics(BaseModel):
    """
    Training metrics for a single epoch.
    """
    epoch: int
    loss: float
    accuracy: float
    validation_loss: float = Field(..., alias="validationLoss")
    validation_accuracy: float = Field(..., alias="validationAccuracy")
    learning_rate: float = Field(..., alias="learningRate")
    timestamp: str

    class Config:
        allow_population_by_field_name = True


class TrainingConfig(BaseModel):
    """
    Configuration for a training session.
    """
    model_type: str = Field(..., alias="modelType")
    batch_size: int = Field(..., alias="batchSize")
    epochs: int
    learning_rate: float = Field(..., alias="learningRate")
    optimizer: str
    dataset_id: str = Field(..., alias="datasetId")
    validation_split: float = Field(..., alias="validationSplit")
    early_stopping_patience: int = Field(..., alias="earlyStoppingPatience")
    use_checkpointing: bool = Field(..., alias="useCheckpointing")
    checkpoint_frequency: int = Field(..., alias="checkpointFrequency")

    class Config:
        allow_population_by_field_name = True


class TrainingSessionBase(BaseModel):
    """
    Base model for training sessions.
    """
    name: Optional[str] = None
    agent_id: Optional[str] = Field(None, alias="agentId")

    class Config:
        allow_population_by_field_name = True


class TrainingSessionCreate(TrainingSessionBase):
    """
    Model for creating a training session.
    """
    config: TrainingConfig


class TrainingSessionUpdate(BaseModel):
    """
    Model for updating a training session.
    """
    name: Optional[str] = None
    status: Optional[str] = None
    progress: Optional[float] = None
    end_time: Optional[str] = Field(None, alias="endTime")

    class Config:
        allow_population_by_field_name = True


class TrainingSessionResponse(TrainingSessionBase):
    """
    Model for training session responses.
    """
    id: str
    status: str
    progress: float
    start_time: str = Field(..., alias="startTime")
    end_time: Optional[str] = Field(None, alias="endTime")
    config: TrainingConfig
    metrics: List[Dict[str, Any]] = []
    model_id: Optional[str] = Field(None, alias="modelId")

    class Config:
        allow_population_by_field_name = True


class DatasetBase(BaseModel):
    """
    Base model for datasets.
    """
    name: str
    description: Optional[str] = None
    size: int
    format: str
    metadata: Optional[Dict[str, Any]] = None


class DatasetCreate(DatasetBase):
    """
    Model for creating a dataset.
    """
    file_path: str = Field(..., alias="filePath")

    class Config:
        allow_population_by_field_name = True


class DatasetResponse(DatasetBase):
    """
    Model for dataset responses.
    """
    id: str
    created_at: str = Field(..., alias="createdAt")
    updated_at: Optional[str] = Field(None, alias="updatedAt")

    class Config:
        allow_population_by_field_name = True


class ModelBase(BaseModel):
    """
    Base model for ML models.
    """
    name: str
    description: Optional[str] = None
    agent_id: Optional[str] = Field(None, alias="agentId")
    training_session_id: Optional[str] = Field(None, alias="trainingSessionId")


class ModelCreate(ModelBase):
    """
    Model for creating an ML model.
    """
    config: Dict[str, Any]
    metrics: Optional[List[Dict[str, Any]]] = None

    class Config:
        allow_population_by_field_name = True


class ModelResponse(ModelBase):
    """
    Model for ML model responses.
    """
    id: str
    created_at: str = Field(..., alias="createdAt")
    updated_at: Optional[str] = Field(None, alias="updatedAt")
    config: Dict[str, Any]
    metrics: Optional[List[Dict[str, Any]]] = None
    performance: Optional[Dict[str, Any]] = None

    class Config:
        allow_population_by_field_name = True


class TrainingWebSocketMessage(BaseModel):
    """
    Model for WebSocket messages during training.
    """
    type: str  # 'update', 'error', 'complete'
    session_id: str = Field(..., alias="sessionId")
    data: Dict[str, Any]

    class Config:
        allow_population_by_field_name = True
