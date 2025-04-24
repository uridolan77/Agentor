"""
API endpoints for training-related operations.
"""

from typing import List, Dict, Optional, Any
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query, Path, Body
from fastapi.responses import JSONResponse

from schemas.training import (
    TrainingSessionCreate,
    TrainingSessionUpdate,
    TrainingSessionResponse,
    DatasetResponse,
    ModelResponse
)
from services.training import TrainingService, session_manager
from auth.security import get_current_user

# Create router
router = APIRouter(prefix="/training", tags=["training"])


@router.post("/sessions", response_model=TrainingSessionResponse)
async def create_training_session(
    session_data: TrainingSessionCreate,
    current_user = Depends(get_current_user)
):
    """
    Create a new training session.
    """
    try:
        session = await TrainingService.create_session(session_data)
        return session
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create training session: {str(e)}")


@router.get("/sessions", response_model=List[TrainingSessionResponse])
async def list_training_sessions(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    current_user = Depends(get_current_user)
):
    """
    List training sessions, optionally filtered by agent ID.
    """
    try:
        sessions = await TrainingService.list_sessions(agent_id)
        return sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list training sessions: {str(e)}")


@router.get("/sessions/{session_id}", response_model=TrainingSessionResponse)
async def get_training_session(
    session_id: str = Path(..., description="Training session ID"),
    current_user = Depends(get_current_user)
):
    """
    Get a training session by ID.
    """
    try:
        session = await TrainingService.get_session(session_id)
        return session
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get training session: {str(e)}")


@router.post("/sessions/{session_id}/stop", response_model=TrainingSessionResponse)
async def stop_training_session(
    session_id: str = Path(..., description="Training session ID"),
    current_user = Depends(get_current_user)
):
    """
    Stop a training session.
    """
    try:
        session = await TrainingService.stop_session(session_id)
        return session
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop training session: {str(e)}")


@router.post("/sessions/{session_id}/save", response_model=Dict[str, str])
async def save_model_from_session(
    session_id: str = Path(..., description="Training session ID"),
    current_user = Depends(get_current_user)
):
    """
    Save a model from a completed training session.
    """
    try:
        model_id = await TrainingService.save_model(session_id)
        return {"model_id": model_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save model: {str(e)}")


@router.get("/datasets", response_model=List[Dict[str, Any]])
async def list_datasets(
    current_user = Depends(get_current_user)
):
    """
    List available datasets.
    """
    try:
        datasets = await TrainingService.list_datasets()
        return datasets
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}")


@router.get("/models", response_model=List[Dict[str, Any]])
async def list_models(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    current_user = Depends(get_current_user)
):
    """
    List available models, optionally filtered by agent ID.
    """
    try:
        models = await TrainingService.list_models(agent_id)
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.websocket("/sessions/{session_id}/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str = Path(..., description="Training session ID")
):
    """
    WebSocket endpoint for real-time training updates.
    """
    try:
        # Connect to session
        await session_manager.connect(session_id, websocket)

        # Keep connection open until client disconnects
        try:
            while True:
                # Wait for messages (not used in this implementation)
                await websocket.receive_text()
        except WebSocketDisconnect:
            session_manager.disconnect(session_id, websocket)
    except Exception as e:
        # Handle any other exceptions
        if websocket.client_state.CONNECTED:
            await websocket.close(code=1011, reason=f"Internal server error: {str(e)}")
