"""
Training service for agent models.
Provides functionality for managing training sessions, datasets, and models.
"""

import os
import json
import uuid
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

import aiofiles
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from db.database import get_db_session
from db.models import Agent, TrainingModel, TrainingDataset, TrainingSession
from schemas.training import (
    TrainingSessionCreate,
    TrainingSessionUpdate,
    TrainingSessionResponse,
    TrainingMetrics,
    TrainingConfig
)

# Configure logging
logger = logging.getLogger(__name__)

# Training session manager
class TrainingSessionManager:
    """
    Manages active training sessions and WebSocket connections.
    """
    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.connections: Dict[str, List[WebSocket]] = {}
        self.tasks: Dict[str, asyncio.Task] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        """
        Connect a client to a training session.
        """
        await websocket.accept()

        if session_id not in self.connections:
            self.connections[session_id] = []

        self.connections[session_id].append(websocket)

        # Send current session state if available
        if session_id in self.active_sessions:
            await websocket.send_json(self.active_sessions[session_id])

    def disconnect(self, session_id: str, websocket: WebSocket):
        """
        Disconnect a client from a training session.
        """
        if session_id in self.connections:
            if websocket in self.connections[session_id]:
                self.connections[session_id].remove(websocket)

            if not self.connections[session_id]:
                del self.connections[session_id]

    async def broadcast(self, session_id: str, data: Dict[str, Any]):
        """
        Broadcast data to all clients connected to a training session.
        """
        if session_id in self.connections:
            # Update active session state
            if session_id in self.active_sessions:
                self.active_sessions[session_id].update(data)
            else:
                self.active_sessions[session_id] = data

            # Broadcast to all connected clients
            for connection in self.connections[session_id]:
                try:
                    await connection.send_json(data)
                except WebSocketDisconnect:
                    await self.disconnect(session_id, connection)
                except Exception as e:
                    logger.error(f"Error broadcasting to client: {e}")

    def start_training_task(self, session_id: str, config: Dict[str, Any]):
        """
        Start a training task for a session.
        """
        if session_id in self.tasks and not self.tasks[session_id].done():
            logger.warning(f"Training task for session {session_id} already running")
            return

        task = asyncio.create_task(self._run_training(session_id, config))
        self.tasks[session_id] = task

    def stop_training_task(self, session_id: str):
        """
        Stop a training task for a session.
        """
        if session_id in self.tasks and not self.tasks[session_id].done():
            self.tasks[session_id].cancel()
            return True
        return False

    async def _run_training(self, session_id: str, config: Dict[str, Any]):
        """
        Run a training task for a session.
        This is a simulated training process for demonstration purposes.
        In a real implementation, this would interface with a machine learning framework.
        """
        try:
            # Initialize metrics
            metrics = []

            # Simulate training epochs
            for epoch in range(1, config['epochs'] + 1):
                # Simulate training step
                await asyncio.sleep(2)  # Simulate training time

                # Calculate simulated metrics
                progress = (epoch / config['epochs']) * 100
                loss = 1.0 / (epoch + 1) + 0.1 * (1 - epoch / config['epochs'])
                accuracy = 0.5 + 0.4 * (epoch / config['epochs'])
                val_loss = loss * 1.2
                val_accuracy = accuracy * 0.9

                # Create metrics object
                epoch_metrics = {
                    'epoch': epoch,
                    'loss': loss,
                    'accuracy': accuracy,
                    'validationLoss': val_loss,
                    'validationAccuracy': val_accuracy,
                    'learningRate': config['learningRate'] * (0.9 ** (epoch // 5)),
                    'timestamp': datetime.now().isoformat()
                }

                metrics.append(epoch_metrics)

                # Broadcast update
                await self.broadcast(session_id, {
                    'status': 'running',
                    'progress': progress,
                    'metrics': metrics
                })

                # Check for early stopping
                if epoch > config['earlyStoppingPatience']:
                    recent_losses = [m['validationLoss'] for m in metrics[-config['earlyStoppingPatience']:]]
                    if all(recent_losses[i] >= recent_losses[i-1] for i in range(1, len(recent_losses))):
                        logger.info(f"Early stopping triggered for session {session_id}")
                        break

            # Training completed
            await self.broadcast(session_id, {
                'status': 'completed',
                'progress': 100,
                'endTime': datetime.now().isoformat(),
                'metrics': metrics
            })

            # Update database
            await self._update_session_in_db(session_id, 'completed', metrics)

        except asyncio.CancelledError:
            # Training was cancelled
            logger.info(f"Training cancelled for session {session_id}")

            # Update status
            await self.broadcast(session_id, {
                'status': 'stopped',
                'endTime': datetime.now().isoformat()
            })

            # Update database
            await self._update_session_in_db(session_id, 'stopped', metrics if 'metrics' in locals() else [])

        except Exception as e:
            # Training failed
            logger.error(f"Training failed for session {session_id}: {e}")

            # Update status
            await self.broadcast(session_id, {
                'status': 'failed',
                'error': str(e),
                'endTime': datetime.now().isoformat()
            })

            # Update database
            await self._update_session_in_db(session_id, 'failed', metrics if 'metrics' in locals() else [])

    async def _update_session_in_db(self, session_id: str, status: str, metrics: List[Dict[str, Any]]):
        """
        Update a training session in the database.
        """
        try:
            async with get_db_session() as db:
                session = await db.get(TrainingSession, session_id)
                if session:
                    session.status = status
                    session.metrics = json.dumps(metrics)
                    session.end_time = datetime.now()
                    await db.commit()
        except Exception as e:
            logger.error(f"Error updating session in database: {e}")


# Create a global session manager
session_manager = TrainingSessionManager()


# Training service
class TrainingService:
    """
    Service for managing training sessions, datasets, and models.
    """
    @staticmethod
    async def create_session(session_data: TrainingSessionCreate) -> TrainingSessionResponse:
        """
        Create a new training session.
        """
        session_id = str(uuid.uuid4())

        # Create session in database
        async with get_db_session() as db:
            # Check if agent exists
            if session_data.agent_id:
                agent = await db.get(Agent, session_data.agent_id)
                if not agent:
                    raise ValueError(f"Agent with ID {session_data.agent_id} not found")

            # Check if dataset exists
            dataset = await db.get(TrainingDataset, session_data.config.dataset_id)
            if not dataset:
                raise ValueError(f"Dataset with ID {session_data.config.dataset_id} not found")

            # Create session
            session = TrainingSession(
                id=session_id,
                name=session_data.name or f"Training Session {session_id[:8]}",
                agent_id=session_data.agent_id,
                dataset_id=session_data.config.dataset_id,
                config=json.dumps(session_data.config.dict()),
                status="idle",
                start_time=datetime.now(),
                metrics="[]"
            )

            db.add(session)
            await db.commit()
            await db.refresh(session)

        # Prepare response
        response = TrainingSessionResponse(
            id=session.id,
            name=session.name,
            agent_id=session.agent_id,
            status="idle",
            progress=0,
            start_time=session.start_time.isoformat(),
            config=session_data.config,
            metrics=[]
        )

        # Start training task
        session_manager.start_training_task(session_id, session_data.config.dict())

        return response

    @staticmethod
    async def get_session(session_id: str) -> TrainingSessionResponse:
        """
        Get a training session by ID.
        """
        async with get_db_session() as db:
            session = await db.get(TrainingSession, session_id)
            if not session:
                raise ValueError(f"Training session with ID {session_id} not found")

            # Parse config and metrics
            config = TrainingConfig(**json.loads(session.config))
            metrics = json.loads(session.metrics)

            # Calculate progress
            progress = 0
            if metrics and config.epochs > 0:
                latest_epoch = metrics[-1]['epoch'] if metrics else 0
                progress = (latest_epoch / config.epochs) * 100

            return TrainingSessionResponse(
                id=session.id,
                name=session.name,
                agent_id=session.agent_id,
                status=session.status,
                progress=progress,
                start_time=session.start_time.isoformat(),
                end_time=session.end_time.isoformat() if session.end_time else None,
                config=config,
                metrics=metrics
            )

    @staticmethod
    async def list_sessions(agent_id: Optional[str] = None) -> List[TrainingSessionResponse]:
        """
        List training sessions, optionally filtered by agent ID.
        """
        async with get_db_session() as db:
            query = db.query(TrainingSession)

            if agent_id:
                query = query.filter(TrainingSession.agent_id == agent_id)

            sessions = await query.order_by(TrainingSession.start_time.desc()).all()

            return [
                TrainingSessionResponse(
                    id=session.id,
                    name=session.name,
                    agent_id=session.agent_id,
                    status=session.status,
                    progress=100 if session.status in ['completed', 'failed', 'stopped'] else
                             (len(json.loads(session.metrics)) / json.loads(session.config)['epochs'] * 100
                              if json.loads(session.metrics) and json.loads(session.config)['epochs'] > 0
                              else 0),
                    start_time=session.start_time.isoformat(),
                    end_time=session.end_time.isoformat() if session.end_time else None,
                    config=TrainingConfig(**json.loads(session.config)),
                    metrics=json.loads(session.metrics)
                )
                for session in sessions
            ]

    @staticmethod
    async def stop_session(session_id: str) -> TrainingSessionResponse:
        """
        Stop a training session.
        """
        # Stop training task
        stopped = session_manager.stop_training_task(session_id)

        if not stopped:
            # If task wasn't running, update database directly
            async with get_db_session() as db:
                session = await db.get(TrainingSession, session_id)
                if not session:
                    raise ValueError(f"Training session with ID {session_id} not found")

                if session.status == 'running':
                    session.status = 'stopped'
                    session.end_time = datetime.now()
                    await db.commit()
                    await db.refresh(session)

        # Return updated session
        return await TrainingService.get_session(session_id)

    @staticmethod
    async def save_model(session_id: str) -> str:
        """
        Save a model from a completed training session.
        """
        async with get_db_session() as db:
            session = await db.get(TrainingSession, session_id)
            if not session:
                raise ValueError(f"Training session with ID {session_id} not found")

            if session.status != 'completed':
                raise ValueError(f"Cannot save model from session with status {session.status}")

            # Create model
            model_id = str(uuid.uuid4())
            model = TrainingModel(
                id=model_id,
                name=f"Model from {session.name}",
                agent_id=session.agent_id,
                training_session_id=session.id,
                created_at=datetime.now(),
                config=session.config,
                metrics=session.metrics
            )

            db.add(model)
            await db.commit()

            return model_id

    @staticmethod
    async def list_datasets() -> List[Dict[str, Any]]:
        """
        List available datasets.
        """
        async with get_db_session() as db:
            datasets = await db.query(TrainingDataset).all()

            return [
                {
                    'id': dataset.id,
                    'name': dataset.name,
                    'description': dataset.description,
                    'size': dataset.size,
                    'created_at': dataset.created_at.isoformat()
                }
                for dataset in datasets
            ]

    @staticmethod
    async def list_models(agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available models, optionally filtered by agent ID.
        """
        async with get_db_session() as db:
            query = db.query(TrainingModel)

            if agent_id:
                query = query.filter(TrainingModel.agent_id == agent_id)

            models = await query.order_by(TrainingModel.created_at.desc()).all()

            return [
                {
                    'id': model.id,
                    'name': model.name,
                    'agent_id': model.agent_id,
                    'training_session_id': model.training_session_id,
                    'created_at': model.created_at.isoformat(),
                    'config': json.loads(model.config),
                    'metrics': json.loads(model.metrics)
                }
                for model in models
            ]
