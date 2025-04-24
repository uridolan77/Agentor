from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

from agentor.llm_gateway.api.auth import validate_api_key, get_settings, Settings, RBACMiddleware
from agentor.llm_gateway.api.middleware import InputValidationMiddleware
from agentor.llm_gateway.api.auth_routes import router as auth_router
from agentor.llm_gateway.api.model_routes import router as model_router
from agentor.llm_gateway.llm.base import LLMRequest, LLMResponse
from agentor.llm_gateway.utils.tracing import TracingMiddleware, setup_tracing
from agentor.llm_gateway.security import setup_security, cleanup_security


# Configure the application
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure the application on startup and shutdown."""
    # Startup
    settings = get_settings()

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add input validation middleware
    app.add_middleware(InputValidationMiddleware)

    # Add RBAC middleware
    app.add_middleware(RBACMiddleware)

    # Set up tracing
    tracer = setup_tracing("llm_gateway")

    # Add tracing middleware
    app.add_middleware(TracingMiddleware, tracer=tracer)

    # Set up security features
    security_components = setup_security(app)
    app.state.security_components = security_components

    yield

    # Shutdown
    # Clean up security components
    await cleanup_security(app.state.security_components)

# Create the app with lifespan
app = FastAPI(title="LLM Gateway API", lifespan=lifespan)

# Include routers
app.include_router(auth_router)
app.include_router(model_router)


class GenerateRequest(BaseModel):
    """Request to generate text from an LLM."""
    prompt: str
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stop_sequences: Optional[list[str]] = None


class GenerateResponse(BaseModel):
    """Response from generating text from an LLM."""
    text: str
    model: str
    usage: Dict[str, int]


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    api_key: str = Depends(validate_api_key),
    settings: Settings = Depends(get_settings)
):
    """Generate text from an LLM.

    Args:
        request: The generation request
        api_key: The validated API key
        settings: The application settings

    Returns:
        The generated text
    """
    # Get security components
    security_components = app.state.security_components
    session_pool = security_components["session_pool"]

    # Create or validate session
    session_token = None

    # In a real implementation, we would use the LLM gateway to generate text
    # For this example, we'll just return a dummy response
    response = GenerateResponse(
        text="This is a dummy response.",
        model=request.model,
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    )

    # Create a new session for this request
    try:
        session_token = await session_pool.create_session(user_id=api_key, metadata={
            "model": request.model
        })

        # In a real implementation, we would add the session token to the response headers
        # response.headers["X-Session-Token"] = session_token
    except Exception as e:
        # Log the error but don't fail the request
        print(f"Error creating session: {e}")

    return response


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
