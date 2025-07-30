import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from contextlib import asynccontextmanager

# Correctly import your pipeline and configuration manager based on the path
# This assumes you run the app from the project root directory
from backend.src.pipeline.rag_pipeline import LegalRAGPipeline
from backend.src.config_settings.config_manager import ConfigurationManager
from backend.src.utils.logger import logger


# --- API Data Models (Schemas) ---
# Defines the expected structure for requests and responses.
# This provides validation and generates OpenAPI documentation automatically.

class QueryRequest(BaseModel):
    """The request model for a user's query."""
    query: str = Field(..., description="The legal question from the user.",
                       example="What is the statement of guarantee?")
    namespace: str = Field(..., description="The knowledge base or document set to search in.",
                           example="companies-act-2015-v1")
    # Optional: You could add model_id here if you want the frontend to choose the LLM
    # model_id: str | None = Field(None, example="gpt-4-turbo")


class Citation(BaseModel):
    """The response model for a single citation."""
    section: str | None = None
    title: str | None = None
    source_url: str | None = None
    # Add any other fields your pipeline's citation objects have
    # e.g., document_id: str | None = None


class QueryResponse(BaseModel):
    """The response model containing the answer and its sources."""
    answer: str
    citations: List[Citation]


# --- Lifespan Management ---
# This is the modern FastAPI way to handle startup and shutdown events.
# We use this to initialize the expensive RAG pipeline object only once.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan. The RAG pipeline is initialized
    on startup and will be available application-wide.
    """
    logger.info("Application startup...")
    # Initialize and store the RAG pipeline in the application state
    # This "singleton" pattern ensures we don't reload models on every request.
    app.state.rag_pipeline = LegalRAGPipeline(ConfigurationManager())
    logger.info("LegalRAGPipeline initialized successfully.")

    yield

    # --- Cleanup tasks can go here (on shutdown) ---
    logger.info("Application shutdown.")


# --- FastAPI Application Setup ---

app = FastAPI(
    title="Legal RAG API",
    description="API to interact with a legal RAG pipeline.",
    version="1.0.0",
    lifespan=lifespan  # Register the lifespan context manager
)

# --- CORS Middleware ---
# This is essential for allowing your frontend application to communicate
# with this backend API.
# WARNING: For production, be more restrictive with the origins.
origins = [
    "http://localhost",
    "http://localhost:3000",  # Example for a React frontend
    "http://localhost:5173",  # Example for a Vite/Svelte/Vue frontend
    # Add your deployed frontend's URL here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# --- API Endpoints ---

@app.get("/", tags=["Health"])
async def read_root():
    """A simple health check endpoint to confirm the server is running."""
    return {"status": "Legal RAG API is running"}


@app.post("/query", response_model=QueryResponse, tags=["RAG Pipeline"])
async def handle_query(request: Request, query_request: QueryRequest):
    """
    Receives a user query, processes it through the RAG pipeline,
    and returns the generated answer with citations.
    """
    try:
        logger.info(f"Received query for namespace '{query_request.namespace}': '{query_request.query[:100]}...'")

        # Access the pipeline instance from the application state
        pipeline: LegalRAGPipeline = request.app.state.rag_pipeline

        # Process the query using the pipeline
        response_data = await pipeline.process_query(
            query=query_request.query,
            namespace=query_request.namespace,
            # model_id=query_request.model_id # Uncomment if you add model_id to request
        )

        # FastAPI will automatically validate the response against the QueryResponse model
        return response_data

    except Exception as e:
        logger.error(f"An error occurred while processing the query: {e}", exc_info=True)
        # For a client-facing error, it's better to be generic
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred. Please check the logs."
        )


# --- Main Entry Point ---

if __name__ == "__main__":

    # `reload=True` automatically restarts the server when you change code
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)