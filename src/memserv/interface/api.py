"""FastAPI interface for memory service."""
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from ..core.models import MemoryInput, Memory, MemoryQuery, MemoryResponse
from ..core.memory_store import MemoryStore

app = FastAPI(
    title="Memory Service API",
    description="A service for storing and retrieving user memories with AI-powered tagging",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize memory store (singleton)
memory_store = MemoryStore.get_instance()


@app.post("/memories", response_model=Memory)
async def store_memory(memory_input: MemoryInput):
    """Store a new memory with auto-generated tags."""
    try:
        memory = memory_store.store_memory(memory_input)
        return memory
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memories/query", response_model=MemoryResponse)
async def query_memories(query: MemoryQuery):
    """Query memories based on content similarity and optional tag filters."""
    try:
        response = memory_store.query_memories(query)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}/memories", response_model=List[Memory])
async def get_user_memories(user_id: str, limit: int = Query(default=10, ge=1, le=100)):
    """Get all memories for a specific user."""
    try:
        memories = memory_store.get_user_memories(user_id, limit)
        return memories
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}/tags", response_model=List[str])
async def get_user_tags(user_id: str):
    """Get all unique tags for a specific user."""
    try:
        tags = memory_store.get_user_tags(user_id)
        return tags
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memories/{user_id}/{memory_id}", response_model=Memory)
async def get_memory_by_id(user_id: str, memory_id: str):
    """Get a specific memory by ID and user ID."""
    try:
        memory = memory_store.get_memory_by_id(memory_id, user_id)
        if memory is None:
            raise HTTPException(status_code=404, detail="Memory not found")
        return memory
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memories/{user_id}/{memory_id}")
async def delete_memory(user_id: str, memory_id: str):
    """Delete a specific memory by ID and user ID."""
    try:
        success = memory_store.delete_memory(memory_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"message": "Memory deleted successfully", "memory_id": memory_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """Get service statistics."""
    try:
        stats = memory_store.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "memory-service",
        "version": "0.1.0"
    }


if __name__ == "__main__":
    import uvicorn
    from ..core.config import config
    
    uvicorn.run(app, host=config.host, port=config.port)