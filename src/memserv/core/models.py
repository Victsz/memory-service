"""Data models for memory service."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class MemoryInput(BaseModel):
    """Input model for storing memories."""
    content: str = Field(..., description="The content to remember")
    user_id: str = Field(..., description="User identifier")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class Memory(BaseModel):
    """Memory model with generated tags and embeddings."""
    id: str = Field(..., description="Unique memory identifier")
    content: str = Field(..., description="The memory content")
    user_id: str = Field(..., description="User identifier")
    tags: List[str] = Field(default_factory=list, description="Auto-generated tags")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class MemoryQuery(BaseModel):
    """Query model for retrieving memories."""
    query: str = Field(..., description="Search query")
    user_id: str = Field(..., description="User identifier")
    limit: int = Field(default=5, description="Maximum number of results")
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags")


class MemoryResult(BaseModel):
    """Result model for memory retrieval."""
    memory: Memory = Field(..., description="The retrieved memory")
    score: float = Field(..., description="Similarity score")


class MemoryResponse(BaseModel):
    """Response model for memory queries."""
    results: List[MemoryResult] = Field(..., description="Retrieved memories")
    total: int = Field(..., description="Total number of results")