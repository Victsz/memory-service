"""Service layer interface for memory operations."""
from abc import ABC, abstractmethod
from typing import List, Protocol

from .models import MemoryInput, Memory, MemoryQuery, MemoryResponse


class MemoryServiceProtocol(Protocol):
    """Protocol defining the memory service interface."""
    
    def store_memory(self, memory_input: MemoryInput) -> Memory:
        """Store a new memory."""
        ...
    
    def query_memories(self, query: MemoryQuery) -> MemoryResponse:
        """Query memories based on similarity."""
        ...
    
    def get_user_memories(self, user_id: str, limit: int = 10) -> List[Memory]:
        """Get all memories for a user."""
        ...


class MemoryService:
    """Service layer for memory operations."""
    
    def __init__(self, store: MemoryServiceProtocol):
        self.store = store
    
    def store_memory(self, memory_input: MemoryInput) -> Memory:
        """Store a new memory with validation."""
        if not memory_input.content.strip():
            raise ValueError("Memory content cannot be empty")
        if not memory_input.user_id.strip():
            raise ValueError("User ID cannot be empty")
        
        return self.store.store_memory(memory_input)
    
    def query_memories(self, query: MemoryQuery) -> MemoryResponse:
        """Query memories with validation."""
        if not query.query.strip():
            raise ValueError("Query cannot be empty")
        if not query.user_id.strip():
            raise ValueError("User ID cannot be empty")
        if query.limit <= 0:
            raise ValueError("Limit must be positive")
        
        return self.store.query_memories(query)
    
    def get_user_memories(self, user_id: str, limit: int = 10) -> List[Memory]:
        """Get user memories with validation."""
        if not user_id.strip():
            raise ValueError("User ID cannot be empty")
        if limit <= 0:
            raise ValueError("Limit must be positive")
        
        return self.store.get_user_memories(user_id, limit)