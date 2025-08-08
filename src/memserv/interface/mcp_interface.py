"""MCP (Model Context Protocol) interface for memory service using FastMCP."""
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastmcp import FastMCP

from ..core.models import MemoryInput, MemoryQuery
from ..core.memory_store import MemoryStore
from ..core.service import MemoryService


# Create FastMCP server instance
mcp = FastMCP(
    name="MemoryService",
    instructions="""
    This is a personal memory management service that helps store, retrieve, and query memories.
    
    Available tools:
    - store_memory: Store new memories with automatic tag generation
    - query_memories: Search memories using semantic similarity
    - get_user_memories: Retrieve all memories for a specific user
    
    All operations require a user_id to maintain user-specific memory isolation.
    """
)


class MCPMemoryInterface:
    """MCP interface for memory service operations."""
    
    def __init__(self):
        # Use the singleton MemoryStore instance
        self.memory_store = MemoryStore.get_instance()


# Initialize the interface
interface = MCPMemoryInterface()


@mcp.tool
def store_memory(content: str, user_id: str, metadata: Optional[Dict[str, Any]] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """Store a new memory with auto-generated tags and optional custom tags.
    
    Args:
        content: The content to remember
        user_id: User identifier
        metadata: Additional metadata (optional)
        tags: Optional custom tags to add to auto-generated tags
    
    Returns:
        Dict containing success status, stored memory data with combined tags, and message
    """
    try:
        memory_input = MemoryInput(
            content=content,
            user_id=user_id,
            metadata=metadata or {}
        )
        memory = interface.memory_store.store_memory(memory_input, custom_tags=tags)
        return {
            "success": True,
            "memory": memory.model_dump(mode='json'),
            "message": f"Memory stored with ID: {memory.id}",
            "tags_info": {
                "total_tags": len(memory.tags),
                "custom_tags_added": len(tags) if tags else 0
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to store memory"
        }


@mcp.tool
def query_memories(query: str, user_id: str, limit: int = 5, tags: Optional[List[str]] = None, 
                  include_archived: bool = False) -> Dict[str, Any]:
    """Query memories based on content similarity.
    
    Args:
        query: Search query
        user_id: User identifier
        limit: Maximum number of results (default: 5)
        tags: Filter by tags (optional)
        include_archived: Whether to include archived memories (default: False)
    
    Returns:
        Dict containing success status, search results, total count, and message
    """
    try:
        memory_query = MemoryQuery(
            query=query,
            user_id=user_id,
            limit=limit,
            tags=tags or []
        )
        response = interface.memory_store.query_memories(memory_query, include_archived=include_archived)
        archived_info = " (including archived)" if include_archived else ""
        return {
            "success": True,
            "results": [result.model_dump(mode='json') for result in response.results],
            "total": response.total,
            "message": f"Found {response.total} relevant memories{archived_info}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to query memories"
        }


@mcp.tool
def update_memory(memory_id: str, user_id: str, content: str, 
                 metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Update an existing memory with new content.
    
    Args:
        memory_id: ID of the memory to update
        user_id: User identifier
        content: New content for the memory
        metadata: Optional metadata (default: None)
    
    Returns:
        Dict containing success status, new memory data, update info, and message
    """
    try:
        result = interface.memory_store.update_memory(memory_id, user_id, content, metadata)
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to update memory"
        }


@mcp.tool
def get_user_memories(user_id: str, limit: int = 10, tags: Optional[List[str]] = None, 
                     include_archived: bool = False) -> Dict[str, Any]:
    """Get all memories for a specific user with optional tag filtering.
    
    Args:
        user_id: User identifier
        limit: Maximum number of results (default: 10)
        tags: Optional list of tags to filter memories by
        include_archived: Whether to include archived memories (default: False)
    
    Returns:
        Dict containing success status, filtered memories list, count, and message
    """
    try:
        memories = interface.memory_store.get_user_memories(user_id, limit, tags, include_archived)
        filter_info = f" with tags {tags}" if tags else ""
        archived_info = " (including archived)" if include_archived else ""
        return {
            "success": True,
            "memories": [memory.model_dump(mode='json') for memory in memories],
            "count": len(memories),
            "message": f"Retrieved {len(memories)} memories for user {user_id}{filter_info}{archived_info}",
            "filter_applied": {
                "tags": tags or [],
                "filtered": bool(tags),
                "include_archived": include_archived
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to get user memories"
        }


@mcp.resource("memory://stats/{user_id}")
def get_user_stats(user_id: str) -> Dict[str, Any]:
    """Get memory statistics for a specific user.
    
    Args:
        user_id: User identifier
    
    Returns:
        Dict containing user memory statistics
    """
    try:
        memories = interface.memory_store.get_user_memories(user_id, limit=1000)  # Get all for stats
        total_memories = len(memories)
        
        # Calculate tag distribution
        tag_counts = {}
        for memory in memories:
            for tag in memory.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return {
            "user_id": user_id,
            "total_memories": total_memories,
            "tag_distribution": tag_counts,
            "most_common_tags": sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    except Exception as e:
        return {
            "error": str(e),
            "message": "Failed to get user statistics"
        }


@mcp.resource("memory://health")
def get_service_health() -> Dict[str, Any]:
    """Get service health status.
    
    Returns:
        Dict containing service health information
    """
    try:
        # Test memory store connection
        test_store = interface.memory_store
        
        return {
            "status": "healthy",
            "service": "Memory Service",
            "version": "0.1.0",
            "memory_store": "connected",
            "timestamp": str(datetime.now())
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "message": "Service health check failed"
        }


# Legacy compatibility functions for existing code
class MCPMemoryInterface:
    """Legacy MCP interface for backward compatibility."""
    
    def __init__(self):
        self.memory_store = MemoryStore.get_instance()
    
    def store_memory_tool(self, content: str, user_id: str, metadata: Dict[str, Any] = None, tags: List[str] = None) -> Dict[str, Any]:
        """Legacy wrapper for store_memory tool."""
        return store_memory(content, user_id, metadata, tags)
    
    def query_memories_tool(self, query: str, user_id: str, limit: int = 5, tags: List[str] = None) -> Dict[str, Any]:
        """Legacy wrapper for query_memories tool."""
        return query_memories(query, user_id, limit, tags)
    
    def get_user_memories_tool(self, user_id: str, limit: int = 10, tags: List[str] = None) -> Dict[str, Any]:
        """Legacy wrapper for get_user_memories tool."""
        return get_user_memories(user_id, limit, tags)


def get_mcp_tools():
    """Legacy function for backward compatibility."""
    interface = MCPMemoryInterface()
    
    return {
        "store_memory": {
            "description": "Store a new memory with auto-generated tags and optional custom tags",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to remember"
                    },
                    "user_id": {
                        "type": "string", 
                        "description": "User identifier"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional metadata (optional)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional custom tags to add to auto-generated tags"
                    }
                },
                "required": ["content", "user_id"]
            },
            "handler": interface.store_memory_tool
        },
        "query_memories": {
            "description": "Query memories based on content similarity",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User identifier"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by tags (optional)"
                    }
                },
                "required": ["query", "user_id"]
            },
            "handler": interface.query_memories_tool
        },
        "get_user_memories": {
            "description": "Get all memories for a specific user with optional tag filtering",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "User identifier"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of tags to filter memories by"
                    }
                },
                "required": ["user_id"]
            },
            "handler": interface.get_user_memories_tool
        }
    }