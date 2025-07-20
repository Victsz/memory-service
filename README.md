# Memory Service

AI-powered memory storage and retrieval system built with LlamaIndex and FastMCP.

## Features

- **Memory Storage**: Store user content with automatic AI-powered tagging
- **Semantic Search**: Query memories using natural language with vector similarity
- **Multi-Interface**: Both FastMCP (for AI agents) and REST API (for applications)
- **OpenAI-like API**: Compatible with OpenAI, SiliconFlow, and other providers
- **User Isolation**: Memories are isolated by user ID
- **Tag-based Filtering**: Filter memories by automatically generated tags

## Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   FastMCP       │    │   FastAPI       │
│   Interface     │    │   Interface     │
│   (AI Agents)   │    │   (Apps)        │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └──────────┬───────────┘
                     │
          ┌─────────────────┐
          │  Memory Service │
          │  (Core Logic)   │
          └─────────┬───────┘
                    │
          ┌─────────────────┐
          │   LlamaIndex    │
          │   + Storage     │
          └─────────────────┘
```

## Installation

1. Clone and navigate to the project:
```bash
cd memory-service
```

2. Install dependencies:
```bash
uv sync
```

3. Copy and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Configuration

### OpenAI Configuration
```env
API_KEY=your-openai-key
API_BASE=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-3.5-turbo
```

### SiliconFlow Configuration
```env
API_KEY=your-siliconflow-key
API_BASE=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-m3
LLM_MODEL=Pro/deepseek-ai/DeepSeek-V3
```

## Usage

### Start Memory Service
#### Both Modes (FastAPI + MCP)
```bash
# Run both FastAPI (port 8000) and MCP server (port 8001)
uv run python main.py --mode both --mcp-port 8001
```

#### STDIO Mode (for MCP clients like Claude Desktop)
```python
# For integration with MCP clients via STDIO
from main import run_mcp_stdio
run_mcp_stdio()
```

### API Documentation
Once the server is running, you can access:
- **Interactive API Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

## FastMCP Interface

### Tools (Actions)

#### store_memory
Store a new memory for a user.
```python
await client.call_tool("store_memory", {
    "content": "I learned about FastMCP today",
    "user_id": "user123",
    "metadata": {"source": "documentation"}
})
```

#### query_memories
Query memories using semantic search.
```python
await client.call_tool("query_memories", {
    "query": "What did I learn about FastMCP?",
    "user_id": "user123",
    "limit": 5
})
```

#### get_user_tags
Get all unique tags for a user.
```python
await client.call_tool("get_user_tags", {"user_id": "user123"})
```

### Resources (Read-only)

#### memory://users/{user_id}/memories
Get all memories for a user.

#### memory://users/{user_id}/tags
Get all tags for a user.

#### memory://stats
Get service statistics.

## REST API Interface

### Store Memory
```bash
curl -X POST http://localhost:8000/memories \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Begal is my son",
    "user_id": "victor",
    "metadata": {"source": "documentation"}
  }'
```

### Query Memories
```bash
curl -X POST http://localhost:8000/memories/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What did I learn about LlamaIndex?",
    "user_id": "victor",
    "limit": 5,
    "tags": []
  }'
```

### Get User Memories
```bash
curl http://localhost:8000/users/user123/memories?limit=10
```

### Get User Tags
```bash
curl http://localhost:8000/users/user123/tags
```

### Get Memory by ID
```bash
curl http://localhost:8000/memories/user123/memory-id-here
```

### Delete Memory
```bash
curl -X DELETE http://localhost:8000/memories/user123/memory-id-here
```

### Get Service Statistics
```bash
curl http://localhost:8000/stats
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Testing with FastMCP Client

### StreamHTTP Client Example
```python
import asyncio
from fastmcp import Client

async def test_memory_service():
    # Connect to MCP server via streamhttp
    async with Client("http://localhost:8001/mcp") as client:
        # Store a memory
        result = await client.call_tool("store_memory", {
            "content": "FastMCP streamhttp interface works great!",
            "user_id": "test_user",
            "metadata": {"category": "learning", "transport": "streamhttp"}
        })
        print("Stored memory:", result.data)
        
        # Query memories
        results = await client.call_tool("query_memories", {
            "query": "Tell me about FastMCP streamhttp",
            "user_id": "test_user",
            "limit": 3
        })
        print("Query results:", results.data)
        
        # Get user memories
        memories = await client.call_tool("get_user_memories", {
            "user_id": "test_user",
            "limit": 5
        })
        print("User memories:", memories.data)
        
        # Read service health resource
        health = await client.read_resource("memory://health")
        print("Service health:", health.data)
        
        # Read user stats resource
        stats = await client.read_resource("memory://stats/test_user")
        print("User stats:", stats.data)

if __name__ == "__main__":
    asyncio.run(test_memory_service())
```

### Running Tests
```bash
# Test the MCP server functionality
uv run python test_mcp_server.py

# Run the client example (requires MCP server running)
uv run python main.py --mode mcp --port 8001  # Terminal 1
uv run python mcp_client_example.py          # Terminal 2
```

## MCP Configuration

### StreamHTTP MCP Server Configuration
{
  "mcpServers":{
       "memory-service-http": {
     "type":"streamableHttp",
      "url": "http://localhost:8002/mcp"
   }
  }
}
```
## Data Storage

- **Memories**: Stored as JSON files in `./data/memories/`
- **Vector Index**: LlamaIndex vector store in `./data/index/`
- **File Pattern**: `{user_id}_{memory_id}.json`

## Development

### Project Structure
```
src/memserv/
├── core/                    # Core business logic
│   ├── __init__.py         # Core module exports
│   ├── config.py           # Configuration management
│   ├── models.py           # Pydantic data models
│   ├── memory_store.py     # LlamaIndex-based storage
│   └── service.py          # Service layer abstraction
├── interface/              # API interfaces
│   ├── __init__.py         # Interface module exports
│   ├── api.py              # FastAPI REST interface
│   └── mcp_interface.py    # FastMCP server interface
└── __init__.py             # Main package entry point
```

### Key Design Principles

1. **Separation of Concerns**: Core logic in `service.py`, interfaces are thin wrappers
2. **Multiple Interfaces**: Same service logic exposed via FastMCP and FastAPI
3. **OpenAI-like Compatibility**: Works with various LLM providers
4. **User Isolation**: All operations are scoped to user IDs
5. **Extensible**: Easy to add new interfaces or modify existing ones
6. **Robust Error Handling**: Structured output with retry logic and graceful fallbacks

### AI-Powered Features

#### Automatic Tag Generation
- Uses LlamaIndex structured output with Pydantic models
- Implements retry logic (up to 2 retries) for reliability
- Falls back to empty tags if all attempts fail
- Considers common tags for consistency

#### Content Enhancement
- Automatically expands short content (<100 words) for better context
- Uses structured output to ensure clean, formatted responses
- Gracefully falls back to original content if enhancement fails
- Preserves original content with clear separation from enhancements

#### Error Resilience
- **Tag Generation**: Retries on failure, logs attempts, returns empty list as fallback
- **Content Expansion**: Single attempt with graceful fallback to original content
- **Structured Output**: Uses `LLMTextCompletionProgram` with `PydanticOutputParser` for non-function-calling models
- **Data Integrity**: Never loses original content due to AI processing failures

## License

MIT License