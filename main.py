"""Main entry point for memory service."""
import argparse
import asyncio
import uvicorn
from src.memserv.interface.api import app
from src.memserv.interface.mcp_interface import mcp
from src.memserv.core.config import config
from src.memserv.core.memory_store import MemoryStore


def initialize_service():
    """预初始化服务，避免第一次请求时的延迟。"""
    assert config.api_key, (f"   - API Key: {'✅ Set' if config.api_key else '❌ Not Set'}")
    print("🔧 Pre-initializing Memory Service...")
    print(f"🔧 Configuration:")
    print(f"   - API Base: {config.api_base}")
    print(f"   - LLM Model: {config.llm_model}")
    print(f"   - Embedding Model: {config.embedding_model}")
    print(f"   - Data Directory: {config.data_dir}")
    
    # 预初始化 MemoryStore（单例模式）
    memory_store = MemoryStore.get_instance()
    
    print("✅ Memory Service pre-initialization completed!")
    return memory_store


def run_fastapi_server(host: str, port: int, reload: bool, log_level: str):
    """Run the FastAPI server."""
    print(f"🚀 Starting Memory Service FastAPI on {host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🌐 FastAPI Server starting at http://{host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )


def run_mcp_server(host: str, port: int, path: str = "/mcp"):
    """Run the FastMCP server with streamhttp transport."""
    print(f"🔧 Starting Memory Service MCP Server on {host}:{port}{path}")
    print(f"🔗 MCP Endpoint: http://{host}:{port}{path}")
    
    # Run the FastMCP server with streamhttp transport
    mcp.run(
        transport="http",
        host=host,
        port=port,
        path=path,
        log_level="info"
    )


async def run_mcp_server_async(host: str, port: int, path: str = "/mcp"):
    """Run the FastMCP server asynchronously with streamhttp transport."""
    print(f"🔧 Starting Memory Service MCP Server (async) on {host}:{port}{path}")
    print(f"🔗 MCP Endpoint: http://{host}:{port}{path}")
    
    # Run the FastMCP server asynchronously
    await mcp.run_async(
        transport="http",
        host=host,
        port=port,
        path=path,
        log_level="info"
    )


def main():
    """Main entry point for the memory service."""
    parser = argparse.ArgumentParser(description="Memory Service")
    parser.add_argument("--host", default=config.host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=config.port, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", help="Log level")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip model warmup for faster startup")
    
    # MCP server options
    parser.add_argument("--mode", choices=["fastapi", "mcp", "both"], default="both", 
                       help="Server mode: fastapi (REST API), mcp (MCP server), or both")
    parser.add_argument("--mcp-port", type=int, default=8001, help="Port for MCP server (when running both)")
    parser.add_argument("--mcp-path", default="/mcp", help="Path for MCP endpoint")
    
    args = parser.parse_args()
    
    # 预初始化服务（除非在开发模式下使用 reload）
    if not args.reload and not args.skip_warmup:
        initialize_service()
    else:
        print("⚠️  Skipping pre-initialization in reload/skip-warmup mode")
    
    if args.mode == "fastapi":
        # Run only FastAPI server
        run_fastapi_server(args.host, args.port, args.reload, args.log_level)
        
    elif args.mode == "mcp":
        # Run only MCP server
        run_mcp_server(args.host, args.port, args.mcp_path)
        
    elif args.mode == "both":
        # Run both servers (FastAPI and MCP)
        print("🚀 Starting both FastAPI and MCP servers...")
        print(f"📚 FastAPI Documentation: http://{args.host}:{args.port}/docs")
        print(f"🔗 MCP Endpoint: http://{args.host}:{args.mcp_port}{args.mcp_path}")
        
        # This would require running both servers concurrently
        # For now, we'll run MCP server on a different port
        import threading
        
        # Start FastAPI server in a separate thread
        fastapi_thread = threading.Thread(
            target=run_fastapi_server,
            args=(args.host, args.port, args.reload, args.log_level),
            daemon=True
        )
        fastapi_thread.start()
        
        # Run MCP server in main thread
        try:
            run_mcp_server(args.host, args.mcp_port, args.mcp_path)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down servers...")


def run_mcp_only():
    """Convenience function to run only the MCP server."""
    initialize_service()
    run_mcp_server("127.0.0.1", 8001, "/mcp")


def run_mcp_stdio():
    """Run MCP server with STDIO transport (for MCP clients like Claude Desktop)."""
    initialize_service()
    print("🔧 Starting Memory Service MCP Server with STDIO transport")
    print("📡 Ready for MCP client connections via STDIO")
    
    # Run with STDIO transport for MCP clients
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()