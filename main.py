"""Main entry point for memory service."""
import argparse
import asyncio
import logging
import os
from logging.handlers import TimedRotatingFileHandler
import uvicorn
from src.memserv.interface.api import app
from src.memserv.interface.mcp_interface import mcp
from src.memserv.core.config import config
from src.memserv.core.memory_store import MemoryStore
from pathlib import Path

def setup_logging(log_level: str = "info"):
    """Setup logging with rotation, keeping logs for 7 days."""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Setup rotating file handler (daily rotation, keep 7 days)
    file_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, "memory-service.log"),
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(numeric_level)
    root_logger.addHandler(file_handler)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)
    
    # Log the setup completion
    logging.info(f"Logging setup completed - Level: {log_level.upper()}, Retention: 7 days")
    logging.info(f"Log files location: {os.path.abspath(log_dir)}")


def initialize_service():
    """预初始化服务，避免第一次请求时的延迟。"""
    assert config.api_key, (f"   - API Key: {'✅ Set' if config.api_key else '❌ Not Set'}")
    logging.info("🔧 Pre-initializing Memory Service...")
    logging.info(f"🔧 Configuration:")
    logging.info(f"   - API Base: {config.api_base}")
    logging.info(f"   - LLM Model: {config.llm_model}")
    logging.info(f"   - Embedding Model: {config.embedding_model}")
    logging.info(f"   - Data Directory: {Path(config.data_dir).absolute()}")
    
    # 预初始化 MemoryStore（单例模式）
    memory_store = MemoryStore.get_instance()
    
    logging.info("✅ Memory Service pre-initialization completed!")
    return memory_store


def run_fastapi_server(host: str, port: int, reload: bool, log_level: str):
    """Run the FastAPI server."""
    logging.info(f"🚀 Starting Memory Service FastAPI on {host}:{port}")
    logging.info(f"📚 API Documentation: http://{host}:{port}/docs")
    logging.info(f"🌐 FastAPI Server starting at http://{host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )


def run_mcp_server(host: str, port: int, path: str = "/mcp"):
    """Run the FastMCP server with streamhttp transport."""
    logging.info(f"🔧 Starting Memory Service MCP Server on {host}:{port}{path}")
    logging.info(f"🔗 MCP Endpoint: http://{host}:{port}{path}")
    
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
    logging.info(f"🔧 Starting Memory Service MCP Server (async) on {host}:{port}{path}")
    logging.info(f"🔗 MCP Endpoint: http://{host}:{port}{path}")
    
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
    parser.add_argument("--mcp-port", type=int, default=None, help="Port for MCP server")
    parser.add_argument("--mcp-path", default="/mcp", help="Path for MCP endpoint")
    
    args = parser.parse_args()
    
    # Setup logging first
    setup_logging(args.log_level)
    logging.info("Memory Service starting up...")
    
    # 预初始化服务（除非在开发模式下使用 reload）
    if not args.reload and not args.skip_warmup:
        initialize_service()
    else:
        logging.info("⚠️  Skipping pre-initialization in reload/skip-warmup mode")
    
    if args.mode == "fastapi":
        # Run only FastAPI server
        run_fastapi_server(args.host, args.port, args.reload, args.log_level)
        
    elif args.mode == "mcp":
        # Run only MCP server
        run_mcp_server(args.host, args.mcp_port or args.port, args.mcp_path)
        
    elif args.mode == "both":
        # Run both servers (FastAPI and MCP)
        logging.info("🚀 Starting both FastAPI and MCP servers...")
        logging.info(f"📚 FastAPI Documentation: http://{args.host}:{args.port}/docs")
        logging.info(f"🔗 MCP Endpoint: http://{args.host}:{args.mcp_port}{args.mcp_path}")
        
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
            logging.info("\n🛑 Shutting down servers...")


def run_mcp_only():
    """Convenience function to run only the MCP server."""
    initialize_service()
    run_mcp_server("127.0.0.1", 8001, "/mcp")


def run_mcp_stdio():
    """Run MCP server with STDIO transport (for MCP clients like Claude Desktop)."""
    initialize_service()
    logging.info("🔧 Starting Memory Service MCP Server with STDIO transport")
    logging.info("📡 Ready for MCP client connections via STDIO")
    
    # Run with STDIO transport for MCP clients
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()