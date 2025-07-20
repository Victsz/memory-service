#!/usr/bin/env python3
"""Test script for FastMCP streamhttp interface."""
import asyncio
import sys
from pathlib import Path
TEST_PORT = 7999
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastmcp import Client
from src.memserv.interface.mcp_interface import mcp


async def test_mcp_server_in_memory():
    """Test MCP server functionality in-memory (no network)."""
    print("🧪 Testing FastMCP Memory Service (In-Memory)")
    print("=" * 50)
    
    try:
        # Test with in-memory server (no network required)
        async with Client(mcp) as client:
            print("✅ Connected to MCP server")
            
            # Test ping
            print("\n📡 Testing connection...")
            ping_result = await client.ping()
            print(f"Ping result: {ping_result}")
            
            # List available tools
            print("\n🔧 Available tools:")
            tools_response = await client.list_tools()
            tools = tools_response.tools if hasattr(tools_response, 'tools') else tools_response
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")
            
            # List available resources
            print("\n📚 Available resources:")
            resources_response = await client.list_resources()
            resources = resources_response.resources if hasattr(resources_response, 'resources') else resources_response
            for resource in resources:
                print(f"  - {resource.uri}: {resource.name or 'No name'}")
            
            # Test store_memory tool
            print("\n💾 Testing store_memory tool...")
            store_result = await client.call_tool("store_memory", {
                "content": "This is a test memory for FastMCP integration",
                "user_id": "test_user_mcp",
                "metadata": {"source": "mcp_test", "priority": "high"}
            })
            print(f"Store result: {store_result.data}")
            
            # Test query_memories tool
            print("\n🔍 Testing query_memories tool...")
            query_result = await client.call_tool("query_memories", {
                "query": "FastMCP integration test",
                "user_id": "test_user_mcp",
                "limit": 3
            })
            print(f"Query result: {query_result.data}")
            
            # Test get_user_memories tool
            print("\n👤 Testing get_user_memories tool...")
            user_memories_result = await client.call_tool("get_user_memories", {
                "user_id": "test_user_mcp",
                "limit": 5
            })
            print(f"User memories result: {user_memories_result.data}")
            
            # Test health resource
            print("\n🏥 Testing health resource...")
            health_result = await client.read_resource("memory://health")
            health_data = health_result.data if hasattr(health_result, 'data') else health_result
            print(f"Health status: {health_data}")
            
            # Test user stats resource
            print("\n📊 Testing user stats resource...")
            stats_result = await client.read_resource("memory://stats/test_user_mcp")
            stats_data = stats_result.data if hasattr(stats_result, 'data') else stats_result
            print(f"User stats: {stats_data}")
            
            print("\n✅ All tests completed successfully!")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_mcp_server_http():
    """Test MCP server via HTTP (requires running server)."""
    
    print("\n🌐 Testing FastMCP Memory Service (HTTP)")
    print("=" * 50)
    print(f"Note: This requires the MCP server to be running on http://127.0.0.1:{TEST_PORT}/mcp")
    
    try:
        # Test with HTTP transport
        async with Client(f"http://127.0.0.1:{TEST_PORT}/mcp") as client:
            print("✅ Connected to HTTP MCP server")
            
            # Test ping
            ping_result = await client.ping()
            print(f"HTTP Ping result: {ping_result}")
            
            # Test a simple tool call
            store_result = await client.call_tool("store_memory", {
                "content": "HTTP test memory via FastMCP streamhttp",
                "user_id": "test_user_http",
                "metadata": {"transport": "http", "test": True}
            })
            print(f"HTTP Store result: {store_result.data}")
            
            print("✅ HTTP test completed successfully!")
            
    except Exception as e:
        print(f"⚠️  HTTP test failed (server may not be running): {e}")


def main():
    """Run all tests."""
    print("🚀 FastMCP Memory Service Test Suite")
    print("=" * 60)
    
    # Run in-memory tests
    asyncio.run(test_mcp_server_in_memory())
    
    # Run HTTP tests
    asyncio.run(test_mcp_server_http())
    
    print("\n🎉 Test suite completed!")
    print("\nTo test HTTP functionality:")
    print(f"1. Run: python main.py --mode mcp --port {TEST_PORT}")
    print("2. Then run this test script again")


if __name__ == "__main__":
    main()