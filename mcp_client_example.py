#!/usr/bin/env python3
"""Example client for connecting to FastMCP Memory Service via streamhttp."""
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastmcp import Client


async def main():
    """Example of connecting to FastMCP Memory Service."""
    print("🔗 FastMCP Memory Service Client Example")
    print("=" * 50)
    
    # Connect to the MCP server via HTTP
    server_url = "http://127.0.0.1:8001/mcp"
    print(f"Connecting to: {server_url}")
    
    try:
        async with Client(server_url) as client:
            print("✅ Connected to FastMCP Memory Service")
            
            # Test connection
            await client.ping()
            print("📡 Connection verified")
            
            # Example: Store a memory
            print("\n💾 Storing a memory...")
            store_result = await client.call_tool("store_memory", {
                "content": "I learned about FastMCP streamhttp interface today. It's a modern way to expose MCP servers over HTTP.",
                "user_id": "demo_user",
                "metadata": {
                    "category": "learning",
                    "importance": "high",
                    "source": "documentation"
                }
            })
            
            if store_result.data.get("success"):
                print(f"✅ Memory stored: {store_result.data['message']}")
                memory_id = store_result.data['memory']['id']
                print(f"   Memory ID: {memory_id}")
            else:
                print(f"❌ Failed to store memory: {store_result.data.get('error')}")
            
            # Example: Query memories
            print("\n🔍 Querying memories...")
            query_result = await client.call_tool("query_memories", {
                "query": "FastMCP streamhttp",
                "user_id": "demo_user",
                "limit": 3
            })
            
            if query_result.data.get("success"):
                results = query_result.data['results']
                print(f"✅ Found {len(results)} relevant memories:")
                for i, result in enumerate(results, 1):
                    print(f"   {i}. Score: {result['score']:.3f}")
                    print(f"      Content: {result['memory']['content'][:100]}...")
                    print(f"      Tags: {', '.join(result['memory']['tags'])}")
            else:
                print(f"❌ Query failed: {query_result.data.get('error')}")
            
            # Example: Get user statistics
            print("\n📊 Getting user statistics...")
            stats_result = await client.read_resource("memory://stats/demo_user")
            
            # Handle different response formats
            if hasattr(stats_result, 'data'):
                stats_data = stats_result.data
            else:
                # Extract from list format
                stats_data = stats_result[0].text if isinstance(stats_result, list) else stats_result
            
            # Parse JSON if it's a string
            if isinstance(stats_data, str):
                import json
                stats = json.loads(stats_data)
            else:
                stats = stats_data
            
            if "error" not in stats:
                print(f"✅ User Statistics:")
                print(f"   Total memories: {stats['total_memories']}")
                print(f"   Most common tags: {dict(stats['most_common_tags'][:5])}")
            else:
                print(f"❌ Failed to get stats: {stats.get('error')}")
            
            # Example: Check service health
            print("\n🏥 Checking service health...")
            health_result = await client.read_resource("memory://health")
            
            # Handle different response formats
            if hasattr(health_result, 'data'):
                health_data = health_result.data
            else:
                # Extract from list format
                health_data = health_result[0].text if isinstance(health_result, list) else health_result
            
            # Parse JSON if it's a string
            if isinstance(health_data, str):
                import json
                health = json.loads(health_data)
            else:
                health = health_data
            
            print(f"✅ Service Status: {health['status']}")
            print(f"   Version: {health.get('version', 'unknown')}")
            print(f"   Memory Store: {health.get('memory_store', 'unknown')}")
            
            print("\n🎉 Client example completed successfully!")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nMake sure the MCP server is running:")
        print("  python main.py --mode mcp --port 8001")


if __name__ == "__main__":
    asyncio.run(main())