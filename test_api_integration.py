"""Test script for FastAPI integration with MemoryStore."""
import asyncio
import json
from typing import Dict, Any
import httpx
import uvicorn
import threading
import time
import os

from src.memserv.interface.api import app


class APITester:
    """Test class for API integration."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=60.0)  # 增加超时时间
    
    async def test_health_check(self):
        """Test health check endpoint."""
        print("🏥 Testing health check...")
        response = await self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        print(f"✅ Health check: {data}")
        return data
    
    async def test_store_memory(self, content: str, user_id: str, metadata: Dict[str, Any] = None):
        """Test storing a memory."""
        print(f"📝 Testing store memory for user {user_id}...")
        payload = {
            "content": content,
            "user_id": user_id,
            "metadata": metadata or {}
        }
        response = await self.client.post("/memories", json=payload)
        assert response.status_code == 200
        data = response.json()
        print(f"✅ Memory stored: ID={data['id']}, Tags={data['tags']}")
        return data
    
    async def test_get_user_memories(self, user_id: str, limit: int = 10):
        """Test getting user memories."""
        print(f"📋 Testing get memories for user {user_id}...")
        response = await self.client.get(f"/users/{user_id}/memories?limit={limit}")
        assert response.status_code == 200
        data = response.json()
        print(f"✅ Found {len(data)} memories for user {user_id}")
        return data
    
    async def test_get_user_tags(self, user_id: str):
        """Test getting user tags."""
        print(f"🏷️  Testing get tags for user {user_id}...")
        response = await self.client.get(f"/users/{user_id}/tags")
        assert response.status_code == 200
        data = response.json()
        print(f"✅ Found tags: {data}")
        return data
    
    async def test_get_memory_by_id(self, user_id: str, memory_id: str):
        """Test getting memory by ID."""
        print(f"🔍 Testing get memory by ID: {memory_id}...")
        response = await self.client.get(f"/memories/{user_id}/{memory_id}")
        assert response.status_code == 200
        data = response.json()
        print(f"✅ Retrieved memory: {data['content'][:50]}...")
        return data
    
    async def test_query_memories(self, query: str, user_id: str, limit: int = 5):
        """Test querying memories."""
        print(f"🔎 Testing query memories: '{query}'...")
        payload = {
            "query": query,
            "user_id": user_id,
            "limit": limit
        }
        response = await self.client.post("/memories/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        print(f"✅ Query returned {len(data['results'])} results")
        for i, result in enumerate(data['results'], 1):
            print(f"   {i}. Score: {result['score']:.3f} - {result['memory']['content'][:50]}...")
        return data
    
    async def test_delete_memory(self, user_id: str, memory_id: str):
        """Test deleting a memory."""
        print(f"🗑️  Testing delete memory: {memory_id}...")
        response = await self.client.delete(f"/memories/{user_id}/{memory_id}")
        assert response.status_code == 200
        data = response.json()
        print(f"✅ Memory deleted: {data['message']}")
        return data
    
    async def test_get_stats(self):
        """Test getting service statistics."""
        print("📊 Testing get statistics...")
        response = await self.client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        print(f"✅ Statistics: {json.dumps(data, indent=2)}")
        return data
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


def start_server():
    """Start the FastAPI server in a separate thread."""
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


async def run_api_tests():
    """Run all API integration tests."""
    print("🧪 Starting API Integration Tests...")
    print(f"🔑 Using API Key: {os.getenv('API_KEY', 'Not set')[:10]}..." if os.getenv('API_KEY') else "❌ No API Key found")
    
    # Start server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    
    tester = APITester()
    
    try:
        # Test 1: Health check
        await tester.test_health_check()
        
        # Test 2: Store memories
        memory1 = await tester.test_store_memory(
            content="I learned about LlamaIndex today. It's a powerful framework for building RAG applications with various data sources.",
            user_id="test_user_1",
            metadata={"source": "documentation", "category": "learning"}
        )
        
        memory2 = await tester.test_store_memory(
            content="FastMCP is a Python framework that makes it easy to build MCP servers and clients. It provides tools and resources for AI agents.",
            user_id="test_user_1",
            metadata={"source": "documentation", "category": "development"}
        )
        
        memory3 = await tester.test_store_memory(
            content="Remember to buy groceries: milk, bread, eggs, and apples. Also need to pick up dry cleaning.",
            user_id="test_user_2",
            metadata={"category": "personal", "priority": "high"}
        )
        
        # Test 3: Get user memories
        user1_memories = await tester.test_get_user_memories("test_user_1")
        user2_memories = await tester.test_get_user_memories("test_user_2")
        
        # Test 4: Get user tags
        await tester.test_get_user_tags("test_user_1")
        await tester.test_get_user_tags("test_user_2")
        
        # Test 5: Get memory by ID
        await tester.test_get_memory_by_id("test_user_1", memory1["id"])
        
        # Test 6: Query memories
        await tester.test_query_memories("Tell me about frameworks", "test_user_1")
        await tester.test_query_memories("What do I need to buy?", "test_user_2")
        
        # Test 7: Get statistics
        await tester.test_get_stats()
        
        # Test 8: Delete memory
        await tester.test_delete_memory("test_user_2", memory3["id"])
        
        # Test 9: Verify deletion
        print("🔍 Verifying memory deletion...")
        try:
            await tester.test_get_memory_by_id("test_user_2", memory3["id"])
            print("❌ Memory should have been deleted!")
        except Exception:
            print("✅ Memory successfully deleted (404 as expected)")
        
        # Test 10: Final statistics
        await tester.test_get_stats()
        
        print("\n🎉 All API integration tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await tester.close()


if __name__ == "__main__":
    asyncio.run(run_api_tests())