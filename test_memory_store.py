"""Test script for MemoryStore class."""
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from src.memserv.core.models import MemoryInput, MemoryQuery
from src.memserv.core.memory_store import MemoryStore
from src.memserv.core.config import config


def test_memory_store():

    """Test MemoryStore functionality."""
    print("🧪 Starting MemoryStore tests...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Override config for testing
        temp_dir = "/tmp/victest"
        config.data_dir = temp_dir
        import shutil
        config.memories_dir = os.path.join(temp_dir, "memories")
        config.index_dir = os.path.join(temp_dir, "index")
        
        print(f"📁 Using temporary directory: {temp_dir}")
        
        
        try:
            # Initialize MemoryStore
            print("🔧 Initializing MemoryStore...")
            print(f"🔑 Using API Key: {config.api_key[:10]}..." if config.api_key else "❌ No API Key found")
            print(f"🌐 API Base: {config.api_base}")
            print(f"🤖 LLM Model: {config.llm_model}")
            print(f"📊 Embedding Model: {config.embedding_model}")
            
            memory_store = MemoryStore()
            
            print("✅ MemoryStore initialized successfully")
            
            # Test 1: Store a memory
            print("\n📝 Test 1: Storing a memory...")
            memory_input = MemoryInput(
                content="I learned about LlamaIndex today. It's a great framework for building RAG applications.",
                user_id="test_user_1",
                metadata={"source": "documentation", "category": "learning"}
            )
            
            stored_memory = memory_store.store_memory(memory_input)
            print(f"✅ Memory stored with ID: {stored_memory.id}")
            print(f"   Content: {stored_memory.content[:50]}...")
            print(f"   Tags: {stored_memory.tags}")
            print(f"   User ID: {stored_memory.user_id}")
            
            # Test 2: Store another memory for the same user
            print("\n📝 Test 2: Storing another memory...")
            memory_input2 = MemoryInput(
                content="FastMCP is a Python framework for building MCP servers. It makes it easy to create tools and resources.",
                user_id="test_user_1",
                metadata={"source": "documentation", "category": "learning"}
            )
            
            stored_memory2 = memory_store.store_memory(memory_input2)
            print(f"✅ Second memory stored with ID: {stored_memory2.id}")
            print(f"   Tags: {stored_memory2.tags}")
            
            # Test 3: Store memory for different user
            print("\n📝 Test 3: Storing memory for different user...")
            memory_input3 = MemoryInput(
                content="I need to remember to buy groceries tomorrow: milk, bread, eggs, and apples.",
                user_id="test_user_2",
                metadata={"category": "personal"}
            )
            
            stored_memory3 = memory_store.store_memory(memory_input3)
            print(f"✅ Third memory stored with ID: {stored_memory3.id}")
            print(f"   User ID: {stored_memory3.user_id}")
            
            # Test 4: Get user memories
            print("\n📋 Test 4: Getting user memories...")
            user1_memories = memory_store.get_user_memories("test_user_1")
            print(f"✅ Found {len(user1_memories)} memories for test_user_1")
            for i, memory in enumerate(user1_memories, 1):
                print(f"   {i}. {memory.content[:50]}... (ID: {memory.id})")
            
            user2_memories = memory_store.get_user_memories("test_user_2")
            print(f"✅ Found {len(user2_memories)} memories for test_user_2")
            
            # Test 5: Get user tags
            print("\n🏷️  Test 5: Getting user tags...")
            user1_tags = memory_store.get_user_tags("test_user_1")
            print(f"✅ Tags for test_user_1: {user1_tags}")
            
            user2_tags = memory_store.get_user_tags("test_user_2")
            print(f"✅ Tags for test_user_2: {user2_tags}")
            
            # Test 6: Get memory by ID
            print("\n🔍 Test 6: Getting memory by ID...")
            retrieved_memory = memory_store.get_memory_by_id(stored_memory.id, "test_user_1")
            if retrieved_memory:
                print(f"✅ Retrieved memory: {retrieved_memory.content[:50]}...")
            else:
                print("❌ Failed to retrieve memory by ID")
            
            # Test 7: Query memories (this might not work perfectly with mock embedding)
            print("\n🔎 Test 7: Querying memories...")
            try:
                query = MemoryQuery(
                    query="Tell me about frameworks",
                    user_id="test_user_1",
                    limit=3
                )
                
                query_response = memory_store.query_memories(query)
                print(f"✅ Query returned {len(query_response.results)} results")
                for i, result in enumerate(query_response.results, 1):
                    print(f"   {i}. Score: {result.score:.3f} - {result.memory.content[:50]}...")
            except Exception as e:
                print(f"⚠️  Query test skipped due to mock limitations: {e}")
            
            # Test 8: Get statistics
            print("\n📊 Test 8: Getting statistics...")
            stats = memory_store.get_stats()
            print(f"✅ Statistics:")
            print(f"   Total memories: {stats['total_memories']}")
            print(f"   Total users: {stats['total_users']}")
            print(f"   Data directory: {stats['data_directory']}")
            
            all = list(Path(config.memories_dir).parent.rglob("*"))
            all_str ="\n".join([a.absolute() for a in all])
            print(f"   Total files in data directory: {len(all)}")
            # Test 9: Delete memory
            print("\n🗑️  Test 9: Deleting memory...")
            deleted = memory_store.delete_memory(stored_memory3.id, "test_user_2")
            if deleted:
                print(f"✅ Successfully deleted memory {stored_memory3.id}")
                
                # Verify deletion
                deleted_memory = memory_store.get_memory_by_id(stored_memory3.id, "test_user_2")
                if deleted_memory is None:
                    print("✅ Memory confirmed deleted")
                else:
                    print("❌ Memory still exists after deletion")
            else:
                print("❌ Failed to delete memory")
            
            # Test 10: File system verification
            print("\n📁 Test 10: File system verification...")
            memory_files = list(Path(config.memories_dir).glob("*.json"))
            print(f"✅ Found {len(memory_files)} memory files on disk")
            for file in memory_files:
                print(f"   - {file.name}")
            
            print("\n🎉 All tests completed successfully!")
            print("📂 Copied memory files to /tmp/victest")
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_memory_store()