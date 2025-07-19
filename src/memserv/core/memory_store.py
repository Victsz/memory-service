"""Memory storage and retrieval using LlamaIndex."""
import os
import json
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document, Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like import OpenAILike

from .models import Memory, MemoryInput, MemoryQuery, MemoryResult, MemoryResponse
from .config import config


class MemoryStore:
    """Memory storage and retrieval system using LlamaIndex (Singleton)."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MemoryStore, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # 避免重复初始化
        if self._initialized:
            return
            
        print("🔧 Initializing MemoryStore...")
        
        self.data_dir = Path(config.data_dir)
        self.memories_dir = Path(config.memories_dir)
        self.index_dir = Path(config.index_dir)
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.memories_dir.mkdir(exist_ok=True)
        self.index_dir.mkdir(exist_ok=True)
        
        print(f"📁 Data directories created: {self.data_dir}")
        
        # Initialize LLM and embedding model
        print("🤖 Initializing LLM and embedding models...")
        self.llm = OpenAILike(
            model=config.llm_model,
            api_base=config.api_base,
            api_key=config.api_key,
            context_window=config.context_window,
            temperature=config.temperature,
            is_chat_model=True,
            is_function_calling_model=False,
        )
        
        self.embed_model = OpenAILikeEmbedding(
            model_name=config.embedding_model,
            api_base=config.api_base,
            api_key=config.api_key,
            embed_batch_size=config.embed_batch_size,
        )
        
        # Set global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        print(f"✅ Models initialized - LLM: {config.llm_model}, Embedding: {config.embedding_model}")
        
        # Initialize or load index
        self._init_index()
        
        # 预热模型 - 进行一次小的测试调用
        self._warmup_models()
        
        self._initialized = True
        print("✅ MemoryStore initialization completed!")
    
    def _warmup_models(self):
        """预热模型，避免第一次调用时的延迟。"""
        try:
            print("🔥 Warming up models...")
            
            # 预热 LLM
            warmup_prompt = "Generate one tag for: test"
            self.llm.complete(warmup_prompt)
            print("✅ LLM warmed up")
            
            # 预热 Embedding 模型
            self.embed_model.get_text_embedding("test warmup text")
            print("✅ Embedding model warmed up")
            
        except Exception as e:
            print(f"⚠️  Model warmup failed (this is normal on first run): {e}")
    
    @classmethod
    def get_instance(cls):
        """获取 MemoryStore 单例实例。"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _init_index(self):
        """Initialize or load existing vector index."""
        try:
            # Try to load existing index
            storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore.from_persist_dir(persist_dir=str(self.index_dir)),
                vector_store=SimpleVectorStore.from_persist_dir(persist_dir=str(self.index_dir)),
                index_store=SimpleIndexStore.from_persist_dir(persist_dir=str(self.index_dir)),
            )
            self.index = VectorStoreIndex.from_documents(
                [], storage_context=storage_context, embed_model=self.embed_model
            )
            print("Loaded existing index")
        except:
            # Create new index
            self.index = VectorStoreIndex([], embed_model=self.embed_model)
            print("Created new index")
    
    def _generate_tags(self, content: str) -> List[str]:
        """Generate tags for content using LLM."""
        try:
            prompt = config.tag_generation_prompt.format(
                max_tags=config.max_tags,
                content=content
            )
            response = self.llm.complete(prompt)
            tags_text = response.text.strip()
            
            # Parse comma-separated tags
            tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()]
            return tags[:config.max_tags]
        except Exception as e:
            print(f"Error generating tags: {e}")
            return []
    
    def _save_memory_file(self, memory: Memory) -> str:
        """Save memory to file and return file path."""
        filename = f"{memory.user_id}_{memory.id}.json"
        filepath = self.memories_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(memory.model_dump(), f, ensure_ascii=False, indent=2, default=str)
        
        return str(filepath)
    
    def store_memory(self, memory_input: MemoryInput) -> Memory:
        """Store a new memory."""
        # Create memory object
        memory = Memory(
            id=str(uuid.uuid4()),
            content=memory_input.content,
            user_id=memory_input.user_id,
            metadata=memory_input.metadata or {}
        )
        
        # Generate tags
        memory.tags = self._generate_tags(memory.content)
        
        # Save to file
        filepath = self._save_memory_file(memory)
        
        # Create document for indexing
        doc_metadata = {
            "memory_id": memory.id,
            "user_id": memory.user_id,
            "tags": memory.tags,
            "created_at": memory.created_at.isoformat(),
            "filepath": filepath
        }
        
        document = Document(
            text=memory.content,
            metadata=doc_metadata
        )
        
        # Add to index
        self.index.insert(document)
        
        # Persist index
        self.index.storage_context.persist(persist_dir=str(self.index_dir))
        
        return memory
    
    def query_memories(self, query: MemoryQuery) -> MemoryResponse:
        """Query memories based on similarity."""
        # Create query engine
        query_engine = self.index.as_query_engine(
            similarity_top_k=query.limit * 2  # Get more to filter by user
        )
        
        # Build query string
        query_str = query.query
        if query.tags:
            query_str += f" Tags: {', '.join(query.tags)}"
        
        # Execute query
        response = query_engine.query(query_str)
        
        results = []
        for node in response.source_nodes:
            # Filter by user_id
            if node.metadata.get("user_id") != query.user_id:
                continue
            
            # Filter by tags if specified
            if query.tags:
                node_tags = node.metadata.get("tags", [])
                if not any(tag in node_tags for tag in query.tags):
                    continue
            
            # Load full memory from file
            try:
                filepath = node.metadata.get("filepath")
                if filepath and os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        memory_data = json.load(f)
                    
                    memory = Memory(**memory_data)
                    result = MemoryResult(
                        memory=memory,
                        score=node.score or 0.0
                    )
                    results.append(result)
            except Exception as e:
                print(f"Error loading memory from {filepath}: {e}")
                continue
            
            if len(results) >= query.limit:
                break
        
        return MemoryResponse(
            results=results,
            total=len(results)
        )
    
    def get_user_memories(self, user_id: str, limit: int = 10) -> List[Memory]:
        """Get all memories for a user."""
        memories = []
        
        for filepath in self.memories_dir.glob(f"{user_id}_*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
                memory = Memory(**memory_data)
                memories.append(memory)
            except Exception as e:
                print(f"Error loading memory from {filepath}: {e}")
                continue
        
        # Sort by creation time (newest first)
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories[:limit]
    
    def get_user_tags(self, user_id: str) -> List[str]:
        """Get all unique tags for a user."""
        tags = set()
        memories = self.get_user_memories(user_id, limit=1000)  # Get more memories for tags
        
        for memory in memories:
            tags.update(memory.tags)
        
        return sorted(list(tags))
    
    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Delete a memory by ID and user ID."""
        filepath = self.memories_dir / f"{user_id}_{memory_id}.json"
        
        if filepath.exists():
            try:
                filepath.unlink()
                # Note: We don't remove from vector index as it's complex
                # In production, you might want to rebuild the index periodically
                return True
            except Exception as e:
                print(f"Error deleting memory file {filepath}: {e}")
                return False
        return False
    
    def get_memory_by_id(self, memory_id: str, user_id: str) -> Optional[Memory]:
        """Get a specific memory by ID and user ID."""
        filepath = self.memories_dir / f"{user_id}_{memory_id}.json"
        
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
                return Memory(**memory_data)
            except Exception as e:
                print(f"Error loading memory from {filepath}: {e}")
                return None
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_memories = 0
        users = set()
        
        for filepath in self.memories_dir.glob("*.json"):
            total_memories += 1
            # Extract user_id from filename pattern: {user_id}_{memory_id}.json
            filename = filepath.stem
            if "_" in filename:
                user_id = filename.split("_")[0]
                users.add(user_id)
        
        return {
            "total_memories": total_memories,
            "total_users": len(users),
            "data_directory": str(self.data_dir),
            "memories_directory": str(self.memories_dir),
            "index_directory": str(self.index_dir)
        }