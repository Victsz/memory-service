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
        
        # Initialize LLM and embedding model following LlamaIndex guidance
        print("🤖 Initializing LLM and embedding models...")
        self.embed_model = OpenAILikeEmbedding(
            model_name=config.embedding_model,
            api_base=config.api_base,
            api_key=config.api_key,
            embed_batch_size=config.embed_batch_size,
        )
        
        self.llm = OpenAILike(
            model=config.llm_model,
            api_base=config.api_base,
            api_key=config.api_key,
            context_window=config.context_window,
            is_chat_model=True,
            is_function_calling_model=False,
            temperature=config.temperature,
        )
        
        # Set global settings
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        
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
                common_tags = config.common_tags, 
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
    

    
    def store_memory(self, memory_input: MemoryInput, custom_tags: Optional[List[str]] = None) -> Memory:
        """Store a new memory with optional custom tags."""
        # Create memory object
        memory = Memory(
            id=str(uuid.uuid4()),
            content=memory_input.content,
            user_id=memory_input.user_id,
            metadata=memory_input.metadata or {}
        )
        
        # Generate tags using LLM
        auto_generated_tags = self._generate_tags(memory.content)
        
        # Combine auto-generated tags with custom tags
        all_tags = auto_generated_tags.copy()
        if custom_tags:
            # Add custom tags, avoiding duplicates
            for tag in custom_tags:
                if tag.strip() and tag.strip() not in all_tags:
                    all_tags.append(tag.strip())
        
        memory.tags = all_tags
        
        # Create document for indexing
        doc_metadata = {
            "memory_id": memory.id,
            "user_id": memory.user_id,
            "tags": memory.tags,
            "created_at": memory.created_at.isoformat(),
            "metadata": memory.metadata
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
            
            # Create Memory object directly from node data (no file loading needed)
            try:
                memory = Memory(
                    id=node.metadata.get("memory_id"),
                    content=node.text,
                    user_id=node.metadata.get("user_id"),
                    tags=node.metadata.get("tags", []),
                    created_at=datetime.fromisoformat(node.metadata.get("created_at")),
                    metadata=node.metadata.get("metadata", {})
                )
                
                result = MemoryResult(
                    memory=memory,
                    score=node.score or 0.0
                )
                results.append(result)
            except Exception as e:
                print(f"Error creating memory from node: {e}")
                continue
            
            if len(results) >= query.limit:
                break
        
        return MemoryResponse(
            results=results,
            total=len(results)
        )
    
    def get_user_memories(self, user_id: str, limit: int = 10, tags: Optional[List[str]] = None) -> List[Memory]:
        """Get all memories for a user with optional tag filtering using LlamaIndex storage context."""
        # Access the docstore to get all documents
        docstore = self.index.storage_context.docstore
        all_docs = docstore.docs
        
        # Filter documents by user_id and optionally by tags
        memories = []
        for doc_id, doc in all_docs.items():
            # Filter by user_id
            if doc.metadata.get("user_id") != user_id:
                continue
            
            # Filter by tags if specified
            if tags:
                doc_tags = doc.metadata.get("tags", [])
                if not any(tag in doc_tags for tag in tags):
                    continue
            
            # Create Memory object from document metadata and text
            try:
                memory = Memory(
                    id=doc.metadata.get("memory_id"),
                    content=doc.text,
                    user_id=doc.metadata.get("user_id"),
                    tags=doc.metadata.get("tags", []),
                    created_at=datetime.fromisoformat(doc.metadata.get("created_at")),
                    metadata=doc.metadata.get("metadata", {})
                )
                memories.append(memory)
            except Exception as e:
                print(f"Error creating memory from document {doc_id}: {e}")
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
        """Delete a memory by ID and user ID using LlamaIndex storage context."""
        # Access the docstore to find and delete the specific document
        docstore = self.index.storage_context.docstore
        all_docs = docstore.docs
        
        # Find the document with matching memory_id and user_id
        doc_to_delete = None
        for doc_id, doc in all_docs.items():
            if (doc.metadata.get("memory_id") == memory_id and 
                doc.metadata.get("user_id") == user_id):
                doc_to_delete = doc_id
                break
        
        if doc_to_delete:
            try:
                # Delete from docstore
                docstore.delete_document(doc_to_delete)
                
                # Persist the updated index
                self.index.storage_context.persist(persist_dir=str(self.index_dir))
                
                return True
            except Exception as e:
                print(f"Error deleting memory from index: {e}")
                return False
        
        return False
    
    def get_memory_by_id(self, memory_id: str, user_id: str) -> Optional[Memory]:
        """Get a specific memory by ID and user ID using LlamaIndex storage context."""
        # Access the docstore to find the specific document
        docstore = self.index.storage_context.docstore
        all_docs = docstore.docs
        
        # Find the document with matching memory_id and user_id
        for doc_id, doc in all_docs.items():
            if (doc.metadata.get("memory_id") == memory_id and 
                doc.metadata.get("user_id") == user_id):
                
                # Create Memory object from document
                try:
                    memory = Memory(
                        id=doc.metadata.get("memory_id"),
                        content=doc.text,
                        user_id=doc.metadata.get("user_id"),
                        tags=doc.metadata.get("tags", []),
                        created_at=datetime.fromisoformat(doc.metadata.get("created_at")),
                        metadata=doc.metadata.get("metadata", {})
                    )
                    return memory
                except Exception as e:
                    print(f"Error creating memory from document {doc_id}: {e}")
                    break
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics using LlamaIndex storage context."""
        # Access the docstore to get all documents
        docstore = self.index.storage_context.docstore
        all_docs = docstore.docs
        
        total_memories = len(all_docs)
        users = set()
        
        # Extract unique user IDs from document metadata
        for doc_id, doc in all_docs.items():
            user_id = doc.metadata.get("user_id")
            if user_id:
                users.add(user_id)
        
        return {
            "total_memories": total_memories,
            "total_users": len(users),
            "data_directory": str(self.data_dir),
            "memories_directory": str(self.memories_dir),
            "index_directory": str(self.index_dir)
        }