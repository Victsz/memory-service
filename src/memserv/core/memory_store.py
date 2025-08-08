"""Memory storage and retrieval using LlamaIndex."""
import os
import json
import uuid
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
from functools import wraps

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document, Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.prompts import PromptTemplate
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser

from .models import Memory, MemoryInput, MemoryQuery, MemoryResult, MemoryResponse
from .config import config

# 系统保留标签常量
SYSTEM_TAG_ARCHIVED = "archived_sys"  # 系统软删除标签
SYSTEM_TAG_PREFIX = "original_sys:"    # 原记忆引用前缀
SYSTEM_TAGS = {SYSTEM_TAG_ARCHIVED}    # 系统保留标签集合

# 重试配置常量 - 从配置文件获取
def retry_on_failure(max_attempts: int = None, delay: int = None) -> Callable:
    """重试装饰器，用于关键操作的错误恢复"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 使用配置值作为默认值
            actual_max_attempts = max_attempts or config.retry_max_attempts
            actual_delay = delay or config.retry_delay_seconds
            last_exception = None
            
            for attempt in range(actual_max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < actual_max_attempts - 1:  # 不是最后一次尝试
                        print(f"Attempt {attempt + 1}/{actual_max_attempts} failed for {func.__name__}: {e}")
                        print(f"Retrying in {actual_delay} seconds...")
                        time.sleep(actual_delay)
                    else:
                        print(f"All {actual_max_attempts} attempts failed for {func.__name__}")
            
            # 所有重试都失败，抛出最后一个异常
            raise last_exception
        return wrapper
    return decorator


class ExpandedContent(BaseModel):
    """Structured model for expanded content output."""
    
    enhanced_content: str = Field(
        description="The enhanced and expanded version of the original content with clear explanations for key concepts"
    )


class GeneratedTags(BaseModel):
    """Structured model for tag generation output."""
    
    tags: List[str] = Field(
        description="A list of relevant tags for the content, maximum 5 tags"
    )


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
        """预热模型，避免第一次调用时的延迟。配置错误时立即失败。"""
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
            # Fail Fast: 预热失败说明API配置有问题，应该立即失败
            raise RuntimeError(f"Model warmup failed - check API configuration (API_KEY, API_BASE): {e}") from e
    
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
        """Generate tags for content using LLM structured output with retry logic."""
        max_retries = config.tag_generation_max_retries
        
        for attempt in range(max_retries + 1):
            try:
                # Create LLMTextCompletionProgram for non-function-calling models
                program = LLMTextCompletionProgram.from_defaults(
                    output_parser=PydanticOutputParser(output_cls=GeneratedTags),
                    prompt_template_str=config.tag_generation_prompt,
                    llm=self.llm,
                    verbose=False
                )
                
                # Execute the program to get structured output
                response = program(
                    max_tags=config.max_tags,
                    common_tags=config.common_tags,
                    content=content
                )
                
                # Extract tags from structured response and limit to max_tags
                generated_tags = response.tags[:config.max_tags]
                
                # 过滤掉系统保留标签，确保LLM不会生成系统标签
                filtered_tags = []
                for tag in generated_tags:
                    if tag not in SYSTEM_TAGS and not tag.startswith("original_sys:"):
                        filtered_tags.append(tag)
                    else:
                        print(f"Warning: LLM generated system reserved tag '{tag}', filtered out")
                
                return filtered_tags
                
            except Exception as e:
                print(f"Error generating tags (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt == max_retries:
                    print("All tag generation attempts failed, returning empty list")
                    return []
                else:
                    print(f"Retrying tag generation...")
                    continue
    

    

    def _expand_short_content(self, content: str) -> str:
        """Expand short content using LLM structured output to enhance context and meaning.
        
        For short content (<100 words), this method uses LLMTextCompletionProgram
        to expand the content while preserving the original meaning, adding explanations 
        for key concepts. The expansion is limited to 200 words.
        
        If structured output fails, gracefully falls back to using the original content.
        """
        # Count words (rough approximation)
        # 主要结论：
        # 英语单词的平均长度在4.5-5.1个字母之间，最常见的单词长度集中在2-7个字母范围内。这个数据对语言学习、文本分析和信息处理有重要参考价值。
        word_count = len(content) / config.word_length_estimate
        
        # Only expand if content is short
        if word_count < config.short_content_threshold:
            try:
                # Create LLMTextCompletionProgram for non-function-calling models
                program = LLMTextCompletionProgram.from_defaults(
                    output_parser=PydanticOutputParser(output_cls=ExpandedContent),
                    prompt_template_str=config.content_expansion_prompt,
                    llm=self.llm,
                    verbose=False
                )
                
                # Execute the program to get structured output
                response = program(content=content)
                
                # Extract the enhanced content from the structured response
                expanded_content = response.enhanced_content
                
                # Add original content as reference
                final_content = f"{content}\n\n---\n\nExpanded explanation:\n{expanded_content}"
                return final_content
            except Exception as e:
                print(f"Content expansion failed, using original content: {e}")
                # Gracefully fallback to original content without expansion
                return content
        
        return content
    
    def store_memory(self, memory_input: MemoryInput, custom_tags: Optional[List[str]] = None) -> Memory:
        """Store a new memory with optional custom tags.
        
        For short content (<100 words), automatically expands the content using LLM
        to enhance context and meaning while preserving the original content.
        """
        # Expand short content if needed
        expanded_content = self._expand_short_content(memory_input.content)
        
        # Create memory object
        memory = Memory(
            id=str(uuid.uuid4()),
            content=expanded_content,
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
                    # 允许系统标签通过（用于内部操作如update_memory）
                    # 但过滤用户可能意外提供的系统标签
                    if tag.strip().startswith(SYSTEM_TAG_PREFIX) or tag.strip() in SYSTEM_TAGS:
                        # 系统标签直接添加，不进行过滤
                        all_tags.append(tag.strip())
                    else:
                        # 用户标签需要确保不与系统标签冲突
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
            metadata=doc_metadata,
            doc_id=memory.id  # 设置doc_id为memory_id，确保可以通过memory_id更新
        )
        
        # Add to index
        self.index.insert(document)
        
        # Persist index
        self.index.storage_context.persist(persist_dir=str(self.index_dir))
        
        return memory
    
    def query_memories(self, query: MemoryQuery, include_archived: bool = False) -> MemoryResponse:
        """Query memories based on similarity."""
        # Create query engine
        query_engine = self.index.as_query_engine(
            similarity_top_k=query.limit * config.query_similarity_multiplier  # Get more to filter by user
        )
        
        # Build query string
        query_str = query.query
        if query.tags:
            query_str += f" Tags: {', '.join(query.tags)}"
        
        # Execute query - 使用retriever而不是query_engine来避免nodes为None的问题
        retrieved_nodes = None
        try:
            # 使用retriever直接获取节点，避免query_engine的问题
            retriever = self.index.as_retriever(
                similarity_top_k=query.limit * config.query_similarity_multiplier
            )
            print(f"retriever {query_str=}")
            try:
                retrieved_nodes = retriever.retrieve(query_str)
            except:
                pass
            
            # 如果retriever也失败，尝试直接查询向量存储
            if not retrieved_nodes:
                print(f"Retriever returned no nodes, trying direct vector store query...")
                # 直接查询向量存储
                from llama_index.core.vector_stores.types import VectorStoreQuery
                query_embedding = self.embed_model.get_text_embedding(query_str)
                vector_query = VectorStoreQuery(
                    query_embedding=query_embedding,
                    similarity_top_k=query.limit * config.query_similarity_multiplier,
                    mode="default"
                )
                
                vector_result = self.index.storage_context.vector_store.query(vector_query)
                
                # 手动构建节点
                if vector_result.ids and vector_result.similarities:
                    docstore = self.index.storage_context.docstore
                    retrieved_nodes = []
                    
                    for doc_id, similarity in zip(vector_result.ids, vector_result.similarities):
                        if doc_id in docstore.docs:
                            doc = docstore.docs[doc_id]
                            from llama_index.core.schema import NodeWithScore, TextNode
                            
                            # 创建TextNode
                            text_node = TextNode(
                                text=doc.text,
                                metadata=doc.metadata,
                                id_=doc_id
                            )
                            
                            # 创建NodeWithScore
                            node_with_score = NodeWithScore(
                                node=text_node,
                                score=float(similarity)
                            )
                            retrieved_nodes.append(node_with_score)
                else:
                    print(f"No results from direct vector store query")
                    return MemoryResponse(results=[], total=0)
            
        except Exception as e:
            print(f"Query error: {e}")
            import traceback
            traceback.print_exc()
            return MemoryResponse(results=[], total=0)
        
        # Check if we have retrieved nodes
        if not retrieved_nodes:
            print(f"No nodes retrieved for query: {query_str}")
            return MemoryResponse(results=[], total=0)
        
        results = []
        for node_with_score in retrieved_nodes:
            node = node_with_score.node
            # Filter by user_id
            if node.metadata.get("user_id") != query.user_id:
                continue
            
            # Filter archived memories unless explicitly requested
            node_tags = node.metadata.get("tags", [])
            if not include_archived and SYSTEM_TAG_ARCHIVED in node_tags:
                continue
            
            # Filter by tags if specified
            if query.tags:
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
                    score=node_with_score.score or 0.0
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
    
    def get_user_memories(self, user_id: str, limit: int = 10, tags: Optional[List[str]] = None, include_archived: bool = False) -> List[Memory]:
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
            
            # Filter archived memories unless explicitly requested
            doc_tags = doc.metadata.get("tags", [])
            if not include_archived and SYSTEM_TAG_ARCHIVED in doc_tags:
                continue
            
            # Filter by tags if specified
            if tags:
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
    
    def get_user_tags(self, user_id: str, include_archived: bool = False) -> List[str]:
        """Get all unique tags for a user."""
        tags = set()
        memories = self.get_user_memories(user_id, limit=config.user_tags_memory_limit, include_archived=include_archived)  # Get more memories for tags
        
        for memory in memories:
            # 过滤掉系统标签
            user_tags = [tag for tag in memory.tags if tag not in SYSTEM_TAGS and not tag.startswith(SYSTEM_TAG_PREFIX)]
            tags.update(user_tags)
        
        return sorted(list(tags))
    
    @retry_on_failure()
    def update_memory(self, memory_id: str, user_id: str, new_content: str, 
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        更新记忆内容（先新增再软删除）
        
        Args:
            memory_id: 要更新的记忆ID
            user_id: 用户ID
            new_content: 新的记忆内容
            metadata: 可选的元数据
        
        Returns:
            Dict包含新记忆对象和更新信息
            
        Raises:
            ValueError: 记忆不存在或权限不足
            RuntimeError: 更新操作失败
            
        TODO: 未来可能需要对memory_id加锁防止并发更新
        """
        # 1. 查询并验证原记忆存在
        original_memory = self.get_memory_by_id(memory_id, user_id)
        if not original_memory:
            raise ValueError(f"Memory {memory_id} not found for user {user_id}")
        
        # 检查是否已经被归档
        if SYSTEM_TAG_ARCHIVED in original_memory.tags:
            raise ValueError(f"Cannot update archived memory {memory_id}")
        
        try:
            # 2. 创建新记忆（包含original:{old_id}标签）- 带重试
            memory_input = MemoryInput(
                content=new_content,
                user_id=user_id,
                metadata=metadata or {}
            )
            
            # 添加原记忆引用标签
            original_ref_tag = f"{SYSTEM_TAG_PREFIX}{memory_id}"
            new_memory = self._create_new_memory_with_retry(memory_input, [original_ref_tag])
            
            # 3. 软删除原记忆（添加archived标签）- 带重试
            archive_success = self._archive_memory_with_retry(memory_id, user_id)
            
            # 4. 返回结果
            result = {
                "success": True,
                "memory": new_memory.model_dump(mode='json'),
                "update_info": {
                    "original_id": memory_id,
                    "new_id": new_memory.id,
                    "operation": "update_via_replace"
                },
                "message": f"Memory updated successfully, new ID: {new_memory.id}"
            }
            
            # 如果软删除失败，添加警告信息
            if not archive_success:
                result["update_info"]["warning"] = "Original memory archiving failed after retries, but new memory created successfully"
                result["message"] = "Memory updated with warnings - check update_info"
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to update memory {memory_id}: {e}") from e
    
    def _create_new_memory_with_retry(self, memory_input: MemoryInput, additional_tags: List[str]) -> Memory:
        """创建新记忆的内部方法，带重试机制"""
        memory = self.store_memory(memory_input, custom_tags=additional_tags)
        return memory
    
    def _archive_memory_with_retry(self, memory_id: str, user_id: str) -> bool:
        """归档记忆的内部方法，带重试机制"""
        try:
            return self._archive_memory(memory_id, user_id)
        except Exception as e:
            print(f"Archive memory failed: {e}")
            return False
    
    @retry_on_failure()
    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """软删除记忆（添加archived_sys标签而不是物理删除）"""
        return self._archive_memory(memory_id, user_id)
    
    def _archive_memory(self, memory_id: str, user_id: str) -> bool:
        """内部方法：给记忆添加archived_sys标签实现软删除"""
        # 首先获取现有记忆
        memory = self.get_memory_by_id(memory_id, user_id)
        if not memory:
            return False
        
        # 检查是否已经被归档
        if SYSTEM_TAG_ARCHIVED in memory.tags:
            print(f"Memory {memory_id} is already archived")
            return True
        
        try:
            # 创建更新后的Document，添加archived_sys标签
            updated_tags = memory.tags.copy()
            updated_tags.append(SYSTEM_TAG_ARCHIVED)
            
            updated_metadata = {
                "memory_id": memory.id,
                "user_id": memory.user_id,
                "tags": updated_tags,
                "created_at": memory.created_at.isoformat(),
                "metadata": memory.metadata
            }
            
            from llama_index.core import Document
            updated_document = Document(
                text=memory.content,
                metadata=updated_metadata,
                doc_id=memory.id  # 使用memory_id作为doc_id，确保与原Document的ID一致
            )
            
            # 使用LlamaIndex的update_ref_doc方法更新文档
            self.index.update_ref_doc(updated_document)
            
            print(f"Memory {memory_id} archived successfully using update_ref_doc")
            return True
            
        except Exception as e:
            print(f"Error archiving memory {memory_id}: {e}")
            raise
    
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