"""Configuration for memory service."""
import os
from typing import Optional
from pydantic import BaseModel, Field


class Config(BaseModel):
    """Configuration settings for the memory service."""
    
    # OpenAI-like API settings (默认使用 SiliconFlow 配置)
    api_key: str = Field(default_factory=lambda: os.getenv("API_KEY", ""))
    api_base: str = Field(default_factory=lambda: os.getenv("API_BASE", "https://api.siliconflow.cn/v1"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "Pro/deepseek-ai/DeepSeek-V3"))
    
    # Model settings
    context_window: int = Field(default_factory=lambda: int(os.getenv("CONTEXT_WINDOW", "64000")))
    temperature: float = Field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.0")))
    embed_batch_size: int = Field(default_factory=lambda: int(os.getenv("EMBED_BATCH_SIZE", "10")))
    
    # Storage settings
    data_dir: str = Field(default="./data")
    memories_dir: str = Field(default="./data/memories")
    index_dir: str = Field(default="./data/index")
    
    # Service settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    
    # LLM settings for tagging
    max_tags: int = Field(default=5)
    tag_generation_prompt: str = Field(
        default="""Based on the following content, generate up to {max_tags} relevant tags that describe the main topics, themes, or categories. 
Return only the tags as a comma-separated list, no explanations.

Content: {content}

Tags:"""
    )


# Global config instance
config = Config()