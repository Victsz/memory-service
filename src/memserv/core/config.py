"""Configuration for memory service."""
import os
from typing import Optional
from pydantic import BaseModel, Field, model_validator 


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
    data_dir: str = Field(default_factory=lambda: os.getenv("DATA_DIR", None))
    memories_dir: str = None
    index_dir: str = None
    
    # Service settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    
    # LLM settings for tagging
    max_tags: int = Field(default_factory=lambda: int(os.getenv("MAX_TAGS", "5")))
    common_tags: list[str] = Field(default=["tech","life","investment","knowledge","todo","relationship"])
    
    # Retry configuration
    retry_max_attempts: int = Field(default_factory=lambda: int(os.getenv("RETRY_MAX_ATTEMPTS", "3")))
    retry_delay_seconds: int = Field(default_factory=lambda: int(os.getenv("RETRY_DELAY_SECONDS", "10")))
    tag_generation_max_retries: int = Field(default_factory=lambda: int(os.getenv("TAG_GENERATION_MAX_RETRIES", "2")))
    
    # Content expansion settings
    word_length_estimate: float = Field(default_factory=lambda: float(os.getenv("WORD_LENGTH_ESTIMATE", "4.5")))
    short_content_threshold: int = Field(default_factory=lambda: int(os.getenv("SHORT_CONTENT_THRESHOLD", "100")))
    
    # Query settings
    query_similarity_multiplier: int = Field(default_factory=lambda: int(os.getenv("QUERY_SIMILARITY_MULTIPLIER", "2")))
    user_tags_memory_limit: int = Field(default_factory=lambda: int(os.getenv("USER_TAGS_MEMORY_LIMIT", "1000")))
    
    # Prompt templates
    tag_generation_prompt: str = Field(
        default_factory=lambda: os.getenv("TAG_GENERATION_PROMPT", 
            "Generate relevant tags for the following content. "
            "Generate a maximum of {max_tags} tags, can be fewer, that best describe the content."
            "Consider 1 or 2 tags choosing from when appropriate: {common_tags}. The tags should be closly relevant to the content. \\n\\n"
            "Content: {content}\\n\\n"
            "Return your response as a JSON object with a 'tags' field containing an array of strings."
        )
    )
    
    content_expansion_prompt: str = Field(
        default_factory=lambda: os.getenv("CONTENT_EXPANSION_PROMPT",
            "Enhance and expand the following short content while preserving its original meaning. "
            "Add clear explanations for key concepts and ideas to make the content more comprehensive. "
            "Keep the expansion concise (within 500 tokens) and focus on clarity and depth.\\n\\n"
            "Original content: {content}\\n\\n"
            "Return your response as a JSON object with an 'enhanced_content' field containing the expanded text."
        )
    )

        

    @model_validator(mode='after')
    def set_memories_dir(self):
        # 如果 memories_dir 已经被设置，则不做任何改动
        if self.memories_dir is not None:
            return self
        
        # 否则，使用 data_dir 的值来设置 memories_dir
        if self.data_dir is not None:
            self.memories_dir = f"{self.data_dir}/memories"
        else:
            # 如果 data_dir 也没有设置，你可能需要抛出一个错误
            raise ValueError("data_dir must be set to determine memories_dir")
            
        return self
    @model_validator(mode='after')
    def set_index_dir(self):
        # 如果 index_dir已经被设置，则不做任何改动
        if self.index_dir is not None:
            return self
        
        # 否则，使用 data_dir 的值来设置 memories_dir
        if self.data_dir is not None:
            self.index_dir= f"{self.data_dir}/index"
        else:
            # 如果 data_dir 也没有设置，你可能需要抛出一个错误
            raise ValueError("data_dir must be set to determine memories_dir")
            
        return self

# Global config instance


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    config = Config()
    print(f"{config.api_base=}")
    print(f"{config.data_dir=}")
    print(f"{config.memories_dir=}")
    print(f"{config.index_dir=}")

else:    
    config = Config()