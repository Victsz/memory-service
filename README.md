# Memory Service 项目描述

## 1. 模块概述

**功能定位**：基于 LlamaIndex 和 FastMCP 构建的 AI 驱动的个人记忆存储与检索系统，支持语义搜索和多接口访问。

**依赖关系**：
- **核心框架**：FastMCP (MCP 服务器)、FastAPI (REST API)、LlamaIndex (向量存储与检索)
- **AI 模型**：OpenAI-like API (支持 OpenAI、SiliconFlow 等提供商)
- **数据处理**：Pydantic (数据验证)、python-dotenv (配置管理)
- **日志系统**：loguru (结构化日志)、TimedRotatingFileHandler (日志轮转)

## 2. 快速使用指南

### 接口说明

#### 核心服务接口 (MemoryService)
```python
# 存储记忆
store_memory(memory_input: MemoryInput) -> Memory

# 查询记忆 
query_memories(query: MemoryQuery) -> MemoryResponse

# 获取用户记忆列表
get_user_memories(user_id: str, limit: int = 10) -> List[Memory]
```

#### FastMCP 工具接口
```python
# MCP 工具 (Actions)
store_memory(content: str, user_id: str, metadata: Dict) -> Memory
query_memories(query: str, user_id: str, limit: int, tags: List[str]) -> MemoryResponse
get_user_tags(user_id: str) -> List[str]

# MCP 资源 (Resources)
memory://users/{user_id}/memories  # 用户记忆列表
memory://users/{user_id}/tags      # 用户标签列表
memory://stats                     # 服务统计信息
```

#### REST API 接口
```python
POST /memories                     # 存储记忆
POST /memories/query              # 查询记忆
GET  /users/{user_id}/memories    # 获取用户记忆
GET  /users/{user_id}/tags        # 获取用户标签
GET  /memories/{user_id}/{memory_id}  # 获取特定记忆
PUT  /memories/{user_id}/{memory_id}   # 更新记忆 (特殊逻辑)
DELETE /memories/{user_id}/{memory_id}  # 删除记忆 (软删除)
GET  /stats                       # 服务统计
GET  /health                      # 健康检查
```

### 调用示例

#### 启动服务
```bash
# 同时启动 FastAPI 和 MCP 服务器
uv run python main.py --mode both --mcp-port 8001

# 仅启动 MCP 服务器 (STDIO 模式，用于 Claude Desktop)
python -c "from main import run_mcp_stdio; run_mcp_stdio()"
```

#### FastMCP 客户端调用
```python
import asyncio
from fastmcp import Client

async def example():
    async with Client("http://localhost:8001/mcp") as client:
        # 存储记忆
        result = await client.call_tool("store_memory", {
            "content": "学习了 FastMCP 的使用方法",
            "user_id": "victor",
            "metadata": {"source": "documentation"}
        })
        
        # 查询记忆
        results = await client.call_tool("query_memories", {
            "query": "FastMCP 相关内容",
            "user_id": "victor",
            "limit": 5
        })
```

#### REST API 调用
```bash
# 存储记忆
curl -X POST http://localhost:8000/memories \
  -H "Content-Type: application/json" \
  -d '{"content": "今天学习了向量数据库", "user_id": "victor"}'

# 查询记忆
curl -X POST http://localhost:8000/memories/query \
  -H "Content-Type: application/json" \
  -d '{"query": "向量数据库", "user_id": "victor", "limit": 5}'
```

### 配置项

#### 核心配置参数及默认值
```python
# API 配置
API_KEY=""                                    # API 密钥 (必需)
API_BASE="https://api.siliconflow.cn/v1"     # API 基础 URL
EMBEDDING_MODEL="BAAI/bge-m3"                # 嵌入模型
LLM_MODEL="Pro/deepseek-ai/DeepSeek-V3"      # 语言模型

# 模型参数
CONTEXT_WINDOW=64000                         # 上下文窗口大小
TEMPERATURE=0.0                              # 生成温度
EMBED_BATCH_SIZE=10                          # 嵌入批处理大小

# 存储配置
DATA_DIR="./data"                            # 数据目录

# 标签生成配置
MAX_TAGS=5                                   # 最大标签数量
TAG_GENERATION_MAX_RETRIES=2                 # 标签生成重试次数

# 内容扩展配置
SHORT_CONTENT_THRESHOLD=100                  # 短内容阈值 (字符数)
WORD_LENGTH_ESTIMATE=4.5                     # 平均单词长度估算

# 查询配置
QUERY_SIMILARITY_MULTIPLIER=2                # 查询相似度倍数
USER_TAGS_MEMORY_LIMIT=1000                  # 用户标签记忆限制
```

## 3. 核心机制与实现

### 特殊逻辑

#### AI 驱动的自动标签生成
- **机制**：使用 LlamaIndex 结构化输出 + Pydantic 模型确保标签格式正确
- **重试策略**：最多重试 2 次，失败时返回空标签列表，确保服务不中断
- **标签一致性**：考虑常用标签 `["tech","life","investment","knowledge","todo","relationship"]` 提高标签复用率

#### 智能内容扩展
- **触发条件**：内容长度 < 100 字符时自动触发
- **扩展策略**：保持原意的前提下增加上下文和解释，限制在 500 tokens 内
- **容错机制**：扩展失败时优雅降级到原始内容，确保数据完整性

#### 多接口架构设计
- **分层设计**：核心业务逻辑在 `service.py`，接口层 (`api.py`, `mcp_interface.py`) 为薄包装
- **协议抽象**：使用 `MemoryServiceProtocol` 定义服务接口，支持不同存储后端
- **用户隔离**：所有操作基于 `user_id` 进行数据隔离，确保多用户安全

#### 向量存储与检索优化
- **存储结构**：记忆文件 `./data/memories/{user_id}_{memory_id}.json`，向量索引 `./data/index/`
- **查询策略**：语义相似度搜索 + 标签过滤，支持混合检索模式
- **性能优化**：批量嵌入处理，索引预加载，减少首次查询延迟

#### 文档元数据结构 (Document Metadata)
- **核心字段**：`memory_id`、`user_id`、`created_at`、`is_archived`、`metadata`
- **标签索引**：`user_tags` (用户标签列表)、`tags` (完整标签列表，向后兼容)
- **系统字段**：`system_{key}` (系统标签布尔字段，如 `system_original` 对应 `original_sys:{id}`)
- **过滤优化**：结构化元数据支持高效的 MetadataFilters 查询，避免全文档扫描
- **数据一致性**：所有 Document 元数据与 Memory 模型保持同步，确保查询结果准确性

#### 错误处理与容错机制
- **Fail Fast 原则**：配置缺失或依赖不可用时立即报错，避免静默失败
- **AI 处理容错**：标签生成和内容扩展失败时有明确的降级策略
- **日志轮转**：7 天日志保留，按日轮转，便于问题追踪和系统监控

#### 记忆更新机制 (Update Memory)
- **非原地更新**：采用"新增+软删除"策略，而非直接修改原记忆内容
- **版本追踪**：新记忆包含 `original_sys:{old_id}` 系统标签，建立版本关联关系
- **原子性保证**：先创建新记忆成功后再归档原记忆，确保数据不丢失
- **软删除实现**：通过添加 `archived_sys` 系统标签实现逻辑删除，保留数据完整性
- **重试机制**：关键操作（创建新记忆、归档原记忆）都有重试保护，提高成功率

#### 软删除与系统标签
- **系统标签设计**：`archived_sys` (软删除标记)、`original_sys:{id}` (版本引用)
- **标签隔离**：系统标签与用户标签严格分离，LLM 生成的标签会过滤系统保留标签
- **查询过滤**：默认查询自动排除已归档记忆，可通过 `include_archived` 参数控制
- **数据完整性**：删除操作不会物理删除数据，仅添加归档标签，支持数据恢复

#### 服务预初始化
- **预热机制**：启动时预加载 MemoryStore 单例，避免首次请求延迟
- **配置验证**：启动时验证 API 密钥等关键配置，提前发现配置问题
- **开发模式**：支持 `--reload` 和 `--skip-warmup` 参数，提升开发体验

### 项目结构
```
src/memserv/
├── core/                    # 核心业务逻辑
│   ├── config.py           # 配置管理 (Pydantic 模型)
│   ├── models.py           # 数据模型 (Memory, MemoryQuery 等)
│   ├── memory_store.py     # LlamaIndex 存储实现
│   └── service.py          # 服务层抽象 (业务逻辑)
├── interface/              # 接口层
│   ├── api.py              # FastAPI REST 接口
│   └── mcp_interface.py    # FastMCP 服务器接口
└── __init__.py             # 包入口点
```

### 技术特色
1. **多协议支持**：同一服务逻辑同时支持 MCP 和 REST API
2. **AI 增强**：自动标签生成和内容扩展，提升记忆检索效果
3. **容错设计**：AI 处理失败时的优雅降级，确保服务稳定性
4. **用户隔离**：基于 user_id 的完全数据隔离
5. **版本化更新**：非破坏性的记忆更新机制，支持版本追踪和数据恢复
6. **软删除架构**：通过系统标签实现逻辑删除，保证数据完整性和可恢复性
7. **开发友好**：支持热重载、日志轮转、配置验证等开发特性