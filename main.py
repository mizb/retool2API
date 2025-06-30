import json
import os
import time
import uuid
import asyncio
import threading
from typing import Any, Dict, List, Optional, TypedDict, Union

import httpx
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field


# Retool Account Management
class RetoolAccount(TypedDict):
    domain_name: str
    x_xsrf_token: str
    accessToken: str
    is_valid: bool
    last_used: float
    error_count: int
    agents: List[Dict[str, Any]]


# Global variables
VALID_CLIENT_KEYS: set = set()
RETOOL_ACCOUNTS: List[RetoolAccount] = []
AVAILABLE_MODELS: List[Dict[str, Any]] = []
account_rotation_lock = threading.Lock()
MAX_ERROR_COUNT = 3
ERROR_COOLDOWN = 300  # 5 minutes cooldown for accounts with errors
DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"


# Pydantic Models
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = True
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    name: Optional[str] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class ChatCompletionChoice(BaseModel):
    message: ChatMessage
    index: int = 0
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int] = Field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )


class StreamChoice(BaseModel):
    delta: Dict[str, Any] = Field(default_factory=dict)
    index: int = 0
    finish_reason: Optional[str] = None


class StreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]


# FastAPI App
app = FastAPI(title="Retool OpenAI API Adapter")
security = HTTPBearer(auto_error=False)


def log_debug(message: str):
    """Debug日志函数"""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")


def load_client_api_keys():
    """从client_api_keys.json加载客户端API密钥"""
    global VALID_CLIENT_KEYS
    try:
        with open("client_api_keys.json", "r", encoding="utf-8") as f:
            keys = json.load(f)
            VALID_CLIENT_KEYS = set(keys) if isinstance(keys, list) else set()
            print(f"成功加载 {len(VALID_CLIENT_KEYS)} 个客户端API密钥")
    except FileNotFoundError:
        print("错误: client_api_keys.json未找到。客户端认证将失败。")
        VALID_CLIENT_KEYS = set()
    except Exception as e:
        print(f"加载client_api_keys.json时出错: {e}")
        VALID_CLIENT_KEYS = set()


def load_retool_accounts_from_file():
    """从retool.json加载Retool账户"""
    try:
        with open("retool.json", "r", encoding="utf-8") as f:
            accounts = json.load(f)
            if not isinstance(accounts, list):
                print("警告: retool.json应包含账户对象列表")
                return []
                
            result = []
            for acc in accounts:
                domain_name = acc.get("domain_name")
                x_xsrf_token = acc.get("x_xsrf_token")
                access_token = acc.get("accessToken")
                if domain_name and x_xsrf_token and access_token:
                    result.append({
                        "domain_name": domain_name,
                        "x_xsrf_token": x_xsrf_token,
                        "accessToken": access_token,
                        "is_valid": True,
                        "last_used": 0,
                        "error_count": 0,
                        "agents": []
                    })
            print(f"成功加载 {len(result)} 个Retool账户")
            return result
    except FileNotFoundError:
        print("错误: retool.json未找到。API调用将失败。")
        return []
    except Exception as e:
        print(f"加载retool.json时出错: {e}")
        return []


async def retool_query_agents(client: httpx.AsyncClient, account: Dict[str, Any]):
    """查询账户可用的Agents"""
    url = f"https://{account['domain_name']}/api/agents"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0",
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "x-xsrf-token": account["x_xsrf_token"],
        "Cookie": f"accessToken={account['accessToken']}",
    }
    
    try:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()["agents"]
    except Exception as e:
        log_debug(f"查询账户 {account['domain_name']} 的Agents时出错: {e}")
        return []


async def retool_get_thread_id(client: httpx.AsyncClient, account: Dict[str, Any], agent_id: str):
    """创建新的对话线程"""
    url = f"https://{account['domain_name']}/api/agents/{agent_id}/threads"
    
    payload = {"name": "", "timezone": ""}
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0",
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Content-Type": "application/json",
        "x-xsrf-token": account["x_xsrf_token"],
        "Cookie": f"accessToken={account['accessToken']}",
    }
    
    response = await client.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["id"]


async def retool_send_message(client: httpx.AsyncClient, account: Dict[str, Any], agent_id: str, thread_id: str, message: str):
    """发送消息到线程"""
    url = f"https://{account['domain_name']}/api/agents/{agent_id}/threads/{thread_id}/messages"
    
    payload = {"type": "text", "text": message, "timezone": "Asia/Shanghai"}
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0",
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Content-Type": "application/json",
        "x-xsrf-token": account["x_xsrf_token"],
        "Cookie": f"accessToken={account['accessToken']}",
    }
    
    response = await client.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["content"]["runId"]


async def retool_get_message(client: httpx.AsyncClient, account: Dict[str, Any], agent_id: str, log_id: str):
    """获取消息响应"""
    url = f"https://{account['domain_name']}/api/agents/{agent_id}/logs/{log_id}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0",
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "x-xsrf-token": account["x_xsrf_token"],
        "Cookie": f"accessToken={account['accessToken']}",
    }
    
    for _ in range(300):  # 最多等待300秒
        try:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            if data["status"] == "COMPLETED":
                trace = data["trace"]
                message = trace[-1]["data"]["data"]["content"]
                return message
            else:
                await asyncio.sleep(1)
        except Exception as e:
            log_debug(f"获取消息时出错: {e}")
            return None
    
    return None


def format_messages_for_retool(messages: List[ChatMessage]) -> str:
    """将消息历史格式化为Retool可接受的格式"""
    formatted = ""
    for msg in messages:
        role = "Human" if msg.role == "user" else "Assistant"
        content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
        formatted += f"\n\n{role}: {content}"
    
    # 如果最后一条消息是用户消息，不需要添加Assistant:前缀
    # 如果最后一条是助手消息，需要添加Human:前缀以便模型继续
    if messages and messages[-1].role == "assistant":
        formatted += "\n\nHuman: "
    
    return formatted


def get_best_retool_account(model_id: str) -> Optional[Dict[str, Any]]:
    """获取最佳可用的Retool账户"""
    with account_rotation_lock:
        now = time.time()
        
        # 查找支持此模型的所有agent IDs
        supported_agent_ids = []
        for model in AVAILABLE_MODELS:
            if model["id"] == model_id:
                supported_agent_ids = model["agents"]
                break
        
        if not supported_agent_ids:
            return None
        
        # 筛选出拥有支持此模型的agent且有效的账户
        valid_accounts = []
        for acc in RETOOL_ACCOUNTS:
            has_agent = any(agent["id"] in supported_agent_ids for agent in acc["agents"])
            is_valid = acc["is_valid"] and (
                acc["error_count"] < MAX_ERROR_COUNT or 
                now - acc["last_used"] > ERROR_COOLDOWN
            )
            if has_agent and is_valid:
                # 找出此账户中支持该模型的第一个agent ID
                for agent in acc["agents"]:
                    if agent["id"] in supported_agent_ids:
                        acc["selected_agent_id"] = agent["id"]  # 记录选中的agent ID
                        break
                valid_accounts.append(acc)
        
        if not valid_accounts:
            return None
            
        # 重置冷却期账户的错误计数
        for acc in valid_accounts:
            if acc["error_count"] >= MAX_ERROR_COUNT and now - acc["last_used"] > ERROR_COOLDOWN:
                acc["error_count"] = 0
                
        # 按最后使用时间(最旧优先)和错误计数(最少优先)排序
        valid_accounts.sort(key=lambda x: (x["last_used"], x["error_count"]))
        account = valid_accounts[0]
        account["last_used"] = now
        return account


async def initialize_retool_environment():
    """初始化Retool环境，加载账户和模型"""
    global RETOOL_ACCOUNTS, AVAILABLE_MODELS
    
    # 加载账户
    accounts = load_retool_accounts_from_file()
    if not accounts:
        print("警告: 未找到有效的Retool账户")
        return
    
    # 查询每个账户的可用Agents
    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [retool_query_agents(client, account) for account in accounts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"查询账户 {accounts[i]['domain_name']} 的Agents时出错: {result}")
                accounts[i]["agents"] = []
            else:
                accounts[i]["agents"] = result
                log_debug(f"账户 {accounts[i]['domain_name']} 有 {len(result)} 个Agents")
    
    # 更新全局账户列表
    RETOOL_ACCOUNTS = accounts
    
    # 聚合所有唯一模型 (使用model字段而非name)
    model_map = {}  # 用于映射模型ID到实际模型名称
    all_models = {}
    for account in RETOOL_ACCOUNTS:
        for agent in account["agents"]:
            # 获取实际模型名称，如claude-sonnet-4-20250514
            model_name = agent.get("data", {}).get("model", "unknown")
            # 提取模型系列名称，如claude-sonnet-4
            model_series = model_name.split("-")[0:3]
            model_series = "-".join(model_series)
            
            # 创建模型映射，将agent ID映射到实际模型名称
            model_map[agent["id"]] = model_name
            
            if model_series not in all_models:
                all_models[model_series] = {
                    "id": model_series,  # 使用模型系列作为ID
                    "name": agent["name"],  # 保留agent名称作为显示名称
                    "model_name": model_name,  # 存储完整模型名称
                    "owned_by": "anthropic" if "claude" in model_name.lower() else "openai",
                    "agents": [agent["id"]]  # 存储支持此模型的所有agent IDs
                }
            else:
                # 添加到现有模型的agents列表
                all_models[model_series]["agents"].append(agent["id"])
    
    AVAILABLE_MODELS = list(all_models.values())
    print(f"成功加载 {len(AVAILABLE_MODELS)} 个唯一模型系列")
    
    # 打印模型映射关系
    for model in AVAILABLE_MODELS:
        print(f"  - 模型系列: {model['id']}, 实际模型: {model['model_name']}, 支持的Agents: {len(model['agents'])}")


async def authenticate_client(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """基于Authorization头中的API密钥验证客户端"""
    if not VALID_CLIENT_KEYS:
        raise HTTPException(
            status_code=503,
            detail="服务不可用: 服务器未配置客户端API密钥。",
        )

    if not auth or not auth.credentials:
        raise HTTPException(
            status_code=401,
            detail="需要在Authorization头中提供API密钥。",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if auth.credentials not in VALID_CLIENT_KEYS:
        raise HTTPException(status_code=403, detail="无效的客户端API密钥。")


@app.on_event("startup")
async def startup():
    """应用启动时初始化配置"""
    print("启动Retool OpenAI API适配器服务器...")
    load_client_api_keys()
    await initialize_retool_environment()
    print("服务器初始化完成。")


def get_models_list_response() -> ModelList:
    """构建模型列表响应"""
    model_infos = [
        ModelInfo(
            id=model.get("id", "unknown"),
            name=f"{model.get('name', 'Unknown')} ({model.get('model_name', 'unknown')})",
            created=int(time.time()),
            owned_by=model.get("owned_by", "anthropic") 
        )
        for model in AVAILABLE_MODELS
    ]
    return ModelList(data=model_infos)


@app.get("/v1/models", response_model=ModelList)
async def list_v1_models(_: None = Depends(authenticate_client)):
    """列出可用模型 - 需认证"""
    return get_models_list_response()


@app.get("/models", response_model=ModelList)
async def list_models_no_auth():
    """列出可用模型 - 无需认证(客户端兼容性)"""
    return get_models_list_response()


@app.get("/debug")
async def toggle_debug(enable: bool = Query(None)):
    """切换调试模式"""
    global DEBUG_MODE
    if enable is not None:
        DEBUG_MODE = enable
    return {"debug_mode": DEBUG_MODE}


async def retool_stream_generator(full_message: str, model_id: str):
    """生成模拟的流式响应"""
    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())
    
    # 发送初始角色增量
    yield f"data: {StreamResponse(id=stream_id, created=created_time, model=model_id, choices=[StreamChoice(delta={'role': 'assistant'})]).json()}\n\n"
    
    # 将完整消息分成小块
    chunk_size = 5  # 每个块的字符数
    for i in range(0, len(full_message), chunk_size):
        chunk = full_message[i:i+chunk_size]
        delta = {"content": chunk}
        
        response = StreamResponse(
            id=stream_id,
            created=created_time,
            model=model_id,
            choices=[StreamChoice(delta=delta)]
        )
        
        yield f"data: {response.json()}\n\n"
        await asyncio.sleep(0.01)  # 添加小延迟使流更自然
    
    # 发送完成信号
    yield f"data: {StreamResponse(id=stream_id, created=created_time, model=model_id, choices=[StreamChoice(delta={}, finish_reason='stop')]).json()}\n\n"
    yield "data: [DONE]\n\n"


async def error_stream_generator(error_detail: str, status_code: int):
    """生成错误流响应"""
    yield f'data: {json.dumps({"error": {"message": error_detail, "type": "retool_api_error", "code": status_code}})}\n\n'
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, _: None = Depends(authenticate_client)
):
    """使用Retool后端创建聊天完成"""
    # 检查模型是否存在
    model_exists = any(model["id"] == request.model for model in AVAILABLE_MODELS)
    if not model_exists:
        raise HTTPException(status_code=404, detail=f"未找到模型 '{request.model}'。")

    if not request.messages:
        raise HTTPException(status_code=400, detail="请求中未提供消息。")
    
    log_debug(f"处理模型请求: {request.model}")
    
    # 格式化消息历史
    formatted_message = format_messages_for_retool(request.messages)
    log_debug(f"格式化后的消息: {formatted_message[:100]}...")
    
    # 尝试所有可用账户
    for attempt in range(len(RETOOL_ACCOUNTS)):
        account = get_best_retool_account(request.model)
        if not account:
            raise HTTPException(
                status_code=503, 
                detail=f"没有可用的账户支持模型 '{request.model}'。"
            )
        
        # 获取为此账户选择的agent ID
        agent_id = account.get("selected_agent_id")
        if not agent_id:
            continue

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                # 创建线程
                thread_id = await retool_get_thread_id(client, account, agent_id)
                log_debug(f"创建线程: {thread_id}")
                
                # 发送消息
                log_id = await retool_send_message(client, account, agent_id, thread_id, formatted_message)
                log_debug(f"发送消息, 日志ID: {log_id}")
                
                # 获取响应
                message = await retool_get_message(client, account, agent_id, log_id)
                if not message:
                    raise Exception("获取消息响应超时")
                
                log_debug(f"收到响应: {message[:100]}...")
                
                # 根据请求类型返回流式或非流式响应
                if request.stream:
                    return StreamingResponse(
                        retool_stream_generator(message, request.model),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Accel-Buffering": "no",
                        },
                    )
                else:
                    # 非流式响应
                    return ChatCompletionResponse(
                        model=request.model,
                        choices=[
                            ChatCompletionChoice(
                                message=ChatMessage(
                                    role="assistant",
                                    content=message
                                )
                            )
                        ],
                    )

        except Exception as e:
            log_debug(f"请求错误: {e}")
            
            with account_rotation_lock:
                account["error_count"] += 1
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in [401, 403]:
                    account["is_valid"] = False
                    print(f"账户 {account['domain_name']} 因认证错误被标记为无效。")

    # 所有尝试都失败
    if request.stream:
        return StreamingResponse(
            error_stream_generator("所有Retool API请求尝试均失败。", 503),
            media_type="text/event-stream",
            status_code=503,
        )
    else:
        raise HTTPException(status_code=503, detail="所有Retool API请求尝试均失败。")


if __name__ == "__main__":
    import uvicorn

    # 设置环境变量以启用调试模式
    if os.environ.get("DEBUG_MODE", "").lower() == "true":
        DEBUG_MODE = True
        print("通过环境变量启用调试模式")

    # 检查必要文件
    if not os.path.exists("retool.json"):
        print("警告: retool.json未找到。创建示例文件。")
        dummy_data = [
            {
                "domain_name": "your-domain.retool.com",
                "x_xsrf_token": "your-xsrf-token",
                "accessToken": "your-access-token"
            }
        ]
        with open("retool.json", "w", encoding="utf-8") as f:
            json.dump(dummy_data, f, indent=4)
        print("创建示例retool.json。请替换为有效的Retool数据。")

    if not os.path.exists("client_api_keys.json"):
        print("警告: client_api_keys.json未找到。创建示例文件。")
        dummy_key = f"sk-dummy-{uuid.uuid4().hex}"
        with open("client_api_keys.json", "w", encoding="utf-8") as f:
            json.dump([dummy_key], f, indent=2)
        print(f"创建示例client_api_keys.json，密钥: {dummy_key}")

    # 加载配置
    load_client_api_keys()
    asyncio.run(initialize_retool_environment())

    print("\n--- Retool OpenAI API适配器 ---")
    print(f"调试模式: {DEBUG_MODE}")
    print("端点:")
    print("  GET  /v1/models (需客户端API密钥验证)")
    print("  GET  /models (无需验证)")
    print("  POST /v1/chat/completions (需客户端API密钥验证)")
    print("  GET  /debug?enable=[true|false] (切换调试模式)")

    print(f"\n客户端API密钥: {len(VALID_CLIENT_KEYS)}")
    if RETOOL_ACCOUNTS:
        print(f"Retool账户: {len(RETOOL_ACCOUNTS)}")
        for account in RETOOL_ACCOUNTS:
            print(f"  - {account['domain_name']}: {len(account['agents'])} 个Agents")
    else:
        print("Retool账户: 未加载。检查retool.json。")
    
    if AVAILABLE_MODELS:
        models = sorted([m.get("id", "unknown") for m in AVAILABLE_MODELS])
        print(f"可用模型系列: {len(AVAILABLE_MODELS)}")
        print(f"模型系列列表: {', '.join(models)}")
    else:
        print("可用模型: 未加载。检查账户配置。")
    print("------------------------------------")

    uvicorn.run(app, host="0.0.0.0", port=8000)
