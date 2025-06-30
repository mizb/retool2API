# Retool OpenAI API 适配器
> 将 Retool AI Agents 转换为 OpenAI 兼容的 API 接口

## 🚀 快速开始

### 1. 准备配置文件

创建 `retool.json` 配置你的 Retool 账户：

```json
[
  {
    "domain_name": "your-company.retool.com",
    "x_xsrf_token": "your-xsrf-token",
    "accessToken": "your-access-token"
  }
]
```

创建 `client_api_keys.json` 设置客户端 API 密钥：

```json
[
  "sk-your-custom-api-key-here"
]
```

### 2. 使用 Docker Compose 部署

```bash
# 克隆项目
git clone <your-repo-url>
cd retool2API

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

### 3. 使用 Docker 直接运行

```bash
# 构建镜像
docker build -t retool2api .

# 运行容器
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/retool.json:/app/retool.json:ro \
  -v $(pwd)/client_api_keys.json:/app/client_api_keys.json:ro \
  -e DEBUG_MODE=false \
  --name retool2api \
  retool2api
```

## 🔧 配置说明

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `DEBUG_MODE` | `false` | 启用调试日志输出 |

### 配置文件

#### retool.json
- `domain_name`: Retool 实例域名 (如: company.retool.com)
- `x_xsrf_token`: XSRF 令牌 (从浏览器开发者工具获取)
- `accessToken`: 访问令牌 (从浏览器 Cookie 中获取)

#### client_api_keys.json
客户端 API 密钥列表，用于认证 API 请求。

## 📋 API 端点

| 方法 | 路径 | 认证 | 说明 |
|------|------|------|------|
| `GET` | `/models` | ❌ | 获取可用模型列表 |
| `GET` | `/v1/models` | ✅ | 获取可用模型列表 |
| `POST` | `/v1/chat/completions` | ✅ | 聊天对话接口 |
| `GET` | `/debug?enable=true/false` | ❌ | 切换调试模式 |

## 💡 使用示例

### curl 请求示例

```bash
# 获取模型列表
curl http://localhost:8000/models

# 发送聊天请求
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-custom-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": true
  }'
```

### Python 客户端示例

```python
import openai

client = openai.OpenAI(
    api_key="sk-your-custom-api-key-here",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="claude-sonnet-4",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

## 🔍 故障排查

### 常见问题

1. **容器启动失败**
   ```bash
   # 检查日志
   docker-compose logs retool2api
   
   # 检查配置文件
   cat retool.json
   cat client_api_keys.json
   ```

2. **API 认证失败**
   - 确认 `client_api_keys.json` 中的密钥正确
   - 检查请求头中的 `Authorization: Bearer <your-key>`

3. **Retool 连接失败**
   - 验证 `retool.json` 中的凭据
   - 确认网络连接正常

### 启用调试模式

```bash
# 方法1: 环境变量
DEBUG_MODE=true docker-compose up

# 方法2: API 端点
curl "http://localhost:8000/debug?enable=true"
```

## 📦 项目结构

```
retool2API/
├── main.py              # 主应用文件
├── requirements.txt     # Python 依赖
├── Dockerfile          # Docker 镜像构建
├── docker-compose.yml  # 容器编排配置
├── .dockerignore       # Docker 忽略文件
├── retool.json         # Retool 账户配置
├── client_api_keys.json # 客户端 API 密钥
└── README.md           # 项目文档
```

## ✨ 用star助力本项目


