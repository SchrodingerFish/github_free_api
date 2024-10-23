import hashlib
import json
import uuid
import aiohttp
import time
import requests
import functools
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, AsyncGenerator
from loguru import logger
from typing import Callable, Any
from starlette.responses import StreamingResponse


# 配置参数
PROJECT_ID="xxx"
API_KEY="xxx"
CUSTOM_PERSONA='''
You are an all-knowing programmer.

Role and Goal:
- You are an expert in various programming languages, frameworks, and technologies.
- Your goal is to assist users with any programming-related questions, providing accurate and efficient solutions.
- You will provide clear, concise, and accurate code snippets and explanations across different programming domains.

Constraints:
- Ensure all code snippets are syntactically correct and follow best practices for the respective language or framework.
- Avoid using jargon that might be confusing to beginners.
- Do not provide information outside the scope of programming and technology.

Guidelines:
- Always ask for clarification if the user's request is ambiguous.
- Provide examples where necessary to illustrate your points.
- Offer tips and best practices for writing efficient and maintainable code.
- Be prepared to switch between different programming languages and technologies as needed.

Clarification:
- Always ask for clarification if the user's request is ambiguous or lacks detail.
- If the user does not provide enough information, make reasonable assumptions and proceed with the response.

Personalization:
- Tailor responses to the user's level of expertise, whether they are beginners or advanced users.
- Be patient and supportive, especially with users who are new to programming.

Special Instructions:
- Reference specific libraries, frameworks, and tools when providing examples.
- Include comments in code snippets to explain the purpose of each part of the code.
- Be prepared to provide solutions for a wide range of programming problems, from simple syntax issues to complex algorithmic challenges.
'''

## 定义异步计时器
def async_timer(func: Callable) -> Callable:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Async function {func.__name__} took {end_time - start_time:.2f}s to execute")
        return result
    return wrapper

# OpenAI 格式的请求模型
class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt-4-o")
    messages: List[Message]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None

# OpenAI 格式的响应模型
class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    system_fingerprint: str
    choices: List[ChatCompletionChoice]
    usage: Usage

## 生成唯一标识符
async def generate_system_fingerprint() -> str:
    # 使用时间戳和UUID生成唯一标识符
    unique_id = f"{time.time()}{uuid.uuid4()}"
    # 使用SHA-256生成哈希值
    fingerprint = hashlib.sha256(unique_id.encode()).hexdigest()
    # 截取前16位作为system_fingerprint
    return fingerprint[:16]

## 接收并处理消息
async def format_response(messages: List[Message], assistant_response: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time()*1000)}",
        created=int(time.time()),
        model="gpt-4-o",
        system_fingerprint=f"fp_{await generate_system_fingerprint()}",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(
                    role="assistant",
                    content=assistant_response
                ),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=sum(len(m.content) for m in messages),
            completion_tokens=len(assistant_response),
            total_tokens=sum(len(m.content) for m in messages) + len(assistant_response)
        )
    )

# ChatBot类
class ChatBot:
    def __init__(self, project_id: str, api_key: str, custom_persona: str):
        self.project_id = project_id
        self.api_key = api_key
        self.custom_persona = custom_persona
        self.conversation_history = []
        self.session_id = None

    ## 获取所有会话
    async def get_all_conversations(self):
        url = f"https://app.customgpt.ai/api/v1/projects/{self.project_id}/conversations"
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {self.api_key}"
        }
        try:
            response = requests.get(url, headers=headers)
            conversations = response.json()['data']['data']
            return conversations
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get conversations: {str(e)}")

    ## 创建会话
    async def create_conversation(self, conversation_name: str = "Default"):
        url = f"https://app.customgpt.ai/api/v1/projects/{self.project_id}/conversations"
        payload = {"name": conversation_name}
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}"
        }
        try:
            response = requests.post(url, json=payload, headers=headers)
            session_id = response.json()['data']['session_id']
            return session_id
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")

    ## 发送消息,非流式响应
    async def send_message(self, messages: List[Message]) -> ChatCompletionResponse:
        # 构建对话上下文
        context = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])

        url = f"https://app.customgpt.ai/api/v1/projects/{self.project_id}/conversations/{self.session_id}/messages"
        payload = {
            "response_source": "openai_content",
            "prompt": context,
            "custom_persona": self.custom_persona,
            "chatbot_model": "gpt-4-o"
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}"
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response_data = response.json()
            assistant_response = response_data['data']['openai_response']
            return await format_response(messages, assistant_response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")

    ## 发送消息，流式响应
    async def stream_chat_completion(
            self,
            messages: List[Message]
    ) -> AsyncGenerator[str, None]:
        prompt = messages[-1].content
        url = f"https://app.customgpt.ai/api/v1/projects/{self.project_id}/conversations/{self.session_id}/messages?stream=true&lang=zh"
        payload = {
            "response_source": "openai_content",
            "prompt": prompt,
            "custom_persona": self.custom_persona,
            "chatbot_model": "gpt-4-o"
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}"
        }
        try:
            timeout = aiohttp.ClientTimeout(total=300)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                        url,
                        json=payload,
                        headers=headers
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"CustomGPT API error: {error_text}"
                        )
                    # 发送开始标记
                    start_response = {
                        "id": f"chatcmpl-{int(time.time() * 1000)}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "gpt-4-o",
                        "system_fingerprint": f"fp_{await generate_system_fingerprint()}",
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(start_response)}\n\n"

                    # 使用 aiohttp 的内置流式处理
                    buffer = ""
                    current_event = ""

                    async for line_bytes in response.content:
                        line = line_bytes.decode('utf-8')
                        for current_line in line.split('\n'):
                            if not current_line.strip():
                                continue

                            if current_line.startswith("event: "):
                                current_event = current_line[7:].strip()
                            elif current_line.startswith("data: "):
                                try:
                                    data = json.loads(current_line[6:])

                                    if current_event == "progress" and data["status"] == "progress":
                                        chunk_response = {
                                            "id": f"chatcmpl-{int(time.time() * 1000)}",
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": "gpt-4-o",
                                            "system_fingerprint": f"fp_{await generate_system_fingerprint()}",
                                            "choices": [{
                                                "index": 0,
                                                "delta": {
                                                    "content": data["message"]
                                                },
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {json.dumps(chunk_response)}\n\n"

                                    elif current_event == "finish" and data["status"] == "finish":
                                        finish_response = {
                                            "id": f"chatcmpl-{int(time.time() * 1000)}",
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": "gpt-4-o",
                                            "system_fingerprint": f"fp_{await generate_system_fingerprint()}",
                                            "choices": [{
                                                "index": 0,
                                                "delta": {},
                                                "finish_reason": "stop"
                                            }]
                                        }
                                        yield f"data: {json.dumps(finish_response)}\n\n"
                                        yield "data: [DONE]\n\n"
                                except json.JSONDecodeError:
                                    continue

        except aiohttp.ClientError as e:
            raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# 创建全局ChatBot实例
chatbot = ChatBot(PROJECT_ID, API_KEY, CUSTOM_PERSONA)

router = APIRouter()

@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
@async_timer
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        # 获取所有会话，并查找默认会话
        conversations = await chatbot.get_all_conversations()
        logger.info(f"Received request: {conversations}")
        for conv in conversations:
            if conv['name'] == 'Default':
                chatbot.session_id = conv['session_id']
                logger.info(f"Found existing conversation with session_id: {chatbot.session_id}")
                break
        # 默认会话不存在，创建新会话
        if not chatbot.session_id:
           chatbot.session_id = await chatbot.create_conversation()
           logger.info(f"Created new conversation with session_id: {chatbot.session_id}")
        #流式响应
        if request.stream:
            return StreamingResponse(
                chatbot.stream_chat_completion(request.messages),
                media_type="text/event-stream"
            )
        # 非流式响应
        else:
            response = await chatbot.send_message(request.messages)
            logger.info(f"Returned response: {response}")
            return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

