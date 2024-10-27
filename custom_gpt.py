import hashlib
import json
import uuid
import aiohttp
import time
import requests
import functools
from fastapi import HTTPException, APIRouter, Header
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, AsyncGenerator, Callable, Any
from loguru import logger
from starlette.responses import StreamingResponse

CUSTOM_PERSONA = "You are a helpful assistant."

def async_timer(func: Callable) -> Callable:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Async function {func.__name__} took {end_time - start_time:.2f}s to execute")
        return result
    return wrapper


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


async def generate_system_fingerprint() -> str:
    unique_id = f"{time.time()}{uuid.uuid4()}"
    fingerprint = hashlib.sha256(unique_id.encode()).hexdigest()
    return fingerprint[:16]


async def format_response(messages: List[Message], assistant_response: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time() * 1000)}",
        created=int(time.time()),
        model="gpt-4-o",
        system_fingerprint=f"fp_{await generate_system_fingerprint()}",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(
                    role="user",
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


async def get_all_conversations(project_id, api_key):
    url = f"https://app.customgpt.ai/api/v1/projects/{project_id}/conversations"
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {api_key}"
    }
    try:
        response = requests.get(url, headers=headers)
        conversations = response.json()['data']['data']
        return conversations
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get conversations: {str(e)}")


async def create_conversation(project_id, api_key, conversation_name: str = "Default"):
    url = f"https://app.customgpt.ai/api/v1/projects/{project_id}/conversations"
    payload = {"name": conversation_name}
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}"
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        session_id = response.json()['data']['session_id']
        return session_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")


class ChatBot:
    def __init__(self, custom_persona: str):
        self.custom_persona = custom_persona
        self.conversation_history = []
        self.session_id = None

    async def none_stream_chat_completion(self, messages: List[Message], project_id, api_key) -> ChatCompletionResponse:
        context = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        url = f"https://app.customgpt.ai/api/v1/projects/{project_id}/conversations/{self.session_id}/messages"
        payload = {
            "response_source": "openai_content",
            "prompt": context,
            "custom_persona": self.custom_persona,
            "chatbot_model": "gpt-4-o"
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}"
        }

        try:
            response = requests.post(url, json=payload, headers=headers)
            response_data = response.json()
            assistant_response = response_data['data']['openai_response']
            return await format_response(messages, assistant_response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")

    async def stream_chat_completion(
            self,
            messages: List[Message],
            project_id,
            api_key
    ) -> AsyncGenerator[str, None]:
        prompt = messages[-1].content
        url = f"https://app.customgpt.ai/api/v1/projects/{project_id}/conversations/{self.session_id}/messages?stream=true&lang=zh"
        payload = {
            "response_source": "openai_content",
            "prompt": prompt,
            "custom_persona": self.custom_persona,
            "chatbot_model": "gpt-4-o"
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}"
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


chatbot = ChatBot(CUSTOM_PERSONA)
router = APIRouter()


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
@async_timer
async def chat(request: ChatCompletionRequest, authorization: str = Header(None)):
    try:
        # Verify authorization header
        if not authorization:
            raise HTTPException(
                status_code=401,
                detail="Authorization header is required"
            )

        # Extract the API key - assuming Bearer token format
        if not authorization.startswith('Bearer '):
            raise HTTPException(
                status_code=401,
                detail="Invalid authorization format. Must be 'Bearer <token>'"
            )

        project_id = authorization.replace('Bearer ', '').split("#")[0].strip()
        if not project_id or project_id == "":
            raise HTTPException(
                status_code=401,
                detail="Authorization can't find project_id header,however project_id is required"
            )

        api_key = authorization.split("#")[1].strip()
        if not project_id or not api_key:
            raise HTTPException(status_code=400, detail="Project ID or API Key is missing")

        conversations = await get_all_conversations(project_id, api_key)
        for conv in conversations:
            if conv['name'] == 'Default':
                chatbot.session_id = conv['session_id']
                break

        if not chatbot.session_id:
            chatbot.session_id = await create_conversation(project_id, api_key)

        # system_content = next((message["content"] for message in request.messages if message["role"] == "system"), None)
        system_content = next((message.content for message in request.messages if message.role == "system"), None)

        if system_content:
            chatbot.custom_persona = system_content

        if request.stream:
            return StreamingResponse(
                chatbot.stream_chat_completion(request.messages, project_id, api_key),
                media_type="text/event-stream"
            )
        else:
            response = await chatbot.none_stream_chat_completion(request.messages, project_id, api_key)
            return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
