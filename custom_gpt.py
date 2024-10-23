from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from loguru import logger
import time
import requests
import functools
from typing import Callable, Any

def async_timer(func: Callable) -> Callable:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        print(f"Async function {func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper

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

# OpenAI 格式的请求模型
class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    name: Optional[str] = None

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
    choices: List[ChatCompletionChoice]
    usage: Usage

# ChatBot类
class ChatBot:
    def __init__(self, project_id: str, api_key: str, custom_persona: str):
        self.project_id = project_id
        self.api_key = api_key
        self.custom_persona = custom_persona
        self.conversation_history = []
        self.session_id = None

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

    def format_response(self, messages: List[Message], assistant_response: str) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time()*1000)}",
            created=int(time.time()),
            model="gpt-4-o",
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

    async def send_message(self, messages: List[Message]) -> ChatCompletionResponse:
        if not self.session_id:
            conversations = await self.get_all_conversations()
            logger.info(f"Received request: {conversations}")
            for conv in conversations:
                if conv['name'] == 'Default':
                    self.session_id = conv['session_id']
                    break
            if not self.session_id:
                self.session_id = await self.create_conversation()

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
            return self.format_response(messages, assistant_response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")

# 创建全局ChatBot实例
chatbot = ChatBot(PROJECT_ID, API_KEY, CUSTOM_PERSONA)

router = APIRouter()

@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
@async_timer
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming is not supported")

        response = await chatbot.send_message(request.messages)
        logger.info(f"Returned response: {response}")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

