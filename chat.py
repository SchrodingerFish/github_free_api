from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from loguru import logger
from openai import OpenAI
from starlette.responses import StreamingResponse

# 定义 Pydantic 模型
class Message(BaseModel):
    role: str
    content: str

class RequestData(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False

router = APIRouter()

@router.post("/v1/chat/completions")
async def say_hello(request_data: RequestData, authorization: str = Header(...)):
    logger.info(f"Received request: {request_data}")

    # Extract the token from the Authorization header
    token = authorization.split(" ")[1] if " " in authorization else authorization

    # Initialize OpenAI client with the token from the header
    endpoint = "https://models.inference.ai.azure.com"
    client = OpenAI(
        base_url=endpoint,
        api_key=token,
    )

    try:
        if request_data.stream:
            # 处理流式响应
            async def event_stream():
                try:
                    stream_response = client.chat.completions.create(
                        model=request_data.model,
                        messages=[message.dict() for message in request_data.messages],
                        stream=True
                    )
                    for update in stream_response:
                        content = update.choices[0].delta.content if update.choices[0].delta.content else ""
                        if content:
                            logger.debug(f"Streaming update: {content}")
                        yield f"data: {update.json()}\n\n"
                except Exception as stream_err:
                    logger.error(f"Streaming error: {stream_err}")
                    yield f"data: {{'error': '{str(stream_err)}'}}\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")
        else:
            # 非流式响应
            response = client.chat.completions.create(
                model=request_data.model,
                messages=[message.dict() for message in request_data.messages],
                stream=False
            )
            logger.info(f"Response: {response}")
            return response
    except Exception as err:
        logger.error(f"Error occurred: {err}")
        raise HTTPException(status_code=500, detail=str(err))
