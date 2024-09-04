import os
import dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import uvicorn

# Load environment variables
dotenv.load_dotenv()

# Configure loguru
logger.add("request_logs.log", rotation="7 day",level="INFO", encoding="utf-8")

app = FastAPI()

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

# Define Pydantic models
class Message(BaseModel):
    role: str
    content: str

class RequestData(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False

# Initialize OpenAI client
token = os.environ.get("GITHUB_TOKEN")
if not token:
    logger.error("GITHUB_TOKEN environment variable not set")
    raise RuntimeError("GITHUB_TOKEN environment variable not set")

endpoint = "https://models.inference.ai.azure.com"
client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Hello World"}

@app.post("/v1/chat/completions")
async def say_hello(request_data: RequestData):
    logger.info(f"Received request: {request_data}")
    try:
        if request_data.stream:
            # Define an asynchronous generator for streaming response
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
            # Non-streaming response
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
