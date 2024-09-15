from fastapi import FastAPI
from loguru import logger
from chat import router as chat_router
from search import router as search_router

app = FastAPI()

# 配置日志
logger.add("request_logs.log", rotation="7 day", level="INFO", encoding="utf-8")

# 注册路由
app.include_router(chat_router)
app.include_router(search_router)

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")
