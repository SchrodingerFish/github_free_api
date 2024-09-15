from fastapi import APIRouter, HTTPException
from loguru import logger
import requests
import json

from pydantic import BaseModel


class SearchRequest(BaseModel):
    type: str
    query: str
    tbs: str
    num: int
    page: int
    api_key: str
    retry: bool = True
router = APIRouter()


@router.post("/v1/serper/search")
async def search(request: SearchRequest):
    logger.info(f"Received request: {request.query}")
    url = f"https://google.serper.dev/{request.type}"

    payload = json.dumps({
        "q": request.query,
        "gl": "cn",
        "hl": "zh-cn",
        "tbs": request.tbs,
        "num": request.num,
        "page": request.page,
    })
    headers = {
        'X-API-KEY': request.api_key,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        if request.retry:
            logger.info("Retrying request...")
            return await search(request.query, request.api_key, retry=False)
        else:
            raise HTTPException(status_code=500, detail=f"Request failed after retrying: {str(e)}")
