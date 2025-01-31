import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import HttpUrl
from schemas.request import PredictionRequest, PredictionResponse
from utils.logger import setup_logger
from src.services.llm_service import process_request
from src.services.google_mistral_service import GoogleMistralService

# Initialize
app = FastAPI(title="ITMO University AI Agent")
logger = None
google_mistral_service = None


@app.on_event("startup")
async def startup_event():
    global logger, google_mistral_service
    logger = await setup_logger()
    google_mistral_service = GoogleMistralService()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    body = await request.body()
    await logger.info(
        f"Incoming request: {request.method} {request.url}\n"
        f"Request body: {body.decode()}"
    )

    response = await call_next(request)
    process_time = time.time() - start_time

    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    await logger.info(
        f"Request completed: {request.method} {request.url}\n"
        f"Status: {response.status_code}\n"
        f"Response body: {response_body.decode()}\n"
        f"Duration: {process_time:.3f}s"
    )

    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )

@app.post("/api/google-mistral", response_model=PredictionResponse)
async def predict(body: PredictionRequest):
    """
    Process a query about ITMO University using an AI agent.
    
    The agent will:
    1. Validate and process the request
    2. Search for information using Google
    3. Provide a comprehensive response with sources
    """
    try:
        await logger.info(f"Processing prediction request with id: {body.id}")
        
        result = await process_request(body.query, str(body.id))

        response = PredictionResponse(
            id=body.id,
            answer=result["metadata"]["answer"],
            reasoning=result["response"],
            sources=[HttpUrl(url) for url in result["metadata"]["sources"][:3]]
        )

        await logger.info(f"Successfully processed request {body.id}")
        return response

    except ValueError as e:
        error_msg = str(e)
        await logger.error(f"Validation error for request {body.id}: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        await logger.error(f"Internal error processing request {body.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/request", response_model=PredictionResponse)
async def predict_google_mistral(body: PredictionRequest):
    """
    Process a query about ITMO University using Google Search and MistralAI.
    
    The endpoint will:
    1. Validate the query and extract questions in both languages
    2. Search Google for relevant information
    3. Use MistralAI to generate a comprehensive response
    """
    try:
        await logger.info(f"Processing google-mistral request with id: {body.id}")
        
        result = await google_mistral_service.process_request(body.query, str(body.id))

        response = PredictionResponse(
            id=body.id,
            answer=result["answer"],
            reasoning=result["reasoning"],
            sources=[HttpUrl(url) for url in result["sources"][:3]]
        )

        await logger.info(f"Successfully processed google-mistral request {body.id}")
        return response

    except ValueError as e:
        error_msg = str(e)
        await logger.error(f"Validation error for google-mistral request {body.id}: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        await logger.error(f"Internal error processing google-mistral request {body.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
