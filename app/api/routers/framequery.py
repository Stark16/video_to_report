import traceback
import asyncio
from fastapi import APIRouter, HTTPException

from app.api.schemas.schema_framequery import AnalyzeFrame
from app.api.services.inference_service import InferenceService
from app.api.services.metrics import MetricsService

router = APIRouter()

inference_service = InferenceService()
metrics_service = MetricsService()

inference_semaphore = asyncio.Semaphore(2)

@router.post("/analyze")
async def analyze_frames(request: AnalyzeFrame):
    try:
        async with inference_semaphore:
            results = await asyncio.to_thread(inference_service.analyze_image, request, metricsservice=metrics_service)
    except Exception as e:
        traceback.print_exception(e)
        raise HTTPException(status_code=500, detail=f"Internal Server Error - Image Analysis Error: {str(e)}")
    return results

@router.get("/health")
async def get_health_metrics():
    return metrics_service.get_health_metrics()