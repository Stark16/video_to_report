import traceback
from fastapi import APIRouter, HTTPException

from app.api.schemas.schema_framequery import AnalyzeFrame
from app.api.services.inference_service import InferenceService
from app.api.services.metrics import MetricsService

router = APIRouter()

inference_service = InferenceService()
metrics_service = MetricsService()

@router.post("/analyze")
async def analyze_frames(request:AnalyzeFrame):

    try:
        results = inference_service.analyze_image(request)
    except Exception as e:
        traceback.print_exception(e)
        raise HTTPException(status_code=500, detail="Internal Server Error - Image Analysis Error")

    return results

@router.post("/health")
async def get_health_metrics():
    return metrics_service.get_health_metrics()