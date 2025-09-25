import io
import base64
import time
import torch
from PIL import Image

from app.api.schemas.schema_framequery import AnalyzeFrame
from app.core.config import settings
from app.domain.inference import VLMInference
from app.api.services.metrics import MetricsService

metricsservice = MetricsService()

class InferenceService:

    def __init__(self):
        self.OBJ_VLMInf = VLMInference(device=settings.vlm_compute_device)

    def _decode_image(self, b64_str:str):
        """Method to convert images from base64 to cv2 bg4

        Args:
            b64_str (str): the base64 image string

        Returns:
            np: cv2-np image object
        """
        
        img_bytes = base64.b64decode(b64_str)
        img_stream = io.BytesIO(img_bytes)
        img = Image.open(img_stream)
        return img
    
    def check_free_vram_in_gb(self):
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            torch.cuda.empty_cache()
            total = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
            reserved = torch.cuda.memory_reserved(gpu_id) / 1e9
            allocated = torch.cuda.memory_allocated(gpu_id) / 1e9
            free = total - reserved - allocated
            return free
        else:
            return None
    
    def dynamic_batch(self, imgs, max_batch=8, min_batch=1):
        free_vram = self.check_free_vram_in_gb()
        if free_vram is None:
            batch_size = min(max_batch, len(imgs))
        else:
            est_vram_per_img = 0.15  # Adjust based on profiling
            possible_batch = int(free_vram / est_vram_per_img)
            batch_size = max(min_batch, min(possible_batch, max_batch))
        # Split into batches
        return [imgs[i:i+batch_size] for i in range(0, len(imgs), batch_size)]
    
    def analyze_image(self, request_payload: AnalyzeFrame):
        start_time = time.time()

        decoded_imgs = [self._decode_image(b64) for b64 in request_payload.images]

        batches = self.dynamic_batch(decoded_imgs)

        results = []
        for batch in batches:
            batch_results = self.OBJ_VLMInf.analyze(batch, request_payload.prompt)
            results.extend(batch_results)

        elapsed = time.time() - start_time
        metricsservice.update_latency(elapsed)

        return results