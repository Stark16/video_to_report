import io
import base64
from PIL import Image

from app.api.schemas.schema_framequery import AnalyzeFrame
from app.core.config import settings
from app.domain.inference import VLMInference

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

    def analyze_image(self, request_payload:AnalyzeFrame):
        results = []
        print("\t\t [INFO] Processing Image")
        for b64_str in request_payload.images:
            img_decoded = self._decode_image(b64_str)
            result = self.OBJ_VLMInf.analyze(img_decoded, request_payload.prompt)
            print(result)
            results.append(result)
        return results
