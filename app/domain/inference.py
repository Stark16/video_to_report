# server/app/domain/inference.py

from typing import List, Dict, Any
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import cv2
from PIL import Image
import base64
import io
from huggingface_hub import login
from app.core.config import settings

class VLMInference:
    def __init__(self, model_name:str = "google/gemma-3-4b-it", device:str = None):
        login(settings.hugging_face_token)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32)
        self.model.to(self.device)

    def analyze(self, image: str, query: str) -> Dict[str, Any]:
        """Method to run image analysis for a given b64 image and query, returns a single result dictionary.

        Args:
            image_b64 (str): Input image b64 string
            query (str): query

        Returns:
            Dict[str, Any]: a result dictionary with parsed output
        """
        messages = [
            {"role": "user", "content": [
                {"type": "image", "url": ""},
                {"type": "text", "text": query}
            ]}
        ]
        
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        
        # Generate the response
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.95
            )
        text_output = self.processor.decode(output_ids[0], skip_special_tokens=True)
        
        # inputs = self.processor(messages=messages, images=[image], return_tensors="pt").to(self.device)

        # with torch.no_grad():
        #     outputs = self.model.generate(**inputs, max_new_tokens=128)
        #     text_output = self.processor.decode(outputs[0], skip_special_tokens=True)

        # TODO: Real parsing. For now, just a dummy structure.
        result = {
            "caption": text_output,
            "entities": [],
            "reasoning": text_output,
            "relevance_score": 0.9
        }

        return result


if __name__ == "__main__":
    import json

    test_img_path = r"D:\Career\tasks\secura_ai\video_to_report\data\test_images\4K-security-camera-snapshot.jpg"
    query = "Identify the man standing in front of the camera"
    OBJ_VLMInfer = VLMInference(device='cuda')

    with open(test_img_path, 'r') as f:
        img = Image.open(f)
    
    results = OBJ_VLMInfer.analyze(img, query)
    print(json.dumps(results, indent=2))
    