# server/app/domain/inference.py

from typing import List, Dict, Any, Union
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

    def analyze(self, images: Union[List, Any], query: str) -> List[Dict[str, Any]]:
        """Run image analysis for a batch of images (or single image) and return a list of result dicts.

        Args:
            images: A single image (PIL.Image or compatible) or a list of images to analyze.
            query: The textual query to run against each image.

        Returns:
            List[Dict[str, Any]]: a list of result dictionaries, one per input image.
        """
        if not isinstance(images, list):
            images = [images]

        messages = [
            {"role": "user", "content": [
                {"type": "image", "url": ""},
                {"type": "text", "text": query}
            ]}
        ]

        # Create a prompt per image using the processor's chat template
        prompts = [self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                   for _ in images]

        # Tokenize/process the batch and move to device
        inputs = self.processor(text=prompts, images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate responses in batch
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.95
            )

        # Decode each output into text and build a simple result structure
        results: List[Dict[str, Any]] = []
        for i in range(output_ids.shape[0]):
            text_output = self.processor.decode(output_ids[i], skip_special_tokens=True)
            results.append({
                "caption": text_output,
                "entities": [],
                "reasoning": text_output,
                "relevance_score": 0.9
            })

        return results


if __name__ == "__main__":
    import json

    test_img_path = r"D:\Career\tasks\secura_ai\video_to_report\data\test_images\4K-security-camera-snapshot.jpg"
    query = "Identify the man standing in front of the camera"
    OBJ_VLMInfer = VLMInference(device='cuda')

    with open(test_img_path, 'r') as f:
        img = Image.open(f)
    
    results = OBJ_VLMInfer.analyze(img, query)
    print(json.dumps(results, indent=2))
    