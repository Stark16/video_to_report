# server/app/domain/inference.py

from typing import List, Dict, Any, Union
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import json
from PIL import Image
import re
from huggingface_hub import login
from app.core.config import settings

class VLMInference:
    def __init__(self, model_name:str = "google/gemma-3-4b-it", device:str = None):
        login(settings.hugging_face_token)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16 if "cuda" in self.device else torch.float32)
        self.model.to(self.device)

    def parse_output(self, raw_text: str) -> Dict[str, Any]:
        """
        Extract and parse the JSON object from a raw VLM output string.

        Args:
            raw_text (str): The raw decoded output from the model.

        Returns:
            Dict[str, Any]: Parsed JSON object with defaults if parsing fails.
        """
        
        match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Fallback: try to locate first { ... } block
            match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                json_str = raw_text  # last fallback

        # Step 2: Attempt to parse
        parsed = json.loads(json_str)

        # Step 3: Normalize fields with defaults
        return {
            "caption": parsed.get("caption", ""),
            "entities": parsed.get("entities", []),
            "reasoning": parsed.get("reasoning", ""),
            "relevance_score": float(parsed.get("relevance_score", 0.0)),
            }

    def analyze(self, images: Union[List, Any], query: str, print_results:bool=False) -> List[Dict[str, Any]]:
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
            {"role": "system", "content": 
            "You are an image analysis system. Answer only in strict JSON with keys: "
            "caption (string), reasoning (string), relevance_score (float between 0 and 1), "
            "and entities (list of strings, may be empty). "
            "Do not include extra commentary, disclaimers, or text outside JSON."
            },
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
            parsed_output = self.parse_output(text_output)
            if print_results:
                print("\n\n", parsed_output, "\n\n")
            results.append(parsed_output)

        return results
    