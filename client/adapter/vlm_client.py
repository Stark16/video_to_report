import base64
import requests
import numpy as np
from typing import Generator, List, Dict, Tuple
import hashlib

class VLMClient:
    def __init__(self, frame_generator: Generator[np.ndarray, None, None], query: str):
        self.frame_generator = frame_generator
        self.query = query
        self.api_endpoint = "http://localhost:8000/analyze"
        self.frame_cache = {}

    def image_to_b64(self, image: np.ndarray) -> str:
        import cv2
        _, buffer = cv2.imencode('.jpg', image)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        return b64_str
    
    def _hash_frame(self, frame: np.ndarray) -> str:
        # Compute a quick hash for a frame
        frame_bytes = frame.tobytes()
        return hashlib.md5(frame_bytes).hexdigest()

    def create_payload(self, batchsize: int) -> Tuple[Dict, List[np.ndarray]]:
        images_b64 = []
        frames_batch = []
        query = self.query
        valid_frame_indices = []

        for i in range(batchsize):
            try:
                frame = next(self.frame_generator)
            except StopIteration:
                break

            frame_hash = self._hash_frame(frame)
            if frame_hash in self.frame_cache:
                # Frame already cached, skip sending to LLM
                # But keep track of it for combined results later
                valid_frame_indices.append((len(images_b64), frame_hash))
                continue

            frames_batch.append(frame)
            images_b64.append(self.image_to_b64(frame))
            valid_frame_indices.append((len(images_b64) - 1, frame_hash))

        payload = {
            "images": images_b64,
            "prompt": query,
        }
        return payload, frames_batch, valid_frame_indices

    def send_request(self, payload: Dict) -> Dict:
        response = requests.post(self.api_endpoint, json=payload)
        response.raise_for_status()
        return response.json()

    def process_stream(self, batch_size: int = 1) -> List[Dict]:
        results = []
        i=3
        while True:
            payload, frames_batch, valid_frame_indices = self.create_payload(batch_size)
            if not payload["images"]:
                break

            response_json = self.send_request(payload)
            # Each frame in response corresponds to a frame in current batch in order
            # Merge with cached results and update cache for new frames
            batch_results = response_json
            idx_in_batch = 0
            for idx, frame_hash in valid_frame_indices:
                if frame_hash in self.frame_cache:
                    # Cached result with frame
                    results.append({
                        "frame": frames_batch[idx] if idx < len(frames_batch) else None,
                        "metadata": self.frame_cache[frame_hash]
                    })
                else:
                    # New result from API response
                    meta = batch_results[idx_in_batch]
                    frame = frames_batch[idx_in_batch] if idx_in_batch < len(frames_batch) else None
                    results.append({
                        "frame": frame,
                        "metadata": meta
                    })
                    self.frame_cache[frame_hash] = meta
                    idx_in_batch += 1
            i-=1
            if i==0:
                break
        return results
