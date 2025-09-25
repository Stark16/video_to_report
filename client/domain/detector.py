# client_pipeline/pipeline/detector.py

import torch
from ultralytics import YOLO
from PIL import Image
import os
from typing import List, Dict, Any, Tuple

class YOLODetector:
    """
    Performs object detection using a YOLOv8 model and crops detected objects.
    """
    def __init__(self, model_name: str = "yolov8n.pt", device: str = None):
        """
        Initializes the YOLO model.

        Args:
            model_name (str): The name of the YOLO model to load.
                              Defaults to "yolov8n.pt" (nano version).
            device (str): The device to run the model on (e.g., "cuda" or "cpu").
        """
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_name).to(self.device)
        self.model.eval()

    def infer(self, image: Image) -> List[Dict[str, Any]]:
        """
        Runs inference on a single PIL Image object.

        Args:
            image (Image): The input image (PIL Image object).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, one for each detected object,
                                  containing bounding box, class, and confidence.
        """
        # Run inference on the image
        results = self.model.predict(source=image, verbose=False, device=self.device)

        detections = []
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    label = self.model.names[class_id]

                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "label": label,
                    })

        return detections

    def crop_object(self, image: Image, box: List[float]) -> Image.Image:
        """
        Crops a single object from an image based on its bounding box.

        Args:
            image (Image): The original PIL Image object.
            box (List[float]): The bounding box coordinates [x1, y1, x2, y2].

        Returns:
            Image.Image: The cropped image (PIL Image object).
        """
        x1, y1, x2, y2 = box
        cropped_image = image.crop((x1, y1, x2, y2))
        return cropped_image

# Example Usage:
if __name__ == "__main__":
    from PIL import Image

    try:
        sample_image_path = r"D:\Career\tasks\secura_ai\video_to_report\data\test_images\screenshot.png"
        if not os.path.exists(sample_image_path):
            print(f"Error: The image file at {sample_image_path} was not found.")
            print("Please provide a valid image path to run this script.")
        else:
            pil_image = Image.open(sample_image_path).convert("RGB")

            yolo_detector = YOLODetector()

            print("Running YOLO inference...")
            detected_objects = yolo_detector.infer(pil_image)

            if detected_objects:
                print(f"Found {len(detected_objects)} objects.")
                for i, detection in enumerate(detected_objects):
                    label = detection["label"]
                    confidence = detection["confidence"]
                    box = detection["box"]
                    print(f"  - Object {i+1}: {label} with confidence {confidence:.2f}")

                    if i == 0:
                        cropped_img = yolo_detector.crop_object(pil_image, box)
                        cropped_img.save(f"cropped_{label}_{confidence:.2f}.jpg")
                        print(f"    - Cropped image saved as 'cropped_{label}_{confidence:.2f}.jpg'")
            else:
                print("No objects detected in the image.")
                
    except Exception as e:
        print(f"An error occurred: {e}")