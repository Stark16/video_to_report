import argparse
from typing import List, Dict

import datetime
import uuid
import os
from typing import List, Dict, Any
from PIL import Image
import cv2
from client.pipeline.video_reader import video_reader
from client.adapter.vlm_client import VLMClient
from client.domain.detector import YOLODetector

def yolo_inference_and_save(top_k_frames: List[Dict[str, Any]], video_name: str) -> List[Dict]:
    """
    Run YOLO detection on top K frames, save cropped objects uniquely, and append detection metadata.
    
    Args:
        top_k_frames: List of dicts with keys 'frame' (np.ndarray) and 'metadata' (dict with VLM info)
        video_name: Identifier or basename of video for output path and file naming
    
    Returns:
        List of dicts, each containing enriched metadata including:
            - original VLM metadata
            - YOLO detections: list of dicts with keys bbox, label, confidence, cropped image relative path
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    PATH_self_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(PATH_self_dir, 'data', f"{video_name}_{timestamp}_{unique_id}")
    os.makedirs(output_dir, exist_ok=True)

    yolo = YOLODetector()  # load model with defaults

    enriched_results = []

    for frame_info in top_k_frames:
        frame = frame_info["frame"]
        vlm_meta = frame_info["metadata"]
        pil_image = Image.fromarray(frame)

        detections = yolo.infer(pil_image)
        detection_results = []

        for detection in detections:
            bbox = detection["bbox"]  # expected [x1, y1, x2, y2]
            label = detection["label"]
            conf = detection["confidence"]

            cropped_img = yolo.crop_object(pil_image, bbox)
            # Filename format: video_ts_label_confidence.jpg
            ts = vlm_meta.get("timestamp", "unknown_ts")
            conf_str = f"{conf:.2f}"
            cropped_filename = f"{video_name}_{ts}_{label}_{conf_str}.jpg"
            cropped_path = os.path.join(output_dir, cropped_filename)

            cropped_img.save(cropped_path)

            # Use relative path from script directory
            rel_cropped_path = os.path.relpath(cropped_path, start=PATH_self_dir)

            detection_results.append({
                "bbox": bbox,
                "label": label,
                "confidence": conf,
                "cropped_image_path": rel_cropped_path
            })

        enriched_results.append({
            "vlm_metadata": vlm_meta,
            "yolo_detections": detection_results
        })

    return enriched_results

def show_frames(frames_gen):
    for frame in frames_gen:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def select_top_k_frames(results: List[Dict], k: int) -> List[Dict]:
    # Sort results by metadata relevance_score descending and take top k
    sorted_results = sorted(results, key=lambda x: x["metadata"].get("relevance_score", 0), reverse=True)
    return sorted_results[:k]


def main():
    parser = argparse.ArgumentParser(description="Video Query Pipeline")
    parser.add_argument("--video", type=str, default=r"D:\Career\tasks\secura_ai\video_to_report\data\accident.mp4", help="Path to input video file")
    parser.add_argument("--query", type=str, default="identify an accident or vehicle collision", help="Query string for video analysis")

    args = parser.parse_args()

    frame_gen = video_reader(args.video)
    # show_frames(frame_gen)
    vmlClient = VLMClient(frame_gen, args.query)
    vlm_results = vmlClient.process_stream(batch_size=4)
    top_k_frames = select_top_k_frames(vlm_results, k=3)
    enriched_results = yolo_inference_and_save(top_k_frames, os.path.basename(args.video))


if __name__ == "__main__":
    main()
