import argparse
from typing import List, Dict
import multiprocessing
from functools import partial
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

def process_video(video_path: str, query: str, top_k: int = 3):
    """
    Processes a single video file: runs VLM analysis, selects top frames, and runs YOLO detection.
    """
    print(f"Processing video: {video_path} with query: '{query}'")
    try:
        frame_gen = video_reader(video_path)
        vml_client = VLMClient(frame_gen, query)
        # TODO: Make batch_size configurable
        vlm_results = vml_client.process_stream(batch_size=4)
        
        if not vlm_results:
            print(f"No results from VLM for video: {video_path}")
            return {video_path: []}

        top_k_frames = select_top_k_frames(vlm_results, k=top_k)
        
        if not top_k_frames:
            print(f"No top frames selected for video: {video_path}")
            return {video_path: []}

        enriched_results = yolo_inference_and_save(top_k_frames, os.path.basename(video_path))
        print(f"Finished processing video: {video_path}")
        return {video_path: enriched_results}
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return {video_path: {"error": str(e)}}

def main():
    parser = argparse.ArgumentParser(description="Video Query Pipeline")
    parser.add_argument("--videos", nargs='+', required=True, help="List of paths to input video files")
    parser.add_argument("--query", nargs='+', type=str, default="identify an accident or vehicle collision", help="Query string for video analysis")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent processes to run.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top frames to select for YOLO processing.")

    args = parser.parse_args()

    if not args.videos:
        print("No videos provided. Exiting.")
        return

    # Use partial to create a function with the query and top_k arguments pre-filled
    process_func = partial(process_video, query=args.query, top_k=args.top_k)

    all_results = []
    if args.workers > 1 and len(args.videos) > 1:
        print(f"Starting parallel processing with {args.workers} workers.")
        with multiprocessing.Pool(processes=args.workers) as pool:
            results = pool.map(process_func, args.videos)
            all_results.extend(results)
    else:
        print("Starting sequential processing.")
        for video_path in args.videos:
            result = process_func(video_path)
            all_results.append(result)

    print("\n--- All Videos Processed ---")
    for result in all_results:
        for video_path, data in result.items():
            if "error" in data:
                print(f"Video: {video_path}\n  Error: {data['error']}")
            else:
                print(f"Video: {video_path}\n  Detections saved in a new directory inside 'data/'.")
    
    # You can add code here to aggregate reports if needed.
    # For example, save all_results to a JSON file.
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"report_{timestamp}.json", "w") as f:
        import json
        json.dump(all_results, f, indent=4)
    print(f"\nAggregated report saved to report_{timestamp}.json")


if __name__ == "__main__":
    # This is required for multiprocessing on some platforms (like Windows)
    multiprocessing.freeze_support()
    main()
