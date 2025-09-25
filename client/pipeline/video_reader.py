import subprocess
import numpy as np
import shlex
import json


def get_video_dimensions(video_path: str):
    """Use ffprobe to get the width and height of the video."""
    cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of json {shlex.quote(video_path)}"
    result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")
    info = json.loads(result.stdout)
    width = info["streams"][0]["width"]
    height = info["streams"][0]["height"]
    return width, height


def video_reader(video_path):
    width, height = get_video_dimensions(video_path)
    command = f"ffmpeg -i {shlex.quote(video_path)} -vf fps=1 -f rawvideo -pix_fmt rgb24 -"
    pipe = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10 ** 8)

    frame_size = width * height * 3  # 3 bytes per pixel for rgb24

    while True:
        raw_frame = pipe.stdout.read(frame_size)
        if len(raw_frame) != frame_size:
            break
        frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
        yield frame

    pipe.stdout.close()
    pipe.wait()
