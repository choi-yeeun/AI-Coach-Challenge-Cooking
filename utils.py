from typing import List, Tuple
import numpy as np
from PIL import Image

def load_frames_into_array(
    video_frame_paths: List[str],
    video_input_resolution: Tuple[int,int]
) -> np.ndarray:
    video = []
    for video_frame_path in video_frame_paths:
        im = Image.open(video_frame_path).resize(video_input_resolution, resample=0)
        video.append(np.array(im))
    return np.array(video)
