import os
import cv2
import glob
import argparse
import numpy as np

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract frames from videos in a dataset.")
    parser.add_argument("--captaincook4d_root", type=str, required=True, help="Path to the dataset root directory")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second to extract (default: 5)")
    return parser.parse_args()

def process_video(filepath: str, fps: int = 2) -> list:
    """Extract frames from a video at a specified FPS rate.

    :param filepath: Path to the video file.
    :param fps: Frames per second to extract. Default is 2.
    :return: List of extracted frames as numpy arrays in BGR format.
    """
    frames = []
    cap = cv2.VideoCapture(filepath)
    frame_interval = int(round(cap.get(cv2.CAP_PROP_FPS) / fps))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    return frames

def save_frames_to_folder(video_output_folder: str, frames: list) -> None:
    """Save frames as JPEG images to a specified folder.

    :param video_output_folder: Path to the output folder.
    :param frames: List of frames to save as numpy arrays.
    """
    for frame_idx, frame in enumerate(frames):
        frame_filename = os.path.join(video_output_folder, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(frame_filename, frame)


def extract_frames_from_videos(input_folder: str, output_folder: str, fps: int) -> None:
    """Extract frames from all videos in a folder and save them to an output folder.

    :param input_folder: Path to the folder containing input video files.
    :param output_folder: Path to the folder where extracted frames will be saved.
    :param fps: Frames per second to extract from each video.
    """
    os.makedirs(output_folder, exist_ok=True)
    video_files = glob.glob(os.path.join(input_folder, "*_360p.mp4"))

    print(f"Staring frame extraction of {len(video_files)} videos in: {input_folder}")
    for video_path in tqdm(video_files):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_folder = os.path.join(output_folder, video_name)
        os.makedirs(video_output_folder, exist_ok=True)

        frames = process_video(video_path, fps)
        save_frames_to_folder(video_output_folder, frames)
        print(f"Extracted {len(frames)} frames from {video_name}.mp4 at {fps} fps")

    print("Frame extraction complete.")


def run_extraction(args: argparse.Namespace) -> None:
    """Run the frame extraction pipeline using parsed arguments.

    :param args: Parsed command-line arguments containing captaincook4d_root and fps.
    """
    input_folder = f"{args.captaincook4d_root}/resolution_360p"
    output_folder = f"{args.captaincook4d_root}/resolution_360p_video_frames_{args.fps}fps"
    extract_frames_from_videos(input_folder, output_folder, args.fps)


if __name__ == "__main__":
    args = parse_args()
    run_extraction(args)
