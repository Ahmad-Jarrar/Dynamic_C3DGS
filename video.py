# Create video from images

import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import argparse


def create_video(images, output_path, fps=30):
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for image in images:
        video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.destroyAllWindows()
    video.release()


# images in the folder should match the pattern: exp_{save_prefix}_{t}_{camera_id}.png
def create_videos_from_folder(folder_path, output_path, duration):
    
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".png"):
            img = Image.open(os.path.join(folder_path, filename))
            images.append(np.array(img))

    print(f"Creating video from {len(images)} images")

    if len(images) == 0:
        print(f"No images found in {folder_path}")
        return
    create_video(images, output_path, 30)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create video from images')
    parser.add_argument('--folder', type=str, help='folder containing images')
    parser.add_argument('--duration', type=int, default=50, help='duration in frames')
    parser.add_argument("--output", type=str, help="output video path")
    args = parser.parse_args()

    # Own models
    create_videos_from_folder(args.folder, args.output, args.duration)
        
        