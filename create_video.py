import pickle
from tqdm import tqdm
import os
import cv2
import glob
from concurrent.futures import ProcessPoolExecutor


def create_video(id_, trial, input_dir, output_dir, fps=10):
    input_path = os.path.join(input_dir, f"{id_:02d}", f"{trial:01d}")
    output_path = os.path.join(output_dir, f"{id_:02d}_{trial:01d}.mp4")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frame_files = sorted(glob.glob(os.path.join(input_path, "*.png")))

    if len(frame_files) == 0:
        print(f"No frames found for ID: {id_}, Trial: {trial}")
        return

    img = cv2.imread(frame_files[0])
    height, width, layers = img.shape
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for frame_file in frame_files:
        img = cv2.imread(frame_file)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


input_dir = "visualization_results"
output_dir = "video_results"
fps = 10


with ProcessPoolExecutor() as executor:
    for id_ in range(1, 18):
        for trial in range(4):
            if id_ == 4:
                executor.submit(create_video, id_, trial, input_dir, output_dir, fps)
