import os
import sys
sys.path.insert(0, '../')
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2 as cv
from typing import Tuple
import tensorflow as tf
physical_devices=tf.config.experimental.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0],'GPU')

from config import *



def overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (1, 0, 0),
    alpha: float = 0.5, 
    resize: Tuple[int, int] = None
) -> np.ndarray:
    """Combines image and its segmentation mask into a single image.
    
    Params:
        image: Training image.
        mask: Segmentation mask.
        color: Color for segmentation mask rendering.
        alpha: Segmentation mask's transparency.
        resize: If provided, both image and its mask are resized before blending them together.
    
    Returns:
        image_combined: The combined image.
        
    """
    color = np.asarray(color).reshape(1, 1, 3)  # Change color channel position
    colored_mask = np.expand_dims(mask, -1) * color
    image_overlay = image.copy()  # Make a copy of the image

    # print(image_overlay.shape, mask.shape, colored_mask.shape)
    
    # Apply the overlay to the image
    op1 = (1 - alpha) * image_overlay[mask > 0]
    op2 = alpha * colored_mask[mask > 0]
    image_overlay[mask > 0] = op1 + op2
    # image_overlay[mask > 0] = (1 - alpha) * image_overlay[mask > 0] + alpha * colored_mask[mask > 0]
    
    if resize is not None:
        image = cv.resize(image, resize[::-1])  # Reverse the order for resizing
        image_overlay = cv.resize(image_overlay, resize[::-1])
    
    image_combined = cv.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    
    return image_combined
    

if __name__ == "__main__":
    """ Video Path """
    # video_path = str(data_path / "videos/NASA_video_1.mp4")
    # video_path = str(data_path / "videos/NASA_video_2.mp4")
    # video_path = str(data_path / "videos/NASA_video_3.mp4")
    video_path = str(data_path / "videos/NASA_video_4.mp4")

    stack_frames = True


    """ Load the model """
    model = tf.keras.models.load_model(saved_models_path / "CloudNet_biome8_epochs100_batch16_RGB.h5", compile=False)

    """ Reading frames """
    vs = cv.VideoCapture(video_path)

    if not vs.isOpened():
        print("Error: Could not open video capture.")
    else:
        print("Opened video capture.")

    _, frame = vs.read()
    H, W, _ = frame.shape
    vs.release()

    fourcc = cv.VideoWriter_fourcc('M','J','P','G')
    # out = cv.VideoWriter('output.avi', fourcc, 10, (W, H), True)
    out = cv.VideoWriter(str(results_path / (video_path.split("/")[-1].split(".")[0] + "_output.avi")), fourcc, 10, (W, H*2), True)
    # out = cv.VideoWriter(str(results_path / (video_path.split("/")[-1].split(".")[0] + "_output.avi")), fourcc, 10, (W, H), True)



    cap = cv.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if ret == False:
            cap.release()
            out.release()
            break

        H, W, _ = frame.shape
        ori_frame = frame
        frame = cv.resize(frame, (1024, 1024))
        frame = np.expand_dims(frame, axis=0)
        frame = frame / 255.0

        mask = model.predict(frame)[0]
        mask = mask > 0.5
        mask = mask.astype(np.float32)
        mask = cv.resize(mask, (W, H))
        mask = np.expand_dims(mask, axis=-1)

        combine_frame = ori_frame * mask
        combine_frame = overlay(ori_frame, mask.squeeze(), color=(0, 255, 0), alpha=0.5)
        combine_frame = combine_frame.astype(np.uint8)

        if stack_frames:
            # Stack the original frame on top of the frame with overlay
            stacked_frame = np.vstack((ori_frame, combine_frame))

            cv.imwrite(f"video/{idx}.png", stacked_frame)
            idx += 1

            out.write(stacked_frame)
        else:

            cv.imwrite(f"video/{idx}.png", combine_frame)
            idx += 1

            out.write(combine_frame)
