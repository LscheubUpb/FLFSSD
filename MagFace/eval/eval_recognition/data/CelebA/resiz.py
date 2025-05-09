import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

# Initialize face detector
app = FaceAnalysis()
app.prepare(ctx_id=0)  # Use -1 if you want to use CPU

# Define the folder containing images
folder_path = '.'

# Loop through all images in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(folder_path, filename)
        
        # Read the image
        image = cv2.imread(image_path)
        
        # Detect faces
        faces = app.get(image)
        
        if len(faces) == 0:
            print(f"No faces detected in {filename}, skipping...")
            continue

        # Get the first face's landmarks
        face = faces[0]
        landmarks = face.kps  # Get landmarks for alignment (5, 2) numpy array

        # Align and resize the image to 112x112
        aligned_image = norm_crop(image, landmarks, image_size=112)

        # Overwrite the original image with the aligned version
        cv2.imwrite(image_path, aligned_image)
        print(f"Processed and saved: {filename}")
