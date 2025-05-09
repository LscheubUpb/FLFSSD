'Entries: 5/50 (Not Modified Often)'

import cv2
import os

input_folder = "C:/Users/leosc/Documents/_wichtigeDokumente/Bachelorarbeit/MagFace/eval/eval_recognition/data/Adience/adience_mtcnn160_png_19339"
output_folder = "C:/Users/leosc/Documents/_wichtigeDokumente/Bachelorarbeit/MagFace/eval/eval_recognition/data/Adience/adience_resized"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path)

    if img is not None:
        resized_img = cv2.resize(img, (112, 112))
        cv2.imwrite(os.path.join(output_folder, img_name), resized_img)
