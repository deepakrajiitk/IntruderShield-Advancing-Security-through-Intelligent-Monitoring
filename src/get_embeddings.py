import os
import os.path as osp
import cv2
from retinaface import RetinaFace
import onnxruntime
from scrfd import SCRFD
import numpy as np
import argparse
from arcface_onnx import ArcFaceONNX

ap = argparse.ArgumentParser()
ap.add_argument('--input-dir', default='../dataset/authorized', help='path to the input directory containing the images of authorized people')
ap.add_argument('--output-dir', default='../dataset/cropped_authorized', help='path to the output directory where the cropped faces will be saved')
ap.add_argument('--det-model-name', default='det_10g.onnx', help='detector model name')
ap.add_argument('--rec-model-name', default='w600k_r50.onnx', help='recognizer model name')
ap.add_argument('--detector-thres', default='0.8', help='confidence threshold for face detection')

args = ap.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
confidence_thresh = float(args.detector_thres)
det_model = args.det_model_name
rec_model = args.rec_model_name


onnxruntime.set_default_logger_severity(3)

assets_dir = osp.expanduser('~/.insightface/models/buffalo_l')
detector = SCRFD(os.path.join(assets_dir, det_model))
detector.prepare(0)
model_path = os.path.join(assets_dir, rec_model)
rec = ArcFaceONNX(model_path)
rec.prepare(0)



# Set the minimum face size for detection (in pixels)
min_face_size = 20

embeddings = []
names = []

# Loop over the subdirectories in the input directory
for subdir in os.listdir(input_dir):
    # Create a new directory for the cropped faces of this person
    person_dir = os.path.join(output_dir, subdir)
    os.makedirs(person_dir, exist_ok=True)

    # Loop over the images of this person in the subdirectory
    for filename in os.listdir(os.path.join(input_dir, subdir)):
        # Load the image
        img_path = os.path.join(input_dir, subdir, filename)
        img = cv2.imread(img_path)
        print(f"Processing images for person {subdir}")

        # Detect the faces in the image
        faces = RetinaFace.detect_faces(img, confidence_thresh)

        # Loop over the detected faces and save the cropped faces in the output directory
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = faces[face]['facial_area']
            face_img = img[y1:y2, x1:x2]
            face_img = cv2.resize(face_img, (112, 112))
            bbox, kps = detector.autodetect(face_img, max_num=1)
            if bbox.shape[0]==0:
                print("Face not found in Image")
                continue
            embd = rec.get(face_img, kps[0])
            embeddings.append(embd)
            names.append(subdir)

            # Save the cropped face image in the person's directory
            output_path = os.path.join(person_dir, f"{filename.split('.')[0]}_{i}.jpg")
            cv2.imwrite(output_path, face_img)

np.save(os.path.join(output_dir, "embeddings.npy"), np.array(embeddings))
np.save(os.path.join(output_dir, "names.npy"), np.array(names))

