import os
import cv2
import numpy as np
from retinaface import RetinaFace
import onnxruntime as ort
from arcface_onnx import ArcFaceONNX
from scrfd import SCRFD
import os.path as osp
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--input-dir', default='../test/ch24', help='path to the input directory containing the images to be recognized')
ap.add_argument('--output-dir', default='../results', help='path to the output directory where the result images will be saved')
ap.add_argument('--embed-dir', default='../dataset/cropped_authorized', help='path to the directory containing the authorized embeddings')
ap.add_argument('--det-model-name', default='det_10g.onnx', help='detector model name')
ap.add_argument('--rec-model-name', default='w600k_r50.onnx', help='recognizer model name')
ap.add_argument('--detector-thres', default=0.8, help='confidence threshold for face detection')

args = ap.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir
embed_dir = args.embed_dir
confidence_thresh = float(args.detector_thres)
det_model = args.det_model_name
rec_model = args.rec_model_name


ort.set_default_logger_severity(3)

assets_dir = osp.expanduser('~/.insightface/models/buffalo_l')
detector = SCRFD(os.path.join(assets_dir, det_model))
detector.prepare(0)
model_path = os.path.join(assets_dir, rec_model)
rec = ArcFaceONNX(model_path)
rec.prepare(0)

# Define distance function
def findCosineDistance(vector1, vector2):
    """
    Calculate cosine distance between two vector
    """
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    a = np.dot(vec1.T, vec2)
    b = np.dot(vec1.T, vec1)
    c = np.dot(vec2.T, vec2)
    return 1 - (a/(np.sqrt(b)*np.sqrt(c)))

# Set the minimum face size for detection (in pixels)
min_face_size = 20

# Load the authorized embeddings
embeddings = np.load(os.path.join(embed_dir, "embeddings.npy"))
names = np.load(os.path.join(embed_dir, "names.npy"))
print(len(embeddings), len(names))

# Loop over the images in the input directory
for filename in os.listdir(input_dir):
    # Load the image
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)
    print("running on ", img_path)

    # Detect the faces in the image
    faces = RetinaFace.detect_faces(img, threshold=confidence_thresh)
    # print(faces)

    if isinstance(faces, dict):
        # Loop over the detected faces and recognize the persons
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = faces[face]['facial_area']
            face_img = img[y1:y2, x1:x2]

            # Resize the face image to match the input size of the ONNX model
            face_img = cv2.resize(face_img, (112, 112))

            bbox, kps = detector.autodetect(face_img, max_num=1)
            name = "Unknown"
            if bbox.shape[0]==0:
                print("Face not found in Image")
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                continue
            feat1 = rec.get(face_img, kps[0])
            max_sim = 0
            for i in range(len(embeddings)):
                feat2 = embeddings[i]
                # print(feat1)
                # print(feat2)
                sim = rec.compute_sim(feat1, feat2)
                # print("score with ", names[i], " is ", sim)
                if sim<0.2:
                    conclu = 'They are NOT the same person'
                elif sim>=0.2 and sim<0.28 and sim>max_sim:
                    name = names[i]+"*"
                    max_sim = sim
                    conclu = 'They are LIKELY TO be the same person'
                elif sim>max_sim:
                    name = names[i]
                    max_sim = sim
                    conclu = 'They ARE the same person'

            # Draw a rectangle on the face in the image and write the name of the person above the rectangle
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the result image in the output directory
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, img)
