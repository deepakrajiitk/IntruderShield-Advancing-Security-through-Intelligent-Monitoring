import os
import cv2
import numpy as np
from retinaface import RetinaFace
import onnxruntime as ort
from arcface_onnx import ArcFaceONNX
import os.path as osp
import argparse


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

def recognizer(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    embed_dir = args.embed_dir
    confidence_thresh = float(args.detector_thres)
    rec_model = args.rec_model_name

    ort.set_default_logger_severity(3)

    assets_dir = osp.expanduser('~/.insightface/models/buffalo_l')
    model_path = os.path.join(assets_dir, rec_model)
    rec = ArcFaceONNX(model_path)
    rec.prepare(0)

    # Set the minimum face size for detection (in pixels)
    min_face_size = 20

    # Load the authorized embeddings
    embeddings = np.load(os.path.join(embed_dir, "embeddings.npy"))
    names = np.load(os.path.join(embed_dir, "names.npy"))
    print(len(embeddings), len(names))

    # Create a new folder in the output directory with an increasing serial number
    run_number = 1
    while os.path.exists(os.path.join(output_dir, f"run{run_number}")):
        run_number += 1
    output_dir = os.path.join(output_dir, f"run{run_number}")
    os.makedirs(output_dir)

    # Create the logs file if it doesn't exist
    logs_file = os.path.join(output_dir, 'logs.txt')
    if not os.path.exists(logs_file):
        open(logs_file, 'w').close()



    # Loop over the images in the input directory
    for filename in os.listdir(input_dir):
        # Load the image
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        print("running on ", img_path)

        # Detect the faces in the image
        faces = RetinaFace.detect_faces(img, threshold=confidence_thresh)
        known = []
        unknown_count = 0
        if isinstance(faces, dict):
            # Loop over the detected faces and recognize the persons
            for i, face in enumerate(faces):
                x1, y1, x2, y2 = faces[face]['facial_area']
                face_img = img[y1:y2, x1:x2]

                # Resize the face image to match the input size of the ONNX model
                face_img = cv2.resize(face_img, (112, 112))
                landmarks = faces[face]['landmarks']
                landmarks = np.array([[landmarks["right_eye"][0], landmarks["right_eye"][1]], [landmarks["left_eye"][0], landmarks["left_eye"][1]], [landmarks["nose"][0], landmarks["nose"][1]],
                             [landmarks["mouth_right"][0], landmarks["mouth_right"][1]], [landmarks["mouth_left"][0], landmarks["mouth_left"][1]]])
                name = "Unknown"
                feat1 = rec.get(img, landmarks)
                max_sim = 0
                for i in range(len(embeddings)):
                    feat2 = embeddings[i]
                    sim = rec.compute_sim(feat1, feat2)
                    if sim<0.15:
                        conclu = 'They are NOT the same person'
                    # elif sim>=0.15 and sim<0.28 and sim>max_sim:
                    #     name = names[i]+"**"
                    #     max_sim = sim
                    #     conclu = 'They are LIKELY TO be the same person'
                    elif sim>max_sim:
                        name = names[i]
                        max_sim = sim
                        conclu = 'They ARE the same person'

                if name!="Unknown":
                    known.append(name)
                else:
                    unknown_count += 1

                # Draw a rectangle on the face in the image and write the name of the person above the rectangle
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        log1 = "known persons are: " + ', '.join(str(i) for i in known)
        log2 = "total unknows are: " + str(unknown_count)
        # Append the string to the logs file
        with open(logs_file, 'a') as f:
            f.write(log1 + '\n')
            f.write(log2 + '\n')


        # Save the result image in the output directory
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, img)


# ap = argparse.ArgumentParser()
# ap.add_argument('--input-dir', default='../dataset/test/ch24', help='path to the input directory containing the images to be recognized')
# ap.add_argument('--output-dir', default='../results', help='path to the output directory where the result images will be saved')
# ap.add_argument('--embed-dir', default='../dataset/cropped_authorized', help='path to the directory containing the authorized embeddings')
# ap.add_argument('--rec-model-name', default='w600k_r50.onnx', help='recognizer model name')
# ap.add_argument('--detector-thres', default=0.8, help='confidence threshold for face detection')

# args = ap.parse_args()

# recognizer(args)