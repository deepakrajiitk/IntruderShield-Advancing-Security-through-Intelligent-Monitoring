import os
import cv2
import numpy as np
from retinaface import RetinaFace
import onnxruntime as ort
from arcface_onnx import ArcFaceONNX
import os.path as osp
import argparse
from collections import OrderedDict


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

# clear those names from dictionary whose probablity has become 0


def update_ordered_dict(prev_objects: OrderedDict, decrement: float) -> OrderedDict:
    for key in list(prev_objects.keys()):
        centroid, prob = prev_objects[key]
        prob = prob - decrement
        if prob > 0:
            prev_objects[key] = (centroid, prob)
        else:
            prev_objects.pop(key)
    return prev_objects


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

    # Initialize variables for face tracking
    prev_objects = OrderedDict()

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
    for filename in sorted(os.listdir(input_dir)):
        # Load the image
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        print("running on ", img_path)

        # Detect the faces in the image
        faces = RetinaFace.detect_faces(img, threshold=confidence_thresh)
        print(faces)
        known = []
        unknown_count = 0
        if isinstance(faces, dict):
            # Loop over the detected faces and recognize the persons
            for i, face in enumerate(faces):
                x1, y1, x2, y2 = faces[face]['facial_area']
                centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                landmarks = faces[face]['landmarks']
                landmarks = np.array([[landmarks["right_eye"][0], landmarks["right_eye"][1]], [landmarks["left_eye"][0], landmarks["left_eye"][1]], [landmarks["nose"][0], landmarks["nose"][1]],
                                      [landmarks["mouth_right"][0], landmarks["mouth_right"][1]], [landmarks["mouth_left"][0], landmarks["mouth_left"][1]]])
                name = "Unknown"

                feat1 = rec.get(img, landmarks)
                max_sim = 0
                for i in range(len(embeddings)):
                    feat2 = embeddings[i]
                    sim = rec.compute_sim(feat1, feat2)
                    if sim < 0.15:
                        conclu = 'They are NOT the same person'
                    # elif sim>=0.15 and sim<0.28 and sim>max_sim:
                    #     name = names[i]+"**"
                    #     max_sim = sim
                    #     conclu = 'They are LIKELY TO be the same person'
                    elif sim > max_sim:
                        name = names[i]
                        max_sim = sim
                        conclu = 'They ARE the same person'

                if name != "Unknown":
                    known.append(name)
                    # adding known person to the dictionary
                    prev_objects[name] = (centroid, 1)
                else:
                    unknown_count += 1
                    nearest_dist_thresh = 50
                    # Implement face tracking using centroid tracking
                    for key, value in prev_objects.items():
                        prev_centroid = value[0]
                        distance = np.sqrt(
                            (centroid[0] - prev_centroid[0]) ** 2 + (centroid[1] - prev_centroid[1]) ** 2)
                        if distance < nearest_dist_thresh:  # If the distance between the centroids is less than 50 pixels, consider it as the same object
                            name = key
                            nearest_dist_thresh = distance
                            break

                    if name != "Unknown":
                        known.append(name)
                        prev_objects[name] = (centroid, prev_objects[name][1])
                        name = name + "##"

                # Draw a rectangle on the face in the image and write the name of the person above the rectangle
                cv2.rectangle(img, (int(x1), int(y1)),
                              (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, name, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # cv2.putText(img, "Object ID: " + str(object_id), (int(x1), int(y2) + 15),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # # Display object ID and track object using its centroid
                # if prev_objects[object_id][1] != "Unknown":
                #     cv2.putText(img, "Name: " + prev_objects[object_id][1], (int(x1), int(y2) + 30),
                #                 cv2.FONT_HERSHEY, 0.5, (0, 255, 0), 2)

        prev_objects = update_ordered_dict(prev_objects, 0.2)

        log1 = "known persons are: " + ', '.join(str(i) for i in known)
        log2 = "total unknows are: " + str(unknown_count)
        # Append the string to the logs file
        with open(logs_file, 'a') as f:
            f.write(log1 + '\n')
            f.write(log2 + '\n')

        # Save the result image in the output directory
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, img)


ap = argparse.ArgumentParser()
ap.add_argument('--input-dir', default='../dataset/test/ch24',
                help='path to the input directory containing the images to be recognized')
ap.add_argument('--output-dir', default='../results',
                help='path to the output directory where the result images will be saved')
ap.add_argument('--embed-dir', default='../dataset/cropped_authorized',
                help='path to the directory containing the authorized embeddings')
ap.add_argument('--rec-model-name', default='w600k_r50.onnx',
                help='recognizer model name')
ap.add_argument('--detector-thres', default=0.8,
                help='confidence threshold for face detection')

args = ap.parse_args()

recognizer(args)
