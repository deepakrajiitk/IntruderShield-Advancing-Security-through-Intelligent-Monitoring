import os
import cv2
import numpy as np
from retinaface import RetinaFace
import onnxruntime as ort
from arcface_onnx import ArcFaceONNX
import os.path as osp
import matplotlib.pyplot as plt
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

def main(args):
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

    # Open the RTSP stream
    stream_url = args.rtsp_url
    cap = cv2.VideoCapture(stream_url)

    # Initialize the frame counter
    frame_count = 0


    # Continuously read frames from the RTSP stream
    while True:
        ret, img = cap.read()
        if not ret:
            break

        # Increment the frame counter
        frame_count += 1

        # Only process every 10th frame
        if frame_count % 30 != 0:
            continue

        print("running")

        # Detect the faces in the image
        faces = RetinaFace.detect_faces(img, threshold=confidence_thresh)

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
                    elif sim>=0.15 and sim<0.28 and sim>max_sim:
                        name = names[i]+"**"
                        max_sim = sim
                        conclu = 'They are LIKELY TO be the same person'
                    elif sim>max_sim:
                        name = names[i]
                        max_sim = sim
                        conclu = 'They ARE the same person'

                # Draw a rectangle on the face in the image and write the name of the person above the rectangle
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # # Display the result image
        # cv2.namedWindow("window", cv2.WINDOW_NORMAL)
        # cv2.imshow('window', img)
        # cv2.waitKey(1)
        # Display the result image
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.show(block=False)
        # plt.pause(0.1)
         # Save the result image in the output directory
        output_path = os.path.join(output_dir, str(frame_count)+".jpg")
        cv2.imwrite(output_path, img)

    # Release the resources
    cap.release()
    # cv2.destroyAllWindows()



ap = argparse.ArgumentParser()
ap.add_argument('--input-dir', default='../dataset/test/ch24', help='path to the input directory containing the images to be recognized')
ap.add_argument('--output-dir', default='../results', help='path to the output directory where the result images will be saved')
ap.add_argument('--embed-dir', default='../dataset/cropped_authorized', help='path to the directory containing the authorized embeddings')
ap.add_argument('--rec-model-name', default='w600k_r50.onnx', help='recognizer model name')
ap.add_argument('--detector-thres', default=0.8, help='confidence threshold for face detection')
ap.add_argument('--rtsp-url', default='rtsp://admin:Ntadg@7094@192.168.1.2:554/ch24/0', help='confidence threshold for face detection')

args = ap.parse_args()

main(args)


