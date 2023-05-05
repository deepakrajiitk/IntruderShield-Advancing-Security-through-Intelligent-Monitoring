import os
import cv2
import numpy as np
import os.path as osp
import onnxruntime as ort
from retinaface import RetinaFace
from .arcface_onnx import ArcFaceONNX

__all__ = [
    'Recognizer',
]


class Recognizer:
    def __init__(self) -> None:
        self.output_dir = os.path.join(
            os.path.dirname(__file__), '..', 'results')
        self.embed_dir = os.path.join(os.path.dirname(
            __file__), '..', 'dataset', 'cropped_authorized')
        self.confidence_thresh = 0.8
        self.rec_model = 'w600k_r50.onnx'

        ort.set_default_logger_severity(3)

        assets_dir = osp.expanduser('~/.insightface/models/buffalo_l')
        model_path = os.path.join(assets_dir, self.rec_model)
        self.rec = ArcFaceONNX(model_path)
        self.rec.prepare(0)

        # Load the authorized embeddings
        self.embeddings = np.load(os.path.join(
            self.embed_dir, "embeddings.npy"))
        self.names = np.load(os.path.join(self.embed_dir, "names.npy"))
        print(len(self.embeddings), len(self.names))

        # Create a new folder in the output directory with an increasing serial number
        run_number = 1
        while os.path.exists(os.path.join(self.output_dir, f"run{run_number}")):
            run_number += 1
        self.output_dir = os.path.join(self.output_dir, f"run{run_number}")
        os.makedirs(self.output_dir)

        # Create the logs file if it doesn't exist
        self.logs_file = os.path.join(self.output_dir, 'logs.txt')
        if not os.path.exists(self.logs_file):
            open(self.logs_file, 'w').close()

    def save_logs(self, known, unknown_count):
        log1 = "known persons are: " + ', '.join(str(i) for i in known)
        log2 = "total unknows are: " + str(unknown_count)
        # Append the string to the logs file
        with open(self.logs_file, 'a') as f:
            f.write(log1 + '\n')
            f.write(log2 + '\n')

    def save_image(self, img, filename):
        # Save the result image in the output directory
        output_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(output_path, img)

    def recognizer(self, img, prev_objects):
        # Detect the faces in the image
        faces = RetinaFace.detect_faces(img, threshold=self.confidence_thresh)
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

                feat1 = self.rec.get(img, landmarks)
                max_sim = 0
                for i in range(len(self.embeddings)):
                    feat2 = self.embeddings[i]
                    sim = self.rec.compute_sim(feat1, feat2)
                    if sim < 0.15:
                        conclu = 'They are NOT the same person'
                    # elif sim>=0.15 and sim<0.28 and sim>max_sim:
                    #     name = names[i]+"**"
                    #     max_sim = sim
                    #     conclu = 'They are LIKELY TO be the same person'
                    elif sim > max_sim:
                        name = self.names[i]
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
        return img, known, unknown_count
