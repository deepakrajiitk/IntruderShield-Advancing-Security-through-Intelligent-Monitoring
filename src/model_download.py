import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
# download model from insightface
app = FaceAnalysis(name = 'buffalo_l') 