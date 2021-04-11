import os.path
import dlib
import numpy as np
import cv2 as cv

from utils import *
import consts
import settings


# NOTE: This class is just a wrapper around 'detect_all_dlib.py' which uses dlib library to
#       to find faces and return them for the next step of face enhancement.
class FaceDetector:
    @staticmethod
    def standard_face_pts():
        return  np.array([196.0, 226.0, 316.0, 226.0, 256.0, 286.0, 220.0, 360.4, 292.0, 360.4],
                         np.float32) / 256.0 - 1.0  # NOTE: In range [-1.0, 1.0]

    @staticmethod
    def compute_transformation_matrix(img, face_landmarks, normalize:bool, target_face_scale:float):
        std_face_pts = FaceDetector.standard_face_pts()
        target_face_pts = ((std_face_pts * target_face_scale + 1.0) / 2.0 * settings.LOAD_SIZE).reshape((5, 2)).astype(np.float32)

        h, w, c = img.shape
        face_landmarks = face_landmarks.astype(np.float32)
        if normalize:
            face_landmarks[:, 0] = face_landmarks[:, 0] / h * 2.0 - 1.0
            face_landmarks[:, 1] = face_landmarks[:, 1] / w * 2.0 - 1.0

        # (affine matrix, inverse affine matrix)
        affine_matrix = cv.estimateAffinePartial2D(face_landmarks, target_face_pts)[0]
        inverse_affine_matrix = cv.invertAffineTransform(affine_matrix)
        return affine_matrix, inverse_affine_matrix

    @staticmethod
    def get_landmark(face_landmarks, id):
        part = face_landmarks.part(id)
        return (part.x, part.y)

    @staticmethod
    def search(face_landmarks):
        x1, y1 = FaceDetector.get_landmark(face_landmarks, 36)
        x2, y2 = FaceDetector.get_landmark(face_landmarks, 39)
        x3, y3 = FaceDetector.get_landmark(face_landmarks, 42)
        x4, y4 = FaceDetector.get_landmark(face_landmarks, 45)

        x_nose, y_nose = FaceDetector.get_landmark(face_landmarks, 30)

        x_left_mouth, y_left_mouth = FaceDetector.get_landmark(face_landmarks, 48)
        x_right_mouth, y_right_mouth = FaceDetector.get_landmark(face_landmarks, 54)

        x_left_eye = (x1 + x2) // 2
        y_left_eye = (y1 + y2) // 2
        x_right_eye = (x3 + x4) // 2
        y_right_eye = (y3 + y4) // 2

        return np.array(
            [
                [x_left_eye, y_left_eye],
                [x_right_eye, y_right_eye],
                [x_nose, y_nose],
                [x_left_mouth, y_left_mouth],
                [x_right_mouth, y_right_mouth],
            ]
        )


    def __init__(self, face_landmarks_weights_filename:str, target_face_scale:float=1.3,
                 output_shape:tuple=(settings.LOAD_SIZE, settings.LOAD_SIZE)):
        if not os.path.isfile(face_landmarks_weights_filename):
            raise FileNotFoundError("Cannot find face landmarks weights file '{:s}' used by dlib library to detect faces in an image!".format(face_landmarks_weights_filename))

        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_locator = dlib.shape_predictor(face_landmarks_weights_filename)
        self.target_face_scale = target_face_scale
        self.output_shape = output_shape

    def __call__(self, img:np.ndarray):
        assert len(img.shape) == 3, "'img' parameter must be an image array with (H, W, C) channel order."
        assert img.shape[-1] == consts.NUM_RGB_CHANNELS, "The last channel must have {:d} dimensions for a RGB image array.".format(consts.NUM_RGB_CHANNELS)

        faces = self.face_detector(img)

        aligned_faces_with_affines = []
        for face in faces:
            face_landmarks = FaceDetector.search(self.landmark_locator(img, face))

            landmarks_affine,\
            inverse_landmarks_affine = FaceDetector.compute_transformation_matrix(img,
                                                                                  face_landmarks,
                                                                                  normalize=False,
                                                                                  target_face_scale=self.target_face_scale)
            aligned_faces_with_affines.append([cv.warpAffine(img, landmarks_affine, self.output_shape),
                                              landmarks_affine,
                                              inverse_landmarks_affine])
        return aligned_faces_with_affines