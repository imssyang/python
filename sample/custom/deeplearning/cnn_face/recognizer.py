# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
import os
import sys
import time
import pickle
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from dataclasses import dataclass, field
from typing import Optional
from datasets import Person
from mtcnn import MTCNN
from facenet import FaceCNN


@dataclass
class FaceData:
    name: Optional[str] = None
    image: Optional[np.ndarray] = field(default_factory=lambda: None)
    image_face: Optional[np.ndarray] = field(default_factory=lambda: None)
    bounding_box: Optional[np.ndarray] = field(default_factory=lambda: np.zeros(4, dtype=np.int32))
    embedding: Optional[np.ndarray] = field(default_factory=lambda: None)


class FaceRecognition:
    def __init__(self, model_path, classifier_path, crop_size=160, crop_margin=32):
        self.mtcnn = self._setup_mtcnn()
        self.session = self._load_model_session(model_path)
        self.model, self.class_names = self._load_classifier(classifier_path)
        self.crop_size = crop_size
        self.crop_margin = crop_margin

    def _setup_mtcnn(self, gpu_memory_fraction = 0.3):
        with tf.Graph().as_default():
            gpu_options = tfv1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tfv1.Session(config=tfv1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return MTCNN(sess, None)

    def _load_model_session(self, model_path):
        sess = tfv1.Session()
        with sess.as_default():
            FaceCNN(model_path)
        return sess

    def _load_classifier(self, classifier_path):
        model = None
        class_names = None
        with open(classifier_path, 'rb') as infile:
            model, class_names = pickle.load(infile)
        return model, class_names

    def find_faces(self, image, minsize = 20, threshold = (0.6, 0.7, 0.7), factor = 0.709):
        faces = []
        bounding_boxes, _ = self.mtcnn.detect_face(image, minsize, threshold, factor)
        for bb in bounding_boxes:
            img_size = np.asarray(image.shape)[0:2]
            face = FaceData(image=image)
            face.bounding_box[0] = np.maximum(bb[0] - self.crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.crop_margin / 2, img_size[0])
            cropped_image = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image_face = cv2.resize(cropped_image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
            faces.append(face)
        return faces

    def calc_embedding(self, face):
        # Get input and output tensors
        default_graph = tfv1.get_default_graph()
        images_placeholder = default_graph.get_tensor_by_name("input:0")
        embeddings = default_graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = default_graph.get_tensor_by_name("phase_train:0")

        prewhiten_face = Person.prewhiten(face.image_face)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.session.run(embeddings, feed_dict=feed_dict)[0]

    def identify(self, image):
        faces = self.find_faces(image)
        for i, face in enumerate(faces):
            #cv2.imshow("Face: " + str(i), face.image)
            face.embedding = self.calc_embedding(face)
            if face.embedding is not None:
                predictions = self.model.predict_proba([face.embedding])
                best_class_indices = np.argmax(predictions, axis=1)
                face.name = self.class_names[best_class_indices[0]]
        return faces


class ImageRecognition:
    def __init__(self, model_path, classifier_path):
        self.engine = FaceRecognition(model_path, classifier_path)

    def recognize(self, image_path, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        frame = cv2.imread(image_path)
        faces = self.engine.identify(frame)

        self.add_overlays(frame, faces, frame_rate=0)

        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, frame)
        print('Output:', output_path)

    def add_overlays(self, frame, faces, frame_rate):
        if faces is not None:
            for face in faces:
                face_bb = face.bounding_box.astype(int)
                cv2.rectangle(frame,
                            (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                            (0, 255, 0), 2)
                if face.name is not None:
                    cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                thickness=2, lineType=2)

        cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    thickness=2, lineType=2)


class LiveRecognition(ImageRecognition):
    def __init__(self, model_path, classifier_path):
        super().__init__(model_path, classifier_path)

    def recognize(self):
        frame_interval = 3  # Number of frames after which to run face detection
        fps_display_interval = 5  # seconds
        frame_rate = 0
        frame_count = 0
        video_capture = cv2.VideoCapture(0)
        start_time = time.time()
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            if (frame_count % frame_interval) == 0:
                faces = self.engine.identify(frame)

                # Check our current fps
                end_time = time.time()
                if (end_time - start_time) > fps_display_interval:
                    frame_rate = int(frame_count / (end_time - start_time))
                    start_time = time.time()
                    frame_count = 0

            self.add_overlays(frame, faces, frame_rate)

            frame_count += 1
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    image_recognition = ImageRecognition(
        model_path="models/checkpoints/20170512-110547",
        classifier_path="models/lfw_classifier.pkl",
    )
    image_recognition.recognize(
        image_path='images/Anthony_Hopkins_0001.jpg',
        output_dir='images/out',
    )
