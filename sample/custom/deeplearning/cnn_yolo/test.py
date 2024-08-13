import os
import random
import colorsys
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
from yad2k.models.keras_yolo import yolo_eval, yolo_head, yolo_body


class TestModel:
    def __init__(self, model_path, anchors_path, classes_path):
        self.anchors = self.get_anchors(anchors_path)
        self.classes = self.get_classes(classes_path)
        self.model = self.load_model(model_path, self.anchors, len(self.classes))

    def predict_image(self, image_path, output_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (416, 416))
        image = image / 255
        image = np.expand_dims(image, 0)  # Add batch dimension.
        y = self.model.predict(image, batch_size=1)
        classes_num = len(self.classes)
        yolo_outputs = yolo_head(y, self.anchors, classes_num)
        boxes, scores, classes = yolo_eval(yolo_outputs, (416, 416), score_threshold=0.6, iou_threshold=0.5)
        print(f'Found {len(boxes)} boxes for {image_path}')

        image = cv2.imread(image_path)
        origin_shape = image.shape[0:2]
        image = cv2.resize(image, (416, 416))
        hsv_tuples = [(x / classes_num, 1., 1.) for x in range(classes_num)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)) # Generate colors for drawing bounding boxes.
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.
        for i, box in enumerate(boxes):
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colors[classes[i]])
            cv2.putText(image, self.classes[classes[i]], (int(box[0]), int(box[1])), 1, 1, colors[classes[i]], 1)

        image = cv2.resize(image, (origin_shape[1], origin_shape[0]))
        cv2.imwrite(output_path, image)
        #cv2.imshow('image', image)
        #cv2.waitKey(0)

    def load_model(self, model_path, anchors, classes_num):
        image_input = Input(shape=(416, 416, 3))
        yolo_model = yolo_body(image_input, len(anchors), classes_num)
        topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)
        final_layer = Conv2D(len(anchors) * (5 + classes_num), (1, 1), activation='linear')(topless_yolo.output)
        model = Model(image_input, final_layer)
        model.load_weights(model_path)
        # model.summary()
        return model

    @staticmethod
    def get_classes(classes_path):
        """
        loads the classes
        :param classes_path: classes file path
        :return: list classes name
        """
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    @staticmethod
    def get_anchors(anchors_path):
        """
        loads the anchors from a file
        :param anchors_path: anchors file path
        :return: array anchors shape:(5, 2)
        """
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)


if __name__ == "__main__":
    model = TestModel(
        model_path='models/yolov2_trained.h5',
        anchors_path='models/yolov2_anchors.txt',
        classes_path='datasets/VOC2007_classes.txt',
    )
    model.predict_image(
        image_path="images/dog.jpg",
        output_path="images/dog_out.jpg",
    )
