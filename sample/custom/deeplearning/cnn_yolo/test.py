import os
import random
import colorsys
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
from yad2k.models import YOLOv2


class TestModel:
    def __init__(self, model_path, anchors_path, classes_path):
        self.anchors = YOLOv2.load_anchors(anchors_path)
        self.classes = YOLOv2.load_classes(classes_path)
        self.model = self.load_model(model_path, self.anchors, len(self.classes))

    def predict_image(self, image_path, output_path=None):
        if not output_path:
            image_dir, image_name = os.path.split(image_path)
            output_dir = os.path.join(image_dir, 'out')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_path = os.path.join(output_dir, image_name)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (416, 416))
        image = image / 255
        image = np.expand_dims(image, 0)  # Add batch dimension.
        y = self.model.predict(image, batch_size=1)
        classes_num = len(self.classes)
        yolo_outputs = YOLOv2.feat_to_boxes(y, self.anchors, classes_num)
        boxes, scores, classes = YOLOv2.filter_boxes(yolo_outputs, (416, 416), score_threshold=0.6, iou_threshold=0.5)
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
        yolo_model = YOLOv2.Model(image_input, len(anchors), classes_num)
        topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)
        final_layer = Conv2D(len(anchors) * (5 + classes_num), (1, 1), activation='linear')(topless_yolo.output)
        model = Model(image_input, final_layer)
        model.load_weights(model_path)
        # model.summary()
        return model


if __name__ == "__main__":
    model = TestModel(
        model_path='models/yolov2_trained.h5',
        anchors_path='models/yolov2_anchors.txt',
        classes_path='datasets/VOC2007_classes.txt',
    )
    for root, dirs, files in os.walk('images'):
        for file in files:
            image_path = os.path.join(root, file)
            model.predict_image(
                image_path=image_path,
            )
