import os
import numpy as np
import xml.etree.ElementTree as ET
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.layers import Input, Conv2D, Lambda
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import plot_model as plot
from tensorflow.keras.callbacks import (
    TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping,
)
from yad2k.models import YOLOv2, YOLOv2LossLayer, YOLOv2Sequence


# http://host.robots.ox.ac.uk/pascal/VOC/voc2007
class Voc2007Data:
    def __init__(self, classes_path):
        self.classes = YOLOv2.load_classes(classes_path)
        for dataset in ['train', 'trainval', 'val', 'test']:
            data_path = f'datasets/VOC2007_{dataset}.txt'
            with open(data_path, 'w') as f:
                dataset_path = f'datasets/VOC2007/ImageSets/Main/{dataset}.txt'
                image_ids = open(dataset_path).read().strip().split()
                for image_id in image_ids:
                    image_path = os.path.abspath(f'datasets/VOC2007/JPEGImages/{image_id}.jpg')
                    f.write(image_path)
                    self._append_annotation(f, image_id)
                    f.write('\n')
            setattr(self, f'{dataset}_path', data_path)
            print(f'{dataset} -> {data_path}')
        print("done")

    def _append_annotation(self, dataset_file, image_id):
        with open(f'datasets/VOC2007/Annotations/{image_id}.xml') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            for obj in root.iter('object'):
                name = obj.find('name').text
                difficult = obj.find('difficult').text
                if name not in self.classes or int(difficult) == 1:
                    continue

                bndbox = obj.find('bndbox')
                bndbox_vals = (
                    int(bndbox.find('xmin').text),
                    int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text),
                    int(bndbox.find('ymax').text),
                )
                bndbox_info = ','.join([str(val) for val in bndbox_vals])
                classes_id = self.classes.index(name)
                dataset_file.write(f' {bndbox_info},{classes_id}')


class TrainModel:
    def __init__(self, pretrained_path, anchors_path, output_path, checkpoint_dir, epochs, data: Voc2007Data):
        self.anchors = YOLOv2.load_anchors(anchors_path)
        self.classes = data.classes
        self.train_path = data.train_path
        self.val_path = data.val_path
        self.train(pretrained_path, output_path, checkpoint_dir, epochs)

    def train(self, pretrained_path, output_path, checkpoint_dir, epochs):
        gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
        for gpu in gpus:
            # Use gpu memory as many as possible
            tf.config.experimental.set_memory_growth(gpu, True)

        classes_num = len(self.classes)
        is_checkpoint = 'checkpoint' in pretrained_path
        model_body, model = self.create_model(self.anchors, classes_num, None if is_checkpoint else pretrained_path)
        model.compile(optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.
        model.summary()
        output_root = os.path.splitext(output_path)[0]
        plot(model, to_file=f'{output_root}.png', show_shapes=True)
        print(f'Saved model plot to {output_root}.png')

        if is_checkpoint:
            print(f'Load checkpoint weight: {pretrained_path}')
            model.load_weights(pretrained_path)

        now_time = datetime.now().strftime("%y%m%d%H%M%S")
        checkpoint_dir = os.path.join(checkpoint_dir, now_time)
        logging = TensorBoard(log_dir=checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, 'yolov2-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5')
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_weights_only=True, save_best_only=True, save_freq='epoch')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

        batch_size = 8 # Effect GPU memory
        input_shape = (416, 416) # Size of input image must be a multiple of 32
        train_sequence = YOLOv2Sequence(self.train_path, input_shape, batch_size, self.anchors, classes_num)
        val_sequence = YOLOv2Sequence(self.val_path, input_shape, batch_size, self.anchors, classes_num)

        model.fit(
            train_sequence,
            steps_per_epoch=train_sequence.get_epochs(),
            validation_data=val_sequence,
            validation_steps=val_sequence.get_epochs(),
            epochs=epochs,
            workers=4,
            callbacks=[checkpoint, early_stopping])
        model.save_weights(output_path)
        print("Done")

    def create_model(self, anchors, classes_num, pretrained_path, freeze_body=True):
        """
        Create yolov2 model
        model_body: YOLOv2 with new output layer
        model: YOLOv2 with custom loss Lambda layer
        """
        # 1. Create Model
        # Input layers, RGB image (416*416*3)
        image_input = Input(shape=(416, 416, 3))
        anchors_num = len(anchors)
        yolo_model = YOLOv2.Model(image_input, anchors_num, classes_num)
        topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

        # 2. Load Weight
        if pretrained_path:
            print(f"Create and load topless weight from: {pretrained_path}")
            pretrained_base, pretrained_ext = os.path.splitext(pretrained_path)
            topless_path = f'{pretrained_base}_topless{pretrained_ext}'
            model_body = load_model(pretrained_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_path)
            topless_yolo.load_weights(topless_path)

        # 3. Freeze yolo body，only train last layer
        if freeze_body:
            for layer in topless_yolo.layers:
                layer.trainable = False

        # 4. Create final CONV2D layer
        final_layer = Conv2D(anchors_num * (5 + classes_num), (1, 1), activation='linear')(topless_yolo.output)
        model_body = Model(image_input, final_layer)

        # 5. Create Loss layer
        # None表示一个图片中的目标可以不确定，boxes_input表示一张图中所有目标信息，可以设置最大值比如20，表示训练数据中一张图片允许最多有20个目标
        boxes_input = Input(shape=(None, 5))
        # 目标掩码，确定目标位于哪一个单元格中的哪一个anchor
        # 13*13为单元格矩阵，5表示每个单元格有5个anchor，第三维度表示如果该anchor有目标就为1，否则为0
        detectors_mask_input = Input(shape=(13, 13, 5, 1))
        # 目标在单元格中的anchor的编码位置和类别信息
        # 最后一个5表示（x, y, w, h, c）c表示类别索引范围是0--1，目标放到了对应的单元格中的anchor
        matching_boxes_input = Input(shape=(13, 13, 5, 5))
        model_loss = YOLOv2LossLayer(anchors, classes_num)(
            [model_body.output, boxes_input, detectors_mask_input, matching_boxes_input],
        )
        model = Model([image_input, boxes_input, detectors_mask_input, matching_boxes_input], model_loss)
        return model_body, model


if __name__ == "__main__":
    data = Voc2007Data(classes_path="datasets/VOC2007_classes.txt")
    TrainModel(
        pretrained_path='models/yolov2.h5',
        #pretrained_path='models/checkpoint/yolov2-ep003-loss51.287-val_loss52.414.h5',
        anchors_path='models/yolov2_anchors.txt',
        output_path='models/yolov2_trained.h5',
        checkpoint_dir='models/checkpoint',
        epochs=1,
        data=data,
    )
