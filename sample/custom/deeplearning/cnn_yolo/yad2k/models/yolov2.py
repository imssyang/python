"""YOLO_v2 Model Defined in Keras."""
import math
import cv2
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Layer, Lambda, concatenate
from tensorflow.keras.models import Model
from ..utils import compose
from .darknet19 import (
    DarknetConv2D, DarknetConv2D_BN_Leaky, Darknet19,
)


def space_to_depth_x2(x):
    """Thin wrapper for Tensorflow space_to_depth with block_size=2."""
    # Import currently required to make Lambda work.
    # See: https://github.com/fchollet/keras/issues/5088#issuecomment-273851273
    return tf.nn.space_to_depth(x, block_size=2)


def space_to_depth_x2_output_shape(input_shape):
    """Determine space_to_depth output shape for block_size=2.

    Note: For Lambda with TensorFlow backend, output shape may not be needed.
    """
    return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2, 4 *
            input_shape[3]) if input_shape[1] else (input_shape[0], None, None,
                                                    4 * input_shape[3])


class YOLOv2:
    def __init__(self, inputs, anchors_path=None, classes_path=None):
        """Generate a complete YOLO_v2 localization model."""
        self.anchors = self.load_anchors(anchors_path)
        self.classes = self.load_classes(classes_path)
        self.model = self.Model(inputs, len(self.anchors), len(self.classes))
        #outputs = yolo_head(body.output, self.anchors, len(self.classes))

    @classmethod
    def Model(cls, inputs, anchors_num, classes_num):
        """Create YOLO_V2 model CNN body in Keras."""
        darknet18 = Model(inputs, Darknet19.first18_layers()(inputs))
        conv13 = darknet18.layers[43]
        conv20 = compose(
            DarknetConv2D_BN_Leaky(1024, (3, 3)),
            DarknetConv2D_BN_Leaky(1024, (3, 3)),
        )(darknet18.output)
        conv21 = DarknetConv2D_BN_Leaky(64, (1, 1))(conv13.output)
        conv21_reshaped = Lambda(
            space_to_depth_x2,
            output_shape=space_to_depth_x2_output_shape,
            name='space_to_depth',
        )(conv21)

        x = concatenate([conv21_reshaped, conv20])
        x = DarknetConv2D_BN_Leaky(1024, (3, 3))(x)
        x = DarknetConv2D(anchors_num * (classes_num + 5), (1, 1))(x)
        return Model(inputs, x)

    @classmethod
    def feat_to_boxes(cls, feats, anchors, classes_num):
        """Convert final layer features to bounding box parameters.

        Parameters
        ----------
        feats : tensor
            Final convolutional layer features.
        anchors : array-like
            Anchor box widths and heights.
        classes_num : int
            Number of target classes.

        Returns
        -------
        box_xy : tensor
            x, y box predictions adjusted by spatial location in conv layer.
        box_wh : tensor
            w, h box predictions adjusted by anchors and conv spatial resolution.
        box_conf : tensor
            Probability estimate for whether each box contains any object.
        box_class_pred : tensor
            Probability distribution estimate for each box over class labels.
        """
        anchors_num = len(anchors)
        anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
        feats = tf.convert_to_tensor(feats, dtype=tf.float32)

        # Reshape to batch, height, width, anchors_num, box_params.
        anchors_tensor = K.reshape(anchors, [1, 1, 1, anchors_num, 2])

        # Dynamic implementation of conv dims for fully convolutional model.
        conv_dims = K.shape(feats)[1:3]  # assuming channels last [13, 13]

        # In YOLO the height index is the inner most iteration. [0-12]
        conv_height_index = K.arange(0, stop=conv_dims[0])
        conv_height_index = K.tile(conv_height_index, [conv_dims[1]])
        conv_width_index = K.arange(0, stop=conv_dims[1])
        conv_width_index = K.tile(K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
        conv_width_index = K.flatten(K.transpose(conv_width_index))
        conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
        conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
        conv_index = K.cast(conv_index, K.dtype(feats))

        feats = K.reshape(feats, [-1, conv_dims[0], conv_dims[1], anchors_num, classes_num + 5])
        conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

        box_xy = K.sigmoid(feats[..., :2])
        box_wh = K.exp(feats[..., 2:4])
        box_confidence = K.sigmoid(feats[..., 4:5])
        box_class_probs = K.softmax(feats[..., 5:])

        # Adjust preditions to each spatial grid point and anchor size.
        # Note: YOLO iterates over height index before width index.
        box_xy = (box_xy + conv_index) / conv_dims
        box_wh = box_wh * anchors_tensor / conv_dims
        return box_xy, box_wh, box_confidence, box_class_probs

    @classmethod
    def filter_boxes(cls, boxes_outputs, image_shape, max_boxes=10, score_threshold=.6, iou_threshold=.5):
        """Evaluate YOLO model on given input batch and return filtered boxes."""
        box_xy, box_wh, box_confidence, box_class_probs = boxes_outputs
        boxes = cls.boxes_to_corners(box_xy, box_wh)
        boxes, scores, classes = cls.filter_threshold_boxes(
            boxes, box_confidence, box_class_probs, threshold=score_threshold,
        )

        # Scale boxes back to original image shape.
        height = image_shape[0]
        width = image_shape[1]
        image_dims = K.stack([height, width, height, width])
        image_dims = K.reshape(image_dims, [1, 4])
        image_dims = K.cast(image_dims, K.dtype(boxes))
        boxes = boxes * image_dims

        # max_boxes_tensor = tf.convert_to_tensor(max_boxes, dtype=tf.int32)
        max_boxes_tensor = K.constant(max_boxes, dtype='int32')
        nms_index = tf.image.non_max_suppression(
            boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
        boxes = K.gather(boxes, nms_index)
        scores = K.gather(scores, nms_index)
        classes = K.gather(classes, nms_index)
        return boxes, scores, classes

    @staticmethod
    def boxes_to_corners(box_xy, box_wh):
        """Convert YOLO box predictions to bounding box corners."""
        box_mins = box_xy - (box_wh / 2.)
        box_maxes = box_xy + (box_wh / 2.)

        return K.concatenate([
            box_mins[..., 0:1],   # x_min
            box_mins[..., 1:2],   # y_min
            box_maxes[..., 0:1],  # x_max
            box_maxes[..., 1:2]   # y_max
        ])

    @staticmethod
    def filter_threshold_boxes(boxes, box_confidence, box_class_probs, threshold=.6):
        """Filter YOLO boxes based on object and class confidence."""
        # 1*13*13*5*1  X  1*13*13*5*20 -->  1*13*13*5*20
        box_scores = box_confidence * box_class_probs
        # 1*13*13*5 get max index
        box_classes = K.argmax(box_scores, axis=-1)
        # 1*13*13*5 get max score
        box_class_scores = K.max(box_scores, axis=-1)
        # 1*13*13*5 get target mask
        prediction_mask = box_class_scores >= threshold

        boxes = tf.boolean_mask(boxes, prediction_mask)
        scores = tf.boolean_mask(box_class_scores, prediction_mask)
        classes = tf.boolean_mask(box_classes, prediction_mask)
        return boxes, scores, classes

    @staticmethod
    def load_anchors(path):
        """loads the anchors from a file"""
        with open(path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2) # shape:(5, 2)

    @staticmethod
    def load_classes(path):
        """loads the classes"""
        with open(path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


class YOLOv2LossLayer(Layer):
    def __init__(self, anchors, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.anchors = anchors
        self.num_classes = num_classes
        self._name = "yolo_loss"

    def compute_output_shape(self, input_shape):
        return (input_shape[0],)

    def call(self, inputs, **kwargs):
        loss = self.loss_fn(inputs, self.anchors, self.num_classes)
        self.add_loss(loss, inputs=True)
        self.add_metric(loss, aggregation="mean", name="yolo_loss")
        return loss

    def get_config(self):
        pass

    def loss_fn(self, args,
                anchors,
                num_classes,
                rescore_confidence=False,
                print_loss=False):
        """YOLO localization loss function.
        Parameters
        ----------
        yolo_output : tensor
            Final convolutional layer features
        true_boxes : tensor
            Ground truth boxes tensor with shape [batch, num_true_boxes, 5]
            containing box x_center, y_center, width, height, and class.
        detectors_mask : array
            0/1 mask for detector positions where there is a matching ground truth.
        matching_true_boxes : array
            Corresponding ground truth boxes for positive detector positions.
            Already adjusted for conv height and width.
        anchors : tensor
            Anchor boxes for model.
        num_classes : int
            Number of object classes.
        rescore_confidence : bool, default=False
            If true then set confidence target to IOU of best predicted box with
            the closest matching ground truth box.
        print_loss : bool, default=False
            If True then use a tf.Print() to print the loss components.
        Returns
        -------
        mean_loss : float
            mean localization loss across minibatch
        """
        (yolo_output, true_boxes, detectors_mask, matching_true_boxes) = args
        num_anchors = len(anchors)
        object_scale = 5
        no_object_scale = 1
        class_scale = 1
        coordinates_scale = 1
        pred_xy, pred_wh, pred_confidence, pred_class_prob = YOLOv2.feat_to_boxes(
            yolo_output, anchors, num_classes)

        # Unadjusted box predictions for loss.
        # TODO: Remove extra computation shared with yolo_head.
        # ?*13*13*125
        yolo_output_shape = K.shape(yolo_output)
        feats = K.reshape(yolo_output, [
            -1, yolo_output_shape[1], yolo_output_shape[2], num_anchors,
            num_classes + 5
        ])
        # ?*13*13*5*4
        pred_boxes = K.concatenate(
            (K.sigmoid(feats[..., 0:2]), feats[..., 2:4]), axis=-1)

        # TODO: Adjust predictions by image width/height for non-square images?
        # IOUs may be off due to different aspect ratio.

        # Expand pred x,y,w,h to allow comparison with ground truth.
        # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
        pred_xy = K.expand_dims(pred_xy, 4)  # ？*13*13*5*1*2
        pred_wh = K.expand_dims(pred_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        true_boxes_shape = K.shape(true_boxes)

        # batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params
        # ？*1*1*1*20*5
        true_boxes = K.reshape(true_boxes, [
            true_boxes_shape[0], 1, 1, 1, true_boxes_shape[1], true_boxes_shape[2]
        ])
        # ？*1*1*1*20*2
        true_xy = true_boxes[..., 0:2]
        true_wh = true_boxes[..., 2:4]

        # Find IOU of each predicted box with each ground truth box.
        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half
        # ?*13*13*5*1*2
        intersect_mins = K.maximum(pred_mins, true_mins)
        intersect_maxes = K.minimum(pred_maxes, true_maxes)
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
        # ？*13*13*5*20
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        # ？*13*13*5*1
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
        # ？*1*1*1*20
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        # ？*13*13*5*20
        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = intersect_areas / union_areas

        # Best IOUs for each location.
        # ？*13*13*5
        best_ious = K.max(iou_scores, axis=4)  # Best IOU scores.
        # ？*13*13*5*1
        best_ious = K.expand_dims(best_ious)

        # A detector has found an object if IOU > thresh for some true box.
        # ？*13*13*5*1
        object_detections = K.cast(best_ious > 0.6, K.dtype(best_ious))

        # TODO: Darknet region training includes extra coordinate loss for early
        # training steps to encourage predictions to match anchor priors.

        # Determine confidence weights from object and no_object weights. 置信度损失
        # NOTE: YOLO does not use binary cross-entropy here.
        # ？*13*13*5*1
        no_object_weights = (no_object_scale * (1 - object_detections) *
                            (1 - detectors_mask))
        no_objects_loss = no_object_weights * K.square(-pred_confidence)

        if rescore_confidence:
            objects_loss = (object_scale * detectors_mask *
                            K.square(best_ious - pred_confidence))
        else:
            objects_loss = (object_scale * detectors_mask *
                            K.square(1 - pred_confidence))
        # ？*13*13*5*1
        confidence_loss = objects_loss + no_objects_loss

        # Classification loss for matching detections. 分类损失
        # NOTE: YOLO does not use categorical cross-entropy loss here.
        # ？*13*13*5*1
        matching_classes = K.cast(matching_true_boxes[..., 4], 'int32')
        # ？*13*13*5*20
        matching_classes = K.one_hot(matching_classes, num_classes)
        # ？*13*13*5*20
        classification_loss = (class_scale * detectors_mask *
                            K.square(matching_classes - pred_class_prob))

        # Coordinate loss for matching detection boxes. 坐标损失
        # ？*13*13*5*4
        matching_boxes = matching_true_boxes[..., 0:4]
        coordinates_loss = (coordinates_scale * detectors_mask *
                            K.square(matching_boxes - pred_boxes))

        # 全部求和为一个值
        confidence_loss_sum = K.sum(confidence_loss)
        classification_loss_sum = K.sum(classification_loss)
        coordinates_loss_sum = K.sum(coordinates_loss)
        total_loss = 0.5 * (
                confidence_loss_sum + classification_loss_sum + coordinates_loss_sum)

        if print_loss:
            tf.print("yolo_loss", total_loss, {"conf_loss": confidence_loss_sum},
                    {"class_loss": classification_loss_sum},
                    {"box_coord_loss": coordinates_loss_sum}, output_stream=sys.stdout)

        return total_loss


class YOLOv2Sequence(Sequence):

    def __init__(self, path, input_shape, batch_size, anchors, num_classes, max_boxes=20, shuffle=True):
        """
        初始化数据发生器
        :param path: 数据路径
        :param input_shape: 模型输入图片大小
        :param batch_size: 一个批次大小
        :param max_boxes: 一张图像中最多的box数量，不足的补充0， 超出的截取前max_boxes个，默认20
        :param shuffle: 数据乱序
        """
        # 1.打开文件
        self.datasets = []
        with open(path, "r")as f:
            self.datasets = f.readlines()
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.datasets))
        self.anchors = anchors
        self.shuffle = shuffle
        self.num_anchors = len(self.anchors)
        self.num_classes = num_classes
        self.max_boxes = max_boxes

    def __len__(self):
        # 计算每一个epoch的迭代次数
        num_images = len(self.datasets)
        return math.ceil(num_images / float(self.batch_size))

    def __getitem__(self, idx):
        # 生成batch_size个索引
        batch_indexs = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        # 根据索引获取datas集合中的数据
        batch = [self.datasets[k] for k in batch_indexs]
        # 生成数据
        X, y = self.data_generation(batch)
        return X, y

    def get_epochs(self):
        return self.__len__()

    def on_epoch_end(self):
        # 在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def preprocess_true_boxes(self, boxes):
        """Find detector in YOLO where ground truth box should appear.

        Parameters
        ----------
        true_boxes : array
            List of ground truth boxes in form of relative x, y, w, h, class.
            Relative coordinates are in the range [0, 1] indicating a percentage
            of the original image dimensions.
        anchors : array
            List of anchors in form of w, h.
            Anchors are assumed to be in the range [0, conv_size] where conv_size
            is the spatial dimension of the final convolutional features.
        image_size : array-like
            List of image dimensions in form of h, w in pixels.

        Returns
        -------
        detectors_mask : array
            0/1 mask for detectors in [conv_height, conv_width, num_anchors, 1]
            that should be compared with a matching ground truth box.
        matching_true_boxes: array
            Same shape as detectors_mask with the corresponding ground truth box
            adjusted for comparison with predicted parameters at training time.

        根据box来计算 detectors_mask 和 matching_true_boxes
        :param boxes:
            List of ground truth boxes in form of relative x, y, w, h, class.
            Relative coordinates are in the range [0, 1] indicating a percentage
            of the original image dimensions.
        :return:
        detectors_mask : array
            0/1 mask for detectors in [conv_height, conv_width, num_anchors, 1]
            that should be compared with a matching ground truth box.
        matching_true_boxes: array
            Same shape as detectors_mask with the corresponding ground truth box
            adjusted for comparison with predicted parameters at training time.
        """
        height, width = self.input_shape
        num_anchors = len(self.anchors)
        # Downsampling factor of 5x 2-stride max_pools == 32.
        # TODO: Remove hardcoding of downscaling calculations.
        assert height % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
        assert width % 32 == 0, 'Image sizes in YOLO_v2 must be multiples of 32.'
        conv_height = height // 32
        conv_width = width // 32
        num_box_params = boxes.shape[1]
        detectors_mask = np.zeros(
            (conv_height, conv_width, num_anchors, 1), dtype=np.float32)
        matching_true_boxes = np.zeros(
            (conv_height, conv_width, num_anchors, num_box_params),
            dtype=np.float32)

        for box in boxes:
            # scale box to convolutional feature spatial dimensions
            box_class = box[4:5]
            box = box[0:4] * np.array(
                [conv_width, conv_height, conv_width, conv_height])
            i = np.floor(box[1]).astype('int')
            j = np.floor(box[0]).astype('int')
            best_iou = 0
            best_anchor = 0
            for k, anchor in enumerate(self.anchors):
                # Find IOU between box shifted to origin and anchor box.
                # 这里假设box和anchor中心点重叠，并且以目标中心点为坐标原点
                box_maxes = box[2:4] / 2.
                box_mins = -box_maxes
                anchor_maxes = (anchor / 2.)
                anchor_mins = -anchor_maxes
                # 计算实际box和anchor的iou，找到iou最大的anchor
                intersect_mins = np.maximum(box_mins, anchor_mins)
                intersect_maxes = np.minimum(box_maxes, anchor_maxes)
                intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
                intersect_area = intersect_wh[0] * intersect_wh[1]
                box_area = box[2] * box[3]
                anchor_area = anchor[0] * anchor[1]
                iou = intersect_area / (box_area + anchor_area - intersect_area)
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = k

            if best_iou > 0:
                # 置位iou最大的anchor
                detectors_mask[i, j, best_anchor] = 1
                adjusted_box = np.array(
                    [
                        box[0] - j, box[1] - i,
                        np.log(box[2] / self.anchors[best_anchor][0]),
                        np.log(box[3] / self.anchors[best_anchor][1]), box_class
                    ],
                    dtype="object")
                    #dtype=np.float32)
                matching_true_boxes[i, j, best_anchor] = adjusted_box
        return detectors_mask, matching_true_boxes

    def get_detector_mask(self, true_boxes):
        detectors_mask = [0 for i in range(len(true_boxes))]
        matching_true_boxes = [0 for i in range(len(true_boxes))]
        for i, box in enumerate(true_boxes):
            detectors_mask[i], matching_true_boxes[i] = self.preprocess_true_boxes(box)

        return np.array(detectors_mask), np.array(matching_true_boxes)

    def read(self, dataset):
        dataset = dataset.strip().split()
        image_path = dataset[0]
        # 读取图片
        image = cv2.imread(image_path)
        # 获取图片原尺寸
        orig_size = np.array([image.shape[1], image.shape[0]])
        orig_size = np.expand_dims(orig_size, axis=0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # opencv读取通道顺序为BGR，所以要转换
        # 将图片resize 到模型要求输入大小
        image = cv2.resize(image, self.input_shape)
        image = image / 255.

        boxes = np.array([np.array(box.split(","), dtype=np.int64) for box in dataset[1:]])
        # 将真实像素坐标转换为(x_center, y_center, box_width, box_height, class)
        # 计算中点坐标
        boxes_xy = 0.5 * (boxes[:, 0:2] + boxes[:, 2:4])
        # 计算实际宽高
        boxes_wh = boxes[:, 2:4] - boxes[:, 0:2]
        # 计算相对原尺寸的中点坐标和宽高
        boxes_xy = boxes_xy / orig_size
        boxes_wh = boxes_wh / orig_size
        # 拼接上面的x，y，w，h，c，为N*5矩阵, N为图像中实际的box数量
        boxes = np.concatenate((boxes_xy, boxes_wh, boxes[:, 4:]), axis=1)
        # 填充boxes
        box_data = np.zeros((self.max_boxes, 5))
        if len(boxes) > self.max_boxes:
            boxes = boxes[:self.max_boxes]
        box_data[:len(boxes)] = boxes
        return image, box_data

    def data_generation(self, batch):
        """
        生成批量数据
        :param batch:
        :return:
        true_boxes:tensor，真实的boxes tensor shape[batch, num_true_boxes, 5]
            containing x_center, y_center, width, height, and class.
        detectors_mask: array detector 掩码，对于iou最大的anchor的位置为1，
            0/1 mask for detector positions where there is a matching ground truth.
        matching_true_boxes: array
            Corresponding ground truth boxes for positive detector positions.
            Already adjusted for conv height and width.
        y: 全零 [batch ]
        """
        images = []
        true_boxes = []
        for dataset in batch:
            image, box = self.read(dataset)
            images.append(image)
            true_boxes.append(box)
        images = np.array(images)
        # true_boxes B*N*5，B为一个批次图像的数量，N表示一个图像中允许的最大目标数量
        true_boxes = np.array(true_boxes)
        # 根据true_boxes 生成detectors_mask和matching_true_boxes
        detectors_mask, matching_true_boxes = self.get_detector_mask(true_boxes)

        return [images, true_boxes, detectors_mask, matching_true_boxes], np.zeros(self.batch_size)
