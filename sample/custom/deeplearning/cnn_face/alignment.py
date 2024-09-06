"""Performs face alignment and stores face thumbnails in the output directory."""
import os
import random
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
from datasets import LFWDataset
from mtcnn import MTCNN


class FaceAlignment:
    """[summary]
    input_dir:
        Directory with unaligned images.
    output_dir:
        Directory with aligned face thumbnails.
    image_size:
        Image size (height, width) in pixels, default=182
    margin', type=int,
        Margin for the crop around the bounding box (height, width) in pixels, default=44
    random_order',
        Shuffles the order of images to enable alignment using multiple processes, default=True
    gpu_memory_fraction:
        Upper bound on the amount of GPU memory that will be used by the process, default=1.0
    detect_multiple_faces:
        Detect and align multiple faces per image, default=False
    """
    def __init__(self, input_dir):
        self.lfw = LFWDataset(input_dir=input_dir)
        self.persons = self.lfw.persons

    def align_mtcnn(self, gpu_memory_fraction, random_order, detect_multiple_faces, margin, image_size, output_dir):
        output_dir = os.path.expanduser(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with tf.Graph().as_default():
            gpu_options = tfv1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tfv1.Session(config=tfv1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                model = MTCNN(sess, None)
                self.align_faces(
                    model,
                    self.persons,
                    random_order=False,
                    detect_multiple_faces=False,
                    margin=32,
                    image_size=160,
                    output_dir=output_dir)

    def align_faces(self, model, persons, random_order, detect_multiple_faces, margin, image_size, output_dir):
        # Add a random key to the filename to allow alignment using multiple processes
        random_key = np.random.randint(0, high=99999)
        bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
        with open(bounding_boxes_filename, "w") as text_file:
            nrof_images_total = 0
            nrof_successfully_aligned = 0
            if random_order:
                random.shuffle(persons)
            for person in persons:
                output_class_dir = os.path.join(output_dir, person.name)
                if not os.path.exists(output_class_dir):
                    os.makedirs(output_class_dir)
                    if random_order:
                        random.shuffle(person.image_paths)
                for image_path in person.image_paths:
                    nrof_images_total += 1
                    filename = os.path.splitext(os.path.split(image_path)[1])[0]
                    output_filename = os.path.join(output_class_dir, filename+'.png')
                    print(image_path)
                    if not os.path.exists(output_filename):
                        try:
                            img = cv2.imread(image_path)
                        except (IOError, ValueError, IndexError) as e:
                            errorMessage = '{}: {}'.format(image_path, e)
                            print(errorMessage)
                        else:
                            minsize = 20 # minimum size of face
                            threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
                            factor = 0.709 # scale factor
                            faces = self.crop_faces(model, img, minsize, threshold, factor, detect_multiple_faces, margin, image_size, output_dir)
                            if not faces:
                                print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % (output_filename))
                                continue
                            for face, bb in faces:
                                nrof_successfully_aligned += 1
                                filename_base, file_extension = os.path.splitext(output_filename)
                                if detect_multiple_faces:
                                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                else:
                                    output_filename_n = "{}{}".format(filename_base, file_extension)
                                cv2.imwrite(output_filename_n, face)
                                text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
        print('Total number of images: %d' % nrof_images_total)
        print('Number of successfully aligned images: %d' % nrof_successfully_aligned)

    def crop_faces(self, model, img, minsize, threshold, factor, detect_multiple_faces, margin, image_size, output_dir):
        scaled_imgs = []
        if img.ndim < 2:
            return scaled_imgs
        if img.ndim == 2:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[:,:,0:3]
        bounding_boxes, _ = model.detect_face(img, minsize, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces>0:
            det = bounding_boxes[:,0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces>1:
                if detect_multiple_faces:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                    img_center = img_size / 2
                    offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                    offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                    index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                    det_arr.append(det[index,:])
            else:
                det_arr.append(np.squeeze(det))
            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-margin/2, 0)
                bb[1] = np.maximum(det[1]-margin/2, 0)
                bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                scaled = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                scaled_imgs.append((scaled, bb))
        return scaled_imgs


if __name__ == '__main__':
    # http://vis-www.cs.umass.edu/lfw
    faces = FaceAlignment(input_dir='datasets/lfw_subset')
    faces.align_mtcnn(
        gpu_memory_fraction = 0.25,
        random_order=False,
        detect_multiple_faces=False,
        margin=32,
        image_size=160,
        output_dir='datasets/lfw_subset_160')
