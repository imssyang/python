"""An example of how to use your own dataset to train a classifier that recognizes people."""
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from datasets import LFWDataset
from facenet import FaceCNN


class Classifier:
    def __init__(self, mode, data_dir, model_path, classifier_path, batch_size=90, image_size=160):
        # data_dir: data directory containing aligned LFW face patches
        # image_size: Image size (height, width) in pixels
        # batch_size: Number of images to process in a batch
        # train_threshold: use this number of images from each class for training
        # images_threshold: classes with at least this number of images in the dataset
        self.lfw = LFWDataset(
            input_dir='datasets/lfw_subset_160',
            images_threshold=1,
            train_threshold=3)
        self.dataset = self.select_dataset(self.lfw, mode)
        self.class_names = [person.name.replace('_', ' ') for person in self.dataset]
        self.image_paths, self.image_labels = LFWDataset.flatten_image_paths(self.dataset)
        self.run(mode, classifier_path, model_path, self.class_names, self.image_paths, self.image_labels, batch_size, image_size)

    def select_dataset(self, lfw, mode):
        if mode == 'TRAIN':
            # Train a new classifier
            return lfw.trains
        elif mode=='TEST':
            # Test classifier
            return lfw.tests

    def run(self, mode, classifier_path, model_path, class_names, image_paths, image_labels, batch_size, image_size):
        with tf.Graph().as_default():
            with tfv1.Session() as session:
                np.random.seed(seed=666)
                if (mode=='TRAIN'):
                    self.train(session, classifier_path, model_path, image_paths, image_labels, class_names, batch_size, image_size)
                elif (mode=='TEST'):
                    self.classify(session, classifier_path, model_path, image_paths, image_labels, batch_size, image_size)

    def train(self, session, classifier_path, model_path, image_paths, image_labels, class_names, batch_size, image_size):
        FaceCNN(model_path)

        emb_array = self.load_image_embeddings(session, image_paths, batch_size, image_size)
        print('Training classifier', image_labels)

        model = SVC(kernel='linear', probability=True)
        model.fit(emb_array, image_labels)
        with open(classifier_path, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
        print(f'Saved classifier model to file {classifier_path}')

    def classify(self, session, classifier_path, model_path, image_paths, image_labels, batch_size, image_size):
        FaceCNN(model_path)

        emb_array = self.load_image_embeddings(session, image_paths, batch_size, image_size)

        print(f'Loaded classifier model from file {classifier_path}')
        with open(classifier_path, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
            predictions = model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            for i in range(len(best_class_indices)):
                print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
            accuracy = np.mean(np.equal(best_class_indices, image_labels))
            print('Accuracy: %.3f' % accuracy)

    def load_image_embeddings(self, session, image_paths, batch_size, image_size):
        # Get input and output tensors
        default_graph = tfv1.get_default_graph()
        images_placeholder = default_graph.get_tensor_by_name("input:0")
        embeddings = default_graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = default_graph.get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        # Run forward pass to calculate embeddings
        print('Calculating features for images')
        image_num = len(image_paths)
        batch_num = int(math.ceil(1.0*image_num / batch_size))
        emb_array = np.zeros((image_num, embedding_size))
        for i in range(batch_num):
            start_index = i*batch_size
            end_index = min((i+1)*batch_size, image_num)
            paths_batch = image_paths[start_index:end_index]
            images = LFWDataset.load_data(paths_batch, False, False, image_size)
            emb_array[start_index:end_index,:] = session.run(embeddings, feed_dict={
                images_placeholder: images,
                phase_train_placeholder: False,
            })
        return emb_array


if __name__ == '__main__':
    Classifier(
        mode='TRAIN',
        data_dir='datasets/lfw_subset_160',
        model_path='models/checkpoints/20170512-110547',
        classifier_path='models/lfw_classifier.pkl',
        batch_size=90,
        image_size=160,
    )
    Classifier(
        mode='TEST',
        data_dir='datasets/lfw_subset_160',
        model_path='models/checkpoints/20170512-110547',
        classifier_path='models/lfw_classifier.pkl',
        batch_size=90,
        image_size=160,
    )
