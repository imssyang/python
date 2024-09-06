import os
import cv2
import numpy as np


class Person:
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __repr__(self):
        return f'{self.name}, {len(self.image_paths)} images'

    def __str__(self):
        return f'{self.name}, {len(self.image_paths)} images'

    def __len__(self):
        return len(self.image_paths)

    @classmethod
    def prewhiten(cls, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y


class LFWDataset:
    def __init__(self, input_dir, images_threshold=1, train_threshold=10):
        self.persons = self.get_persons(input_dir)
        self.trains, self.tests = self.split_images(self.persons, images_threshold, train_threshold)

    def get_persons(self, input_dir):
        persons = []
        person_names = [
            person_name for person_name in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, person_name))
        ]
        person_names.sort()

        for i in range(len(person_names)):
            person_name = person_names[i]
            person_dir = os.path.join(input_dir, person_name)
            person_images =self.get_image_paths(person_dir)
            if len(person_images) > 0:
                persons.append(Person(
                    person_name,
                    self.get_image_paths(person_dir),
                ))
            else:
                print(f'{person_name} no any images.')
        return persons

    def get_image_paths(self, person_dir):
        image_paths = []
        if os.path.isdir(person_dir):
            image_names = os.listdir(person_dir)
            image_paths = [os.path.join(person_dir, name) for name in image_names]
        return image_paths

    def split_images(self, persons, images_threshold, train_threshold):
        train_set = []
        test_set = []
        for person in persons:
            image_paths = person.image_paths
            if len(image_paths) >= images_threshold:
                np.random.shuffle(image_paths)
                train_set.append(Person(person.name, image_paths[:train_threshold]))
                test_set.append(Person(person.name, image_paths[train_threshold:]))
        return train_set, test_set

    @classmethod
    def flatten_image_paths(cls, dataset):
        image_paths = []
        image_labels = []
        for i in range(len(dataset)):
            image_paths += dataset[i].image_paths
            image_labels += [i] * len(dataset[i].image_paths)
        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(image_paths))
        return image_paths, image_labels

    @classmethod
    def load_data(cls, image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
        image_num = len(image_paths)
        images = np.zeros((image_num, image_size, image_size, 3))
        for i in range(image_num):
            img = cv2.imread(image_paths[i])
            if img.ndim == 2:
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if do_prewhiten:
                img = Person.prewhiten(img)
            img = cls.crop(img, do_random_crop, image_size)
            img = cls.flip(img, do_random_flip)
            images[i,:,:,:] = img
        return images

    @classmethod
    def crop(cls, image, random_crop, image_size):
        if image.shape[1]>image_size:
            sz1 = int(image.shape[1]//2)
            sz2 = int(image_size//2)
            if random_crop:
                diff = sz1-sz2
                (h, v) = (np.random.randint(-diff, diff+1), np.random.randint(-diff, diff+1))
            else:
                (h, v) = (0,0)
            image = image[(sz1-sz2+v):(sz1+sz2+v),(sz1-sz2+h):(sz1+sz2+h),:]
        return image

    @classmethod
    def flip(cls, image, random_flip):
        if random_flip and np.random.choice([True, False]):
            image = np.fliplr(image)
        return image


