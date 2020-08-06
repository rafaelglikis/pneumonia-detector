import datetime
from time import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Evaluator:
    def __init__(self, size, false_negatives, true_negatives, false_positives, true_positives):
        self.size = size
        self.false_negatives = false_negatives
        self.true_negatives = true_negatives
        self.false_positives = false_positives
        self.true_positives = true_positives

    def accuracy(self):
        return (self.true_positives + self.true_negatives) / self.size


def create_time_str():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def timeit(f):
    def timed(*args, **kw):
        start = time()
        result = f(*args, **kw)
        end = time()
        print(f"func:{f.__name__} args:[{args}, {kw}] took: {end-start} sec")
        return result

    return timed


def create_generators():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=15,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        vertical_flip=False
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    shape = (300, 300)
    train_dir = 'dataset/chest_xray/train'
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=shape, batch_size=20)
    test_dir = 'dataset/chest_xray/test'
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=shape, batch_size=20)

    return train_generator, test_generator
