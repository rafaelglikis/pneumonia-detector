from ml.utils import *
import tensorflow as tf
from ml.models import InceptionV3Transfer
tf.get_logger().setLevel('WARNING')

if __name__ == "__main__":
    train_generator, test_generator = create_generators()

    model = InceptionV3Transfer()
    model.launch_tensorboard()
    history = model.train(train_generator)
    model.evaluate(test_generator)
    model.save()
