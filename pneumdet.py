import argparse
from ml.utils import *
import tensorflow as tf
from ml.models import InceptionV3Transfer, VGG16Transfer, ResNet50V2Transfer

tf.get_logger().setLevel('WARNING')


def parse_commandline():
    models = ['inception', 'vgg16', 'resnet50']
    parser = argparse.ArgumentParser(description='Detect pneumonia from chest x rays.')
    parser.add_argument('--train', nargs=1, dest='model', choices=models, help='Train a model.', )
    parser.add_argument('--evaluate', nargs=1, help='Evaluate trained model.')

    return parser.parse_args()


def train(train_gen, test_gen, model):
    model.launch_tensorboard()
    history = model.train(train_gen)
    model.evaluate(test_gen)
    model.save()

    return history


def evaluate(test_gen, filepath):
    print(f"Loading: {filepath}")
    model = tf.keras.models.load_model(filepath)
    print(f"Evaluating: {filepath}")
    model.evaluate(test_gen)


if __name__ == "__main__":
    args = parse_commandline()
    print(args)

    if args.model:
        train_generator, test_generator = create_generators()
        model = args.model[0]
        if model == 'inception':
            train(train_generator, test_generator, InceptionV3Transfer())
        elif model == 'vgg16':
            train(train_generator, test_generator, VGG16Transfer())
        elif model == 'resnet50':
            train(train_generator, test_generator, ResNet50V2Transfer())

    if args.evaluate:
        _, test_generator = create_generators()
        for path in args.evaluate:
            evaluate(test_generator, path)
