import argparse
from ml.utils import *
from ml.models import Model, create_model
from ml.ensemble import Ensemble, ensemble, EnsembleUnit


def parse_commandline():
    models = ['inception', 'vgg16', 'resnet50', 'densenet121', 'xception', 'mobilenet']
    parser = argparse.ArgumentParser(description='Detect pneumonia from chest x rays.')
    parser.add_argument('--train', nargs=1, dest='model', choices=models, help='Train a model.', )
    parser.add_argument('--evaluate', nargs='+', help='Evaluate trained model.')

    ensemble_options = ['evaluate']
    parser.add_argument('--ensemble', nargs=1, choices=ensemble_options, help='Use ensemble.')

    return parser.parse_args()


def train(train_gen, test_gen, model: Model):
    model.launch_tensorboard()
    history = model.train(train_gen)
    model.evaluate(test_gen)
    model.save()

    return history


def evaluate(filepath):
    print(f"Loading: {filepath}")
    print(f"Evaluating: {filepath}")
    model = EnsembleUnit(filepath)
    model.evaluate('dataset/chest_xray/test')


def evaluate_ensemble(ensemble: Ensemble):
    ensemble.evaluate('dataset/chest_xray/test')


if __name__ == "__main__":
    args = parse_commandline()

    if args.ensemble:
        if args.ensemble[0] == 'evaluate':
            evaluate_ensemble(ensemble)

    if args.model:
        train_generator, test_generator = create_generators()
        model = create_model(args.model[0])
        train(train_generator, test_generator, model)

    if args.evaluate:
        for path in args.evaluate:
            evaluate(path)
