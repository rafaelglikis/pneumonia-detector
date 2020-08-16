import os
from tqdm import tqdm
from ml.utils import *
from typing import List
from tensorflow import keras
from abc import ABC, abstractmethod


class Evaluator:
    def __init__(self, size, false_negatives, true_negatives, false_positives, true_positives):
        self.size = size
        self.false_negatives = false_negatives
        self.true_negatives = true_negatives
        self.false_positives = false_positives
        self.true_positives = true_positives

        self.accuracy = (self.true_positives + self.true_negatives) / (
                self.false_negatives + self.true_negatives + self.false_positives + self.true_positives
        )
        self.precision = self.true_positives / (self.true_positives + self.false_positives)
        self.sensitivity = self.true_positives / (self.true_positives + self.false_negatives)
        self.specificity = self.true_negatives / (self.true_negatives + self.false_positives)
        self.f_measure = (2 * self.precision * self.sensitivity) / (self.precision + self.sensitivity)

    def print_confusion_matrix(self):
        print("Confusion Matrix")
        print(f"----------------{self.size} samples")
        print(f" - False Negatives {self.false_negatives}")
        print(f" - True Negatives {self.true_negatives}")
        print(f" - False Positives {self.false_positives}")
        print(f" - True Positives {self.true_positives}")

    def print_evaluation_metrics(self):
        print("Evaluation Metrics")
        print("----------------")
        print(f"Accuracy: {self.accuracy}")
        print(f"Precision: {self.precision}")
        print(f"Sensitivity: {self.sensitivity}")
        print(f"Specificity: {self.specificity}")
        print(f"F Measure: {self.f_measure}")


class Model(ABC):
    def preprocess_image(self, image_path):
        return preprocess_image(
            image_path=image_path,
            target_size=(300, 300)
        )

    @timeit
    def evaluate(self, dir):
        count, fn, tn, fp, tp = 0, 0, 0, 0, 0
        classes = ['NORMAL', 'PNEUMONIA']

        image_class_mapping = {}
        for cls in classes:
            images = os.listdir(f"{dir}/{cls}")
            for img in images:
                image_class_mapping[img] = cls

        for img in tqdm(image_class_mapping):
            cls = image_class_mapping[img]
            count += 1
            prediction = self.predict(f"{dir}/{cls}/{img}")

            if cls == 'NORMAL':
                if prediction[0] > prediction[1]:
                    tn += 1
                else:
                    fn += 1

            elif cls == 'PNEUMONIA':
                if prediction[0] < prediction[1]:
                    tp += 1
                else:
                    fp += 1

        evaluator = Evaluator(
            size=count,
            false_negatives=fn,
            true_negatives=tn,
            false_positives=fp,
            true_positives=tp
        )

        evaluator.print_confusion_matrix()
        evaluator.print_evaluation_metrics()


    @abstractmethod
    def predict(self, image_path):
        pass


class EnsembleUnit(Model):
    def __init__(self, model_path: str, weight: float):
        self.model = keras.models.load_model(model_path)
        self.weight = weight

    def predict(self, image_path):
        image = self.preprocess_image(image_path)
        return self.model.predict(image, batch_size=10)[0]

    def predict_with_weight(self, image_path):
        return self.predict(image_path) * self.weight


class Ensemble(Model):
    def __init__(self, units: List[EnsembleUnit]):
        self.units = units

    def predict(self, image_path) -> List:
        predictions = [unit.predict_with_weight(image_path) for unit in self.units]
        prediction = [0, 0]

        for p in predictions:
            prediction[0] += p[0]
            prediction[1] += p[1]

        return prediction


ensemble = Ensemble([
    EnsembleUnit('ensemble/inception_v3_transfer_20200725-123618', 0.4),  # 0.9439
    EnsembleUnit('ensemble/densenet121_transfer_20200815-105921', 0.2),  # 0.9375
    EnsembleUnit('ensemble/vgg16_v3_transfer_20200726-124845', 0.2),  # 0.9359
    EnsembleUnit('ensemble/xception_20200811-013119', 0.15),  # 0.9343
    EnsembleUnit('ensemble/resnet50_v2_transfer_20200808-134834', 0.05),  # 0.9263
])
