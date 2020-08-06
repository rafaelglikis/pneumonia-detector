import os

import numpy as np
from tqdm import tqdm
from ml.utils import *
from tensorflow import keras
from tensorboard import program
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.applications import InceptionV3, VGG16, ResNet50V2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint


class Model(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'base_model'
        self.time_str = create_time_str()
        self.log_dir = f"logs/fit/{self.time_str}"
        self.callbacks = [
            TensorBoard(log_dir=self.log_dir),
            EarlyStopping(monitor='loss', patience=5, verbose=1),
            ModelCheckpoint(
                monitor='loss',
                filepath=f"models/best_{self.model_name}_{self.time_str}.h5",
                save_best_only=True
            )
        ]

    def launch_tensorboard(self):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.log_dir])
        url = tb.launch()
        print(f"Tensorboard URL: {url}")

    def save(self, **kwargs):
        path = f"models/{self.model_name}_{self.time_str}"
        return super().save(path)


class TransferModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'inception_v3_transfer'

        # Layers
        self.average_poling = GlobalAveragePooling2D()
        self.dense_256 = Dense(256, activation='relu')
        self.dropout_50 = Dropout(0.5)
        self.output_layer = Dense(2, activation='softmax')

        # Compile model with loss, metrics and optimizer
        self.compile(
            loss=categorical_crossentropy,
            metrics=['accuracy'],
            optimizer=Nadam(lr=1e-4)
        )

    @timeit
    def train(self, train_generator):
        return self.fit(
            train_generator,
            epochs=40,
            steps_per_epoch=50,
            callbacks=self.callbacks
        )


class InceptionV3Transfer(TransferModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'inception_v3_transfer'
        self.inception_model = InceptionV3(
            include_top=False,
            weights='imagenet',
            input_shape=(300, 300, 3)
        )
        self.inception_model.trainable = True

    def call(self, inputs, training=None, mask=None):
        x = self.inception_model(inputs)
        x = self.average_poling(x)
        x = self.dense_256(x)
        x = self.dropout_50(x, training=training)

        return self.output_layer(x)


class VGG16Transfer(TransferModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'vgg16_v2_transfer'
        self.vgg16_model = VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=(300, 300, 3)
        )
        self.vgg16_model.trainable = True

    def call(self, inputs, training=None, mask=None):
        x = self.vgg16_model(inputs)
        x = self.average_poling(x)
        x = self.dense_256(x)
        x = self.dropout_50(x, training=training)

        return self.output_layer(x)


class ResNet50V2Transfer(TransferModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'resnet50_v2_transfer'
        self.resnet50_v2_model = ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_shape=(300, 300, 3)
        )
        self.resnet50_v2_model.trainable = True

        # Compile model with loss, metrics and optimizer
        self.compile(
            loss=categorical_crossentropy,
            metrics=['accuracy'],
            optimizer=Nadam(lr=1e-4)
        )

    def call(self, inputs, training=None, mask=None):
        x = self.resnet50_v2_model(inputs)
        x = self.average_poling(x)
        x = self.dense_256(x)
        x = self.dropout_50(x, training=training)

        return self.output_layer(x)


class EnsembleUnit:
    def __init__(self, model_path: str, weight: float):
        self.model = keras.models.load_model(model_path)
        self.weight = weight

    def predict(self, image_path):
        img = image.load_img(image_path, target_size=(300, 300))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x]) / 255

        return self.model.predict(images, batch_size=10)

    def predict_with_weight(self, image_path):
        return self.predict(image_path) * self.weight

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
            prediction = self.predict(f"{dir}/{cls}/{img}")[0]

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

        print(f"Accuracy: {evaluator.accuracy()}")
