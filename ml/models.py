from ml.utils import *
from tensorflow import keras
from tensorboard import program
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.applications import InceptionV3, VGG16, ResNet50V2, DenseNet121, Xception, MobileNetV2


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
                save_best_only=True,
                filepath=f"models/best_{self.model_name}_{self.time_str}"
            )
        ]

    def launch_tensorboard(self):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.log_dir])
        url = tb.launch()
        print(f"Tensorboard URL: {url}")

    def save(self, **kwargs):
        path = f"models/{self.model_name}_{self.time_str}"
        print(f"Saving Model to {path}")
        return super().save(path)


class TransferModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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

    def top(self, inputs, training):
        inputs = self.average_poling(inputs)
        inputs = self.dense_256(inputs)
        inputs = self.dropout_50(inputs, training=training)

        return self.output_layer(inputs)

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


class DenseNet121Transfer(TransferModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'densenet121_transfer'
        self.densenet121 = DenseNet121(
            include_top=False,
            weights='imagenet',
            input_shape=(300, 300, 3)
        )
        self.densenet121.trainable = True

        # Compile model with loss, metrics and optimizer
        self.compile(
            loss=categorical_crossentropy,
            metrics=['accuracy'],
            optimizer=Nadam(lr=1e-4)
        )

    def call(self, inputs, training=None, mask=None):
        x = self.densenet121(inputs)
        return self.top(x, training)


class XceptionTransfer(TransferModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'xception'
        self.xception = Xception(
            include_top=False,
            weights='imagenet',
            input_shape=(300, 300, 3)
        )
        self.xception.trainable = True

        # Compile model with loss, metrics and optimizer
        self.compile(
            loss=categorical_crossentropy,
            metrics=['accuracy'],
            optimizer=Nadam(lr=1e-4)
        )

    def call(self, inputs, training=None, mask=None):
        x = self.xception(inputs)
        return self.top(x, training)


class MobileNetV2Transfer(TransferModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'mobilenet'
        self.mobilenet_v2 = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(300, 300, 3)
        )
        self.mobilenet_v2.trainable = True

        # Compile model with loss, metrics and optimizer
        self.compile(
            loss=categorical_crossentropy,
            metrics=['accuracy'],
            optimizer=Nadam(lr=1e-4)
        )

    def call(self, inputs, training=None, mask=None):
        x = self.mobilenet_v2(inputs)
        return self.top(x, training)


def create_model(model):
    if model == 'inception':
        return InceptionV3Transfer()
    elif model == 'vgg16':
        return VGG16Transfer()
    elif model == 'resnet50':
        return ResNet50V2Transfer()
    elif model == 'densenet121':
        return DenseNet121Transfer()
    elif model == 'xception':
        return XceptionTransfer()
    elif model == 'mobilenet':
        return MobileNetV2Transfer()
    else:
        raise NameError(f"Invalid Model {model}")
