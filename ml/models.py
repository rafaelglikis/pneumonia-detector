from ml.utils import *
from tensorflow import keras
from tensorboard import program
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout


class BaseModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'base_model'
        self.time_str = create_time_str()
        self.log_dir = f"logs/fit/{self.time_str}"
        self.callbacks = [
            TensorBoard(log_dir=self.log_dir),
            EarlyStopping(monitor='loss', patience=5, verbose=1)
        ]

    def launch_tensorboard(self):
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', self.log_dir])
        url = tb.launch()
        print(f"Tensorboard URL: {url}")

    def save(self, **kwargs):
        path = f"models/{self.model_name}_{self.time_str}"
        return super().save(path)


class InceptionV3Transfer(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = 'inception_v3_transfer'

        # Layers
        self.inception_model = InceptionV3(
            include_top=False,
            weights='imagenet',
            input_shape=(300, 300, 3)
        )
        self.inception_model.trainable = True
        self.average_poling = GlobalAveragePooling2D()
        self.dense = Dense(256, activation='relu')
        self.dropout = Dropout(0.5)
        self.output_layer = Dense(2, activation='softmax')

        # Compile model with loss, metrics and optimizer
        self.compile(
            loss=categorical_crossentropy,
            metrics=['accuracy'],
            optimizer=Nadam(lr=1e-4)
        )

    def call(self, inputs, training=None, mask=None):
        x = self.inception_model(inputs)
        x = self.average_poling(x)
        x = self.dense(x)
        x = self.dropout(x, training=training)

        return self.output_layer(x)

    @timeit
    def train(self, train_generator):
        return self.fit(
            train_generator,
            epochs=40,
            steps_per_epoch=50,
            callbacks=self.callbacks
        )
