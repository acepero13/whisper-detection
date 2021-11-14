import tensorflow.keras.models as models
from keras import layers

from extensions.MelSpecLayer import MelSpec
from extensions.MinMaxNormalizationLayer import MinMaxNormalization
from utils import plotter
from utils.Config import Config

config = Config(train_dir='/home/alvaro/Documents/ML/whispering/dataset/dataset/train',
                val_dir='/home/alvaro/Documents/ML/whispering/dataset/dataset/val',
                test_dir='/home/alvaro/Documents/ML/whispering/dataset/dataset/test')


def train():
    train_generator, validation_generator, test_generator = config.load_data()
    sr = train_generator.sample_rate()
    model = build_simple_model(sr)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(train_generator, epochs=config.epochs, validation_data=validation_generator)
    plot_results(history)
    score = model.evaluate(test_generator)
    print('evaluation: ', score)
    model.save(config.model_path)


def plot_results(history):
    plotter.plot_loss(history)
    plotter.plot_accuracy(history)


def build_simple_model(sr):
    model = models.Sequential()
    model.add(MelSpec(sampling_rate=sr, input_shape=config.input_shape))
    model.add(MinMaxNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


if __name__ == '__main__':
    train()
