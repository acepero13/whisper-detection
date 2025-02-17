import numpy as np
from tensorflow.keras import models

from extensions.FileAudioDataGenerator import FileAudioDataGenerator
from extensions.MelSpecLayer import MelSpec
from extensions.MinMaxNormalizationLayer import MinMaxNormalization
from predictions.rec_and_predict import ConsoleColors as Colors
from utils.Config import Config

config = Config(train_dir=None, val_dir=None, test_dir=None)
new_model = models.load_model(config.model_path,
                              custom_objects={'MelSpec': MelSpec, 'MinMaxNormalization': MinMaxNormalization})


def predict_from_wav(filename, expected):
    if filename.endswith(".wav"):
        wav_file, sr, max_length = FileAudioDataGenerator.read_wav(filename,
                                                                   max_length_sec=config.max_length_sec)
        model_prediction = new_model.predict(np.reshape(wav_file, (1, max_length)))
        prediction = model_prediction[0][0]

        if prediction > 0.5:
            print_whisper_prediction(expected, prediction)
        else:
            print_talking_prediction(expected, prediction)


def print_talking_prediction(expected, prediction):
    if expected == 'talking':
        Colors.green('Talking ' + str(prediction))
    else:
        Colors.warning('Expected:' + expected + ' -> TALKING :(' + str(prediction))


def print_whisper_prediction(expected, prediction):
    if expected == 'whisper':
        Colors.green('WHISPERING!!!! ' + str(prediction))
    else:
        Colors.warning('Expected:' + expected + ' -> WHISPERING!!!!' + str(prediction))


if __name__ == '__main__':
    predict_from_wav('/home/alvaro/Music/normal_test.wav', 'talking')
    predict_from_wav('/home/alvaro/Music/whisper.wav', 'whisper')
    predict_from_wav('/home/alvaro/Music/whisper4_test.wav', 'whisper')
    predict_from_wav('/home/alvaro/Music/whisper4.wav', 'whisper')
    predict_from_wav('/home/alvaro/Music/whisper2_test.wav', 'whisper')
    predict_from_wav(
        '/home/alvaro/Documents/ML/whispering/dataset/dataset/test/whisper/0bbb1366f2928815d0cf64cda0fb4971.wav',
        'whisper')
    predict_from_wav(
        '/home/alvaro/Documents/ML/whispering/dataset/dataset/test/whisper/97c060dad2b4bac26836d91bf8e5130d.wav',
        'whisper')

    predict_from_wav(
        '/home/alvaro/Documents/ML/datasets/audio/emotion/ravdess/Actor_01/03-01-02-01-01-02-01.wav',
        'talking')

