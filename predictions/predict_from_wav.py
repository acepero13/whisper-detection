import numpy as np
from tensorflow.keras import models

from extensions import FileAudioDataGenerator
from extensions.MelSpecLayer import MelSpec
from extensions.MinMaxNormalizationLayer import MinMaxNormalization
from utils.Config import Config

config = Config(train_dir=None, val_dir=None, test_dir=None)
new_model = models.load_model(config.model_path,
                              custom_objects={'MelSpec': MelSpec, 'MinMaxNormalization': MinMaxNormalization})


def predict_from_wav(filename, expected):
    if filename.endswith(".wav"):
        wav_file, sr, max_length = FileAudioDataGenerator.FileAudioDataGenerator.read_wav(filename,
                                                                                          max_length_sec=config.max_length_sec)
        pred = new_model.predict(np.reshape(wav_file, (1, max_length)))
        prediction = pred[0][0]

        if prediction > 0.5:
            print('Expected:' + expected + ' -> WHISPERING!!!!', prediction)
        else:
            print('Expected:' + expected + ' ->  talking', prediction)


predict_from_wav('/home/alvaro/Music/normal_test.wav', 'talking')
predict_from_wav('/home/alvaro/Music/whisper.wav', 'whisper')
predict_from_wav('/home/alvaro/Music/whisper4_test.wav', 'whisper')
predict_from_wav('/home/alvaro/Music/whisper4.wav', 'whisper')
predict_from_wav('/home/alvaro/Music/whisper2_test.wav', 'whisper')
predict_from_wav(
    '/home/alvaro/Documents/ML/whispering/dataset/dataset/test/whisper/0bbb1366f2928815d0cf64cda0fb4971.wav', 'whisper')
predict_from_wav(
    '/home/alvaro/Documents/ML/whispering/dataset/dataset/test/whisper/97c060dad2b4bac26836d91bf8e5130d.wav', 'whisper')

predict_from_wav(
    '/home/alvaro/Documents/ML/datasets/audio/emotion/ravdess/Actor_01/03-01-02-01-01-02-01.wav',
    'talking')

# img2wav('/home/alvaro/Music/whisper.wav')
# img2wav('/home/alvaro/Documents/ML/whispering/dataset/wav/whisper/8d52965e9327ef3fb6be286d7a513290.wav')
# img2wav('/home/alvaro/Documents/ML/whispering/thorsten-emotional_v02/whisper/4abd7cf4f7e10b5d91570fdb4b9e8138.wav')
# img2wav('/home/alvaro/Documents/ML/whispering/thorsten-emotional_v02/neutral/0aaf116f8774a140909231e0b610dc98.wav')
