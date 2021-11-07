import numpy as np
from tensorflow.keras import models

from extensions import FileAudioDataGenerator
from extensions.MelSpecLayer import MelSpec
from extensions.MinMaxNormalizationLayer import MinMaxNormalization
from utils.Config import Config

config = Config(train_dir=None, val_dir=None, test_dir=None)
new_model = models.load_model(config.model_path,
                              custom_objects={'MelSpec': MelSpec, 'MinMaxNormalization': MinMaxNormalization})


def predict_from_wav(filename):
    if filename.endswith(".wav"):
        wav_file, sr, max_length = FileAudioDataGenerator.FileAudioDataGenerator.read_wav(filename,
                                                                                          max_length_sec=config.max_length_sec)
        pred = new_model.predict(np.reshape(wav_file, (1, max_length)))
        prediction = pred[0][0]

        if prediction > 0.5:
            print('WHISPERING!!!!', prediction)
        else:
            print('talking', prediction)


predict_from_wav('/home/alvaro/Music/normal_test.wav')
predict_from_wav('/home/alvaro/Music/whisper.wav')
predict_from_wav('/home/alvaro/Music/whisper3_test.wav')
predict_from_wav('/home/alvaro/Music/whisper2.wav')
predict_from_wav('/home/alvaro/Music/whisper2_test.wav')
predict_from_wav(
    '/home/alvaro/Documents/ML/whispering/dataset/dataset/test/whisper/4abd7cf4f7e10b5d91570fdb4b9e8138.wav')
predict_from_wav(
    '/home/alvaro/Documents/ML/whispering/dataset/dataset/test/whisper/cafd435e393351bf7b65cf440e6f5720.wav')
predict_from_wav(
    '/home/alvaro/Documents/ML/whispering/dataset/dataset/test/neutral/5ea8176d00be54145e493f680e861258.wav')

# img2wav('/home/alvaro/Music/whisper.wav')
# img2wav('/home/alvaro/Documents/ML/whispering/dataset/wav/whisper/8d52965e9327ef3fb6be286d7a513290.wav')
# img2wav('/home/alvaro/Documents/ML/whispering/thorsten-emotional_v02/whisper/4abd7cf4f7e10b5d91570fdb4b9e8138.wav')
# img2wav('/home/alvaro/Documents/ML/whispering/thorsten-emotional_v02/neutral/0aaf116f8774a140909231e0b610dc98.wav')
