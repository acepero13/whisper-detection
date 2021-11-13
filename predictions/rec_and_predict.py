import time

import librosa.util
import numpy as np
import sounddevice as sd
import soundfile as sf
from tensorflow.keras import models

from extensions.FileAudioDataGenerator import FileAudioDataGenerator
from extensions.MelSpecLayer import MelSpec
from extensions.MinMaxNormalizationLayer import MinMaxNormalization
from utils.Config import Config

config = Config(train_dir=None, val_dir=None, test_dir=None)
FILENAME = 'my_audio.wav'


class ConsoleColors:
    HEADER = '\033[95m'
    OK_BLUE = '\033[94m'
    OK_CYAN = '\033[96m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END_C = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def warning(msg: str):
        ConsoleColors.__print(msg, ConsoleColors.FAIL)

    @staticmethod
    def success(msg: str):
        ConsoleColors.__print(msg, ConsoleColors.OK_GREEN)

    @staticmethod
    def __print(msg: str, color):
        print(f"{color}" + msg + f"{ConsoleColors.END_C}")

    @staticmethod
    def green(msg: str):
        ConsoleColors.success(msg)

    @staticmethod
    def red(msg: str):
        ConsoleColors.warning(msg)


class Recorder:
    def __init__(self, duration=3, sr=22050):
        self.duration = duration
        self.sr = sr
        self.model = models.load_model(config.model_path,
                                       custom_objects={'MelSpec': MelSpec, 'MinMaxNormalization': MinMaxNormalization})

    def rec(self):

        ConsoleColors.green('start recording')
        time.sleep(0.2)
        my_data = sd.rec(int(self.sr * self.duration), samplerate=self.sr,
                         channels=2, blocking=True)
        sf.write(FILENAME, my_data, self.sr)
        self.predict()

    def predict(self):
        wav_file, sr, max_length = FileAudioDataGenerator.read_wav(FILENAME,
                                                                   max_length_sec=config.max_length_sec)
        wav_file = librosa.util.fix_length(wav_file, config.max_length)
        model_prediction = self.model.predict(np.reshape(wav_file, (1, config.max_length)))
        prediction = model_prediction[0][0]

        if prediction > 0.5:
            ConsoleColors.green('YOU WERE WHISPERING!!!!')
        else:
            ConsoleColors.red('you were talking :)')


if __name__ == '__main__':
    r = Recorder()
    r.rec()
