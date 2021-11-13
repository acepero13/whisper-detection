import time

import librosa.util
import numpy as np
import sounddevice as sd
import soundfile as sf
from tensorflow.keras import models

from extensions import FileAudioDataGenerator
from extensions.MelSpecLayer import MelSpec
from extensions.MinMaxNormalizationLayer import MinMaxNormalization
from utils.Config import Config

config = Config(train_dir=None, val_dir=None, test_dir=None)
FILENAME = 'my_audio.wav'


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Recorder:
    def __init__(self, duration=3, sr=22050):
        self.duration = duration
        self.sr = sr
        self.model = models.load_model(config.model_path,
                                       custom_objects={'MelSpec': MelSpec, 'MinMaxNormalization': MinMaxNormalization})

    def rec(self):

        print(f"{bcolors.OKGREEN}start recording{bcolors.ENDC}")
        time.sleep(0.2)
        my_data = sd.rec(int(self.sr * self.duration), samplerate=self.sr,
                         channels=2, blocking=True)
        sf.write(FILENAME, my_data, self.sr)
        self.predict()

    def predict(self):
        wav_file, sr, max_length = FileAudioDataGenerator.FileAudioDataGenerator.read_wav(FILENAME,
                                                                                          max_length_sec=config.max_length_sec)
        wav_file = librosa.util.fix_length(wav_file, config.max_length)
        pred = self.model.predict(np.reshape(wav_file, (1, config.max_length)))
        prediction = pred[0][0]

        if prediction > 0.5:
            print(f"{bcolors.OKGREEN}YOU WERE WHISPERING!!!!{bcolors.ENDC}")
        else:
            print(f"{bcolors.FAIL}you were talking :){bcolors.ENDC}")


r = Recorder()
r.rec()
