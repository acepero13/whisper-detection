import librosa
import pyaudio
import wave
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models

from extensions.FileAudioDataGenerator import FileAudioDataGenerator
from extensions.MelSpecLayer import MelSpec
from extensions.MinMaxNormalizationLayer import MinMaxNormalization
from utils.Config import Config

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 2.4
WAVE_OUTPUT_FILENAME = "file.wav"
recording = True
n_fft = 1024
hop_length = 256
n_mels = 40
fmin = 20
sr = 44100
fmax = sr / 2
WINDOW_SIZE = 5
config = Config(train_dir=None, val_dir=None, test_dir=None)


class AudioHandler(object):
    def __init__(self, model_path):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = sr
        self.CHUNK = CHUNK
        self.p = None
        self.stream = None
        self.window = []
        self.wav_data = []
        self.frames_saved = 0
        self.predictions = []
        self.model = models.load_model(config.model_path,
                                       custom_objects={'MelSpec': MelSpec, 'MinMaxNormalization': MinMaxNormalization})
        self.count = 0
        self.frames_to_save = int(sr / CHUNK * RECORD_SECONDS)
        print("FRAMES TO SAVE", self.frames_to_save)

    def start(self):
        self.p = pyaudio.PyAudio()
        self.current_audio = 0
        self.frames = []
        print('recording')
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        numpy_array = np.frombuffer(in_data, dtype=np.float32)
        self.frames.append(numpy_array)

        #
        if self.frames_saved > self.frames_to_save:
            self.record()
            self.predict()
            self.reset()
            self.stream.close()
            self.p.close()
        self.frames_saved = self.frames_saved + 1

        #

        return None, pyaudio.paContinue

    def reset(self):
        self.current_audio = self.current_audio + 1
        self.frames = []
        self.frames_saved = 0

    def record(self):
        filename = 'record_' + str(self.current_audio) + '.wav'
        wf = wave.open(filename, "wb")
        wf.setnchannels(self.CHANNELS)
        # set the sample format
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        # set the sample rate
        wf.setframerate(self.RATE)
        # write the frames as bytes
        wf.writeframes(b"".join(self.frames))

    def mainloop(self):
        while self.stream.is_active():
            time.sleep(2.0)

    def predict(self):
        filename = 'record_' + str(self.current_audio) + '.wav'
        wav_file, sr, max_length = FileAudioDataGenerator.read_wav(filename,
                                                                   max_length_sec=config.max_length_sec)
        pred = self.model.predict(np.reshape(wav_file, (1, max_length)))
        prediction = pred[0][0]

        if prediction > 0.5:
            print('!!!!!!!!WHISPERING!!!!', prediction)
        else:
            print('########talking#######', prediction)
        pass


audio = AudioHandler(config.model_path)
audio.start()  # open the the stream
audio.mainloop()  # main operations with librosa
audio.stop()
