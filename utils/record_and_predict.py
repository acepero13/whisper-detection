import librosa
import pyaudio
import wave
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import tensorflow
from tensorflow.keras import models

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 4
WAVE_OUTPUT_FILENAME = "file.wav"
recording = True
n_fft = 1024
hop_length = 256
n_mels = 40
fmin = 20
sr = 16000
fmax = sr / 2
WINDOW_SIZE = 5



count = 0
window = []
frames_to_save = int(sr / CHUNK * RECORD_SECONDS)

print("FRAMES TO SAVE", frames_to_save)

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
        self.model  = models.load_model(model_path)
        self.count = 0

    def start(self):
        self.p = pyaudio.PyAudio()
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
        d = np.frombuffer(in_data, dtype=np.int16)
        self.window.append(numpy_array)
        self.wav_data.append(d)
        #
        if self.frames_saved > frames_to_save:
            print("finished recording")
            data = np.array(self.window).flatten()
            mel_spec_power = librosa.feature.melspectrogram(data, hop_length=hop_length,sr=sr,
                                                            n_mels=n_mels, power=2.0,
                                                            fmin=fmin, fmax=fmax)

            mel_spec_db = librosa.power_to_db(mel_spec_power, ref=np.max)
            plt.imsave("test.png", mel_spec_db)

            # plt.im

            print("SAVI_NG!!!!")

            img = image.load_img('test.png', target_size=(200, 40))
            data = image.img_to_array(img)
            data = np.expand_dims(data, axis=0)
            data = data / 255.
            images = np.vstack([data])

            prediction = self.model.predict(images)

            #print(prediction)

            label = prediction.argmax(axis=-1)
            print("prediction: ", label)
            self.predictions.append(label[0])

            if self.count == 0:
                print("SAVING", len(self.window))
                wf = wave.open('test.wav', 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(np.array(self.wav_data)))
                wf.close()

            if len(self.predictions) > WINDOW_SIZE:
                counts = np.bincount(self.predictions)
                most_probable = np.argmax(counts)
                if(most_probable > 0.5):
                    final_prediction = "Whispering"
                else: final_prediction = "Non-whispering"
                #final_prediction = most_probable > 0.5 
                print('my prediction is: ', final_prediction)
                self.predictions = []

                self.window = []
                self.wav_data = []
                self.count += 1


            self.frames_saved = 0
        self.frames_saved = self.frames_saved + 1

        #

        return None, pyaudio.paContinue

    def mainloop(self):
        while self.stream.is_active():
            time.sleep(2.0)


audio = AudioHandler('/home/alvaro/Documents/ML/whispering/src/models/simple_model_v1_1.h5')
audio.start()  # open the the stream
audio.mainloop()  # main operations with librosa
audio.stop()
