import time
import wave

import numpy as np
import pyaudio

window = []

CHUNK = 1024
FORMAT = pyaudio.paInt16  # paInt8
CHANNELS = 1
RATE = 16000  # sample rate
RECORD_SECONDS = 5


frames_to_save = int(RATE / CHUNK * RECORD_SECONDS)


class AudioHandler(object):
    def __init__(self):
        #self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = RATE
        self.CHUNK = CHUNK
        self.p = None
        self.stream = None
        self.window = []
        self.frames_saved = 0
        self.predictions = []
        self.counter = 0

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
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
        self.window.append(numpy_array)

        if self.frames_saved > frames_to_save:
            self.save_window()
            self.window = []
            print("SAVI_NG!!!!")

            self.frames_saved = 0
        self.frames_saved = self.frames_saved + 1

        #

        return None, pyaudio.paContinue

    def mainloop(self):
        while self.stream.is_active():
            time.sleep(2.0)

    def save_window(self):
        self.counter = self.counter + 1
        filename = "../records/record_" +  str(self.counter) + ".wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.window))
        wf.close()


audio = AudioHandler()
audio.start()  # open the the stream
audio.mainloop()  # main operations with librosa
audio.stop()