import os
import random

import librosa
import numpy as np
from tensorflow import keras


class FileAudioDataGenerator(keras.utils.Sequence):
    def __init__(self, directory,
                 batch_size,
                 max_length_secs=4.2,
                 class_mode='binary',
                 sr=22050,
                 shuffle=True):
        self.directory = directory
        self.batch_size = batch_size
        self.max_length = int(max_length_secs * sr)
        self.shuffle = shuffle
        self.class_mode = class_mode
        self.sr = sr
        self.labels_dict = dict()

        self.folders = [os.path.join(directory, name) for name in os.listdir(directory) if
                        os.path.isdir(os.path.join(directory, name))]

        self.folders.sort()
        cont = 0
        for folder in self.folders:
            label_name = os.path.basename(folder)
            self.labels_dict[label_name] = cont
            cont += 1

        if class_mode == 'binary':
            assert len(self.folders) == 2
        self.filenames = []
        self.__fill_data()
        random.shuffle(self.filenames)

        self.n = len(self.filenames)

    def __fill_data(self):
        for folder in self.folders:
            for i, filename in enumerate(os.listdir(folder)):
                if filename.endswith(".wav"):
                    self.filenames.append((folder, filename))

    def __read_wav_file(self, folder, filename):
        file_path = os.path.join(folder, filename)
        label_name = os.path.basename(folder)
        audio_file, sr = librosa.load(file_path, sr=self.sr)
        audio_file = librosa.util.fix_length(audio_file, self.max_length)

        if label_name in self.labels_dict:
            return audio_file, self.labels_dict[label_name]
        else:
            self.labels_dict[label_name] = len(self.labels_dict.keys())

        return audio_file, self.labels_dict[label_name]

    def on_epoch_end(self):
        pass
        if self.shuffle:
            np.random.shuffle(self.filenames)

    def __data_generation(self, filenames):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization

        x = np.empty((self.batch_size, self.max_length))
        y = np.empty(self.batch_size, dtype=int)
        # Generate data
        for i, filename in enumerate(filenames):
            # Store sample
            xx, yy = self.__read_wav_file(filename[0], filename[1])

            # Store class
            x[i, ] = xx
            y[i] = yy
            # np.append(y, yy)

        if self.labels_dict['whisper'] != 1:
            print("!!!!!!!!!", self.labels_dict)
        return x, y

    def __getitem__(self, index):
        # returns X,y
        batches = self.filenames[index * self.batch_size:(index + 1) * self.batch_size]
        result = self.__data_generation(batches)
        return result

    def __len__(self):
        return self.n // self.batch_size

    def classes(self):
        return self.labels_dict

    def sample_rate(self):
        return self.sr

    @staticmethod
    def read_wav(filename, sr=22050, max_length_sec=4.2):
        max_length = int(max_length_sec * sr)
        audio_file, sr = librosa.load(filename, sr=sr)
        audio_file = librosa.util.fix_length(audio_file, max_length)
        return audio_file, sr, max_length
