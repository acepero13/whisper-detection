import os
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt

from utils import audio

n_fft = 1024
hop_length = 256
n_mels = 40
fmin = 20
max_length = 192610
max = -1
class FileTransformer():

    def transform_wav2_img(self, src_dir, dst_dir):
        folders = [name for name in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, name))]
        for folder in folders:
            wav_folder = src_dir + os.path.sep + folder
            img_folder = dst_dir + os.path.sep + folder
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
            self.transform_path2(wav_folder, img_folder)
        print("Max:", max)



    def transform_path2(self, wav_src, img_dst):
        global max
        for i, filename in enumerate(os.listdir(wav_src)):

            if filename.endswith(".wav"):
                audiofile, sr = librosa.load(wav_src + '/' + filename)
                audiofile = librosa.util.fix_length(audiofile, max_length)
                # normalized_audio_file = librosa.util.normalize(audiofile)
                y = audio.normalize(audiofile, -20)
                fmax = sr / 2
                mel_spec_power = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft,
                                                                hop_length=hop_length,
                                                                n_mels=n_mels, power=2.0,
                                                                fmin=fmin, fmax=fmax)


                chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                #rmse = librosa.feature.rmse(y=y)
                spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                zcr = librosa.feature.zero_crossing_rate(y)
                mfcc = librosa.feature.mfcc(y=y, sr=sr)

                mel_spec_db = librosa.power_to_db(mel_spec_power, ref=np.max)
                dst_fname = img_dst + '/' + (filename[:-4] + '.png')
                plt.imsave(dst_fname, mel_spec_db)


    def transform_path(self, wav_src, img_dst):
        global max
        for i, filename in enumerate(os.listdir(wav_src)):

            if filename.endswith(".wav"):                                                           
                audiofile, sr = librosa.load(wav_src + '/' + filename)
                audiofile = librosa.util.fix_length(audiofile, max_length)
                # normalized_audio_file = librosa.util.normalize(audiofile)
                normalized_audio_file = audio.normalize(audiofile, -20)
                fmax = sr / 2
                mel_spec_power = librosa.feature.melspectrogram(normalized_audio_file, sr=sr, n_fft=n_fft,
                                                                hop_length=hop_length,
                                                                n_mels=n_mels, power=2.0,
                                                                fmin=fmin, fmax=fmax)

                mel_spec_db = librosa.power_to_db(mel_spec_power, ref=np.max)
                dst_fname = img_dst + '/' + (filename[:-4] + '.png')
                plt.imsave(dst_fname, mel_spec_db)