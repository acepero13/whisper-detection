import pyaudio
import wave

# VAD constants
INSTANCES_VAD_IS_RUN = 0
AVERAGE_INTENSITY_OF_RUNS = 0
OUTPUT_FILE = 'analysis.wav'

# pyaudio constants
PYAUDIO_INSTANCE = pyaudio.PyAudio()
PYAUDIO_CHANNELS = 1
PYAUDIO_RATE = 44100
PYAUDIO_INPUT = True
PYAUDIO_FRAMES_PER_BUFFER = 1024

# Listener constants
NUM_FRAMES = int(PYAUDIO_RATE / PYAUDIO_FRAMES_PER_BUFFER)
LAST_NOTIFICATION_TIME = None


def record(duration):
    """Records Input From Microphone Using PyAudio"""
    print(PYAUDIO_INSTANCE)
    try:
        in_stream = open_channel()
        save_temp_wav(duration, in_stream)
    except IOError as err:
        print("Error", err)
        raise err
    in_stream.close()
    print("End recording")


def save_temp_wav(duration, in_stream):
    out = []
    upper_lim = NUM_FRAMES * duration
    for i in range(0, upper_lim):
        data = in_stream.read(PYAUDIO_FRAMES_PER_BUFFER, exception_on_overflow=False)
        out.append(data)

    # now the writing section where we write to file
    data = b''.join(out)
    out_file = wave.open(OUTPUT_FILE, "wb")
    out_file.setnchannels(1)
    out_file.setsampwidth(PYAUDIO_INSTANCE.get_sample_size(pyaudio.paInt16))
    out_file.setframerate(44100)
    out_file.writeframes(data)
    out_file.close()


def open_channel():
    in_stream = PYAUDIO_INSTANCE.open(
        format=pyaudio.paInt16,
        channels=PYAUDIO_CHANNELS,
        rate=PYAUDIO_RATE,
        input=PYAUDIO_INPUT,
        frames_per_buffer=PYAUDIO_FRAMES_PER_BUFFER
    )
    return in_stream
