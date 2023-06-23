from NmfDrum import NmfDrum
from FeaturesTimbre import FeaturesTimbre
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

def demo():
    # read file
    filePath = 'test_audio.wav'
    x, fs = librosa.load(filePath)
    t = np.arange(0, len(x)) / fs
    # transcription
    hh, bd, sd = NmfDrum(filePath, 'Am2')
    
    # audio playback
    sf.write('temp_audio.wav', x, fs)
    #librosa.play('temp_audio.wav')
    #print(f"hh: {hh}")
    #print(f"bd: {bd}")
    #print(f"sd: {sd}")

    # visualization
    plt.subplot(411)
    plt.plot(t, x, 'k')
    plt.title('Original Waveform')
    plt.xlabel('Time (sec)')
    plt.ylabel('Amplitude')

    plt.subplot(412)
    plt.stem(hh, np.ones(len(hh)), 'r')
    plt.title('HiHat Onsets')
    plt.xlabel('Time (sec)')
    plt.ylabel('Activity')

    plt.subplot(413)
    if len(sd) > 0:
        plt.stem(sd, np.ones(len(sd)), 'g')
    plt.title('Snare Drum Onsets')
    plt.xlabel('Time (sec)')
    plt.ylabel('Activity')

    plt.subplot(414)
    plt.stem(bd, np.ones(len(bd)), 'b')
    plt.title('Bass Drum Onsets')
    plt.xlabel('Time (sec)')
    plt.ylabel('Activity')

    #plt.show()

demo()