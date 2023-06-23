import os
import librosa
import soundfile as sf
import numpy as np
from recursiveFileList import recursiveFileList


def generateFlam(strikePath, flamFolder):
    print('========== generating flam samples ==========')
    strikeList = recursiveFileList(strikePath, 'wav')
    os.makedirs(flamFolder, exist_ok=True)

    alpha = np.arange(0.1, 0.8, 0.1)
    deltaTime = np.arange(30, 70, 10) # in millisecond
    fileCount = 0

    for i in range(len(strikeList)):
        # read input strike sounds
        randInd = np.random.permutation(len(strikeList))
        x_m, fs = librosa.load(strikeList[i], sr=None, mono=True)
        x_g, _ = librosa.load(strikeList[randInd[0]], sr=None, mono=True)

        # normalize input
        x_m = x_m/np.max(np.abs(x_m))
        x_g = x_g/np.max(np.abs(x_g))
        L_m = len(x_m)
        L_g = len(x_g)

        for j in range(len(alpha)):
            for k in range(len(deltaTime)):
                fileCount += 1
                print('Creating file # %g ......' % fileCount)

                currentAlpha = alpha[j]
                currentDt = round(deltaTime[k]/1000 * fs) # in samples
                x_synth = np.zeros(L_m + L_g + currentDt)

                # synthesize flam
                x_synth[0:L_g] = currentAlpha * x_g
                x_synth[currentDt:currentDt+L_m] = x_m

                # normalize the final waveform
                x_synth = x_synth/np.max(np.abs(x_synth))

                # savefile
                flamFilePath = f"{flamFolder}/flam_a{int(currentAlpha*10)}_dt{currentDt}_No{fileCount}.wav"
                sf.write(flamFilePath, x_synth, fs)
                print('Done! \n')
