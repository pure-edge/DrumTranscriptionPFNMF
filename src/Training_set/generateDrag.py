import os
import random
import librosa
import soundfile as sf
import numpy as np
from recursiveFileList import recursiveFileList

def generateDrag(strikePath, dragFolder):
    print('========== generating drag samples ==========')
    strikeList = recursiveFileList(strikePath, 'wav')
    os.makedirs(dragFolder, exist_ok=True)

    alpha = np.arange(0.15, 0.65, 0.1)
    deltaTime1 = np.arange(50, 80, 10) # in millisecond
    deltaTime2 = np.arange(45, 85, 10) 
    fileCount = 0

    for i in range(len(strikeList)):
        # read input strike sounds
        randInd = random.sample(range(len(strikeList)), 2)
        x_m, fs = librosa.load(strikeList[i], sr=None)
        x_g1, _ = librosa.load(strikeList[randInd[0]], sr=None)
        x_g2, _ = librosa.load(strikeList[randInd[1]], sr=None)

        # normalize input
        x_m = x_m/np.max(np.abs(x_m))
        x_g1 = x_g1/np.max(np.abs(x_g1))
        x_g2 = x_g2/np.max(np.abs(x_g2))
        L_m = len(x_m)
        L_g1 = len(x_g1)
        L_g2 = len(x_g2)

        for j in range(len(alpha)):
            for k in range(len(deltaTime1)):
                for kk in range(len(deltaTime2)):
                    fileCount = fileCount + 1
                    print(f'Creating file # {fileCount} ......')

                    currentAlpha = alpha[j]
                    currentDt1 = round(deltaTime1[k]/1000 * fs) # in samples
                    currentDt2 = round(deltaTime2[kk]/1000 * fs)
                    x_synth = np.zeros(L_m + L_g1 + L_g2 + currentDt1 + currentDt2)

                    # synthesize drag
                    x_synth[0:L_g1] = currentAlpha * x_g1
                    x_synth[currentDt1: currentDt1 + L_g2] += currentAlpha * x_g2
                    x_synth[currentDt1 + currentDt2: currentDt1 + currentDt2 + L_m] += x_m

                    # normalize the final waveform
                    x_synth = x_synth/np.max(np.abs(x_synth))

                    # save file
                    dragFilePath = os.path.join(dragFolder, f'drag_a{int(alpha[j]*100)}_dt1_{deltaTime1[k]}_dt2_{deltaTime2[kk]}_No{fileCount}.wav')
                    sf.write(dragFilePath, x_synth, fs)
                    print('Done! ')
