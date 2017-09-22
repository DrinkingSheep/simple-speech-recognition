# -*- coding: utf-8 -*-
import librosa
import numpy as np
import matplotlib.pyplot as plt
# librosa.feature.mfcc(y=None, sr=22050, S=None, n_mfcc=20, **kwargs)
file_path='train.wav'


n_mfcc=26
y,sr=librosa.load('train.wav') #로드 단에서 나는 문제는 ffmpeg를 최신버전으로 업데이트하면 해결됨., sr=sr 하면 mfcc 값이 np.ndarray가 아니라고 에러뜸.
print(y)
# Returns:
# M : np.ndarray [shape=(n_mfcc, t)]

 #MFCC sequence

mfcc_y=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=26)
print(y.shape)
#print(mfcc_y)
print(mfcc_y.shape)

plt.plot(mfcc_y)
plt.show()

def
