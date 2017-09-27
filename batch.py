# -*- coding: utf-8 -*-
import librosa
import math as m
import numpy as np
import matplotlib.pyplot as plt
import os
# librosa.feature.mfcc(y=None, sr=22050, S=None, n_mfcc=20, **kwargs)
dir='command'
foldername='wav'
filenames=os.listdir(foldername)
n_mfcc=50
LOOK_LEN=1800
count=1
counth=0
command=[]
hop_length=512
def mfccgen(dir,step):
    count=0
    global command  #글로벌로 선언 안하면 로컬에러남.
    folder=os.listdir(dir)
    i=0

    while True:

        folderpath=dir+'/'+folder[i]

        file = os.listdir(folderpath)
        for j in range(len(file)): # 폴더안에 하나씩 파일 이름 읽음.

            full_filename=folderpath+'/'+file[j]
            #print(full_filename)
            y,sr=librosa.load(full_filename,sr=8000) #로드 단에서 나는 문제는 ffmpeg를 최신버전으로 업데이트하면 해결됨., sr=sr 하면 mfcc 값이 np.ndarray가 아니라고 에러뜸.
            length=len(y)
            if length>3600:
                temp=y[m.floor(length/2-LOOK_LEN):m.floor(length/2+LOOK_LEN)]
                #print(str(length)+'is length of wav')
                #print(str(len(temp))+'is length of temp')
                mfcc=librosa.feature.mfcc(temp,sr=8000,hop_length=hop_length,n_mfcc=n_mfcc)
                #print(mfcc)
                #print(mfcc.shape)
                if folder[i]=='andwae':
                    command = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif folder[i]=='anza':
                    command = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif folder[i]=='byeonsin':
                    command = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                elif folder[i]=='dalyeo':
                    command = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                elif folder[i]=='gaza':
                    command = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                elif folder[i]=='ireona' or folder[i]=='ireoseo' :
                    command = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
                elif folder[i]=='iriwa':
                    command = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                elif folder[i]=='nuweo':
                    command = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                elif folder[i]=='tuieo':
                    command = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
                elif folder[i]=='upduryeo':
                    command = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
                elif folder[i]=='zizeo':
                    command = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                else:
                    print('the file is not assigned')
                    return 1
                count=count+1
                #elif folder[i]=='silence':
                 #   command=11




                yield mfcc,command
            else:
                pass
        i=i+1
        if i==len(folder)-1:
            i=0
        if count==step:
            break

