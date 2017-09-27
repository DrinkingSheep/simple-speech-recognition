# -*- coding: utf-8 -*-
import tensorflow as tf
import librosa
import numpy as np
import os
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def conv_net(X,W,b,keepprob,mfcc_n,img_size):
    input_img=tf.reshape(X,shape=[-1,mfcc_n,img_size,1])
    # conv_net
    layer1=tf.nn.relu(tf.add(tf.nn.conv2d(input_img,W['w1'],strides=[1,1,1,1],padding='SAME'),b['b1']))
    layer1=tf.nn.dropout(layer1,keepprob)
    print(layer1)
    layer2=tf.nn.relu(tf.add(tf.nn.conv2d(layer1,W['w2'],strides=[1,1,1,1],padding='SAME'),b['b2']))
    layer2=tf.nn.dropout(layer2,keepprob)
    print(layer2)
    layer3=tf.nn.relu(tf.add(tf.nn.conv2d(layer2,W['w3'],strides=[1,1,1,1],padding='SAME'),b['b3']))
    layer3=tf.nn.dropout(layer3,keepprob)
    print(layer3)
    layer4=tf.nn.relu(tf.add(tf.nn.conv2d(layer3,W['w4'],strides=[1,1,1,1],padding='SAME'),b['b4']))
    layer4=tf.nn.max_pool(layer4,ksize=[1,2,2,1],strides=[1,2,1,1],padding='SAME')
    #(13,20)
    #layer4=tf.nn.dropout(layer2,keepprob)

    layer4=tf.reshape(layer4,[-1,65])


    print(layer4)
    layer5= tf.matmul(layer4,W['dw1']+b['db1'])
    layer6 = tf.matmul(layer5, W['dw2'] + b['db2'])

    layer7 = tf.matmul(layer6, W['dw3'] + b['db3'])

    layer8=tf.matmul(layer7,W['w5']+b['b5'])
    print (layer8)


    return layer8

'''
def recur_net(X,W,b):
#말과 말 사이의 시간을 타임 스텝으로 잡음. 10000/512 약 20개.
    X=tf.unstack(X,20,1)

    cell = tf.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    #cell=tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    outputs, states=tf.rnn.static_rnn(cell,X,dtype=tf.float32)

    return tf.matmul(outputs[-1],W['out'])+b['out']

'''
