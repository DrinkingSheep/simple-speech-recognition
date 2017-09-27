# -*- coding: utf-8 -*-
import tensorflow as tf
import librosa
import numpy as np
import os
import batch

import core
train_dir='command_train'
test_dir='command_test'


num_classes=11 #명령어 11+ silence, 아직 silence없어서 11개.

keepprob=0.7
learning_rate=0.00005

ksize=3

mfcc_n=26
img_size=8

n1 = 1
n2 = 1
n3 = 1
n4 = 5
#timestep=20

num_hidden=128

cnn_weights = {
    'w1': tf.Variable(tf.random_normal([ksize, ksize, 1, n1],  stddev=0.01)),  # 필터사이즈,필터사이즈,필터개수,필터개수
    'w2': tf.Variable(tf.random_normal([ksize, ksize, n1, n2], stddev=0.01)),
    'w3': tf.Variable(tf.random_normal([ksize, ksize, n2, n3], stddev=0.01)),
    'w4': tf.Variable(tf.random_normal([ksize, ksize, n3, n4], stddev=0.01)),
    'w5': tf.Variable(tf.random_normal([num_hidden,num_classes],stddev=0.1)),

    'dw1': tf.Variable(tf.random_normal([65,num_hidden], stddev=0.01)),
    'dw2': tf.Variable(tf.random_normal([num_hidden,num_hidden], stddev=0.01)),
    'dw3' : tf.Variable(tf.random_normal([num_hidden,num_hidden], stddev=0.01)),
}
cnn_biases = {
    'b1': tf.Variable(tf.random_normal([n1], stddev=0.01)),
    'b2': tf.Variable(tf.random_normal([n2], stddev=0.01)),
    'b3': tf.Variable(tf.random_normal([n3], stddev=0.01)),
    'b4': tf.Variable(tf.random_normal([n4], stddev=0.01)),
    'b5': tf.Variable(tf.random_normal([num_classes], stddev=0.01)),

    'db1': tf.Variable(tf.random_normal([num_hidden], stddev=0.01)),
    'db2': tf.Variable(tf.random_normal([num_hidden], stddev=0.01)),
    'db3': tf.Variable(tf.random_normal([num_hidden], stddev=0.01)),

}
'''
rnn_weights={
    'out': tf.Variable(tf.random_normal([num_hidden, n_class])),
}
rnn_biases={
    'out': tf.Variable(tf.random_normal([n_class])),
}
'''
X = tf.placeholder(tf.float32, [None, mfcc_n, img_size, 1])

#X=tf.placeholder(tf.float32,[None,timestep,n_input])

Y=tf.placeholder(tf.float32,[None,num_classes])
keep_prob=tf.placeholder(tf.float32)

conv_out=core.conv_net(X,cnn_weights,cnn_biases,keepprob=keep_prob,mfcc_n=mfcc_n,img_size=img_size)

#logits=core.recur_net(conv_out,rnn_weights,rnn_biases)
pred=conv_out
print(conv_out)
prediction=tf.nn.softmax(pred)
cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=Y)

cost=tf.reduce_mean(cross_entropy)


train_op=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



correct_pred=tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()
print('function ready')

n_iters=5

parameters = cnn_weights.copy()
parameters.update(cnn_biases)
#parameters.update(rnn_weights)
#parameters.update(rnn_biases)

#saver = tf.train.Saver({name:variable for name,variable in cnn_weights.items()+cnn_biases.items()+rnn_weights.items()+rnn_biases.items()})

with tf.Session() as sess:
    sess.run(init)

#세션 시작
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#
print('start training...')
step=0
step_train=1400
iter_=batch.mfccgen(train_dir,step_train)

while True:
    step=step+1
    try:
        x_train,answer=iter_.__next__()
    except StopIteration:
        pass
    #print(x_train)
    x_train=x_train.reshape(1,mfcc_n,8,1)
    answer = np.reshape(answer, (1, 11)) # answer는 라벨 사이즈.
    answer = np.append(answer,answer,axis=0)
    answer = np.append(answer, answer, axis=0)
    answer = np.append(answer, answer, axis=0)


    sess.run(train_op, feed_dict={X:x_train,Y:answer,keep_prob:0.7})
    if step%100==0:
        print(step,sess.run(cost,feed_dict={X:x_train,Y:answer,keep_prob:1}))


    if step>1500:
        break

print("finish training")
step_test=24
iter__=batch.mfccgen(test_dir,step_test)

is_correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

count=0
step=0

while True:

    step=step+1
    try:
        x_test,answer=iter__.__next__()
    except StopIteration:
        pass

    x_test = x_test.reshape(1, mfcc_n, 8, 1)
    answer = np.reshape(answer, (1, 11)) # answer는 라벨 사이즈.
    answer = np.append(answer,answer,axis=0)
    answer = np.append(answer, answer, axis=0)
    answer = np.append(answer, answer, axis=0)

    print(answer)
    print(sess.run(prediction, feed_dict={X: x_test, Y: answer, keep_prob: 1}))
    print('판정:', sess.run(is_correct, feed_dict={X: x_test, Y: answer, keep_prob: 1}))
    if sess.run(is_correct, feed_dict={X: x_test, Y: answer, keep_prob: 1})[0]==True:
        count=count+1
    if step % 24 == 0:
        print(step)
        break

print(count)
