# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 02:11:36 2019

@author: NaderBrave
"""


import os
import re
import string
import requests
import numpy as np
import collections
import random
import pickle
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

ops.reset_default_graph()

# Start a session
sess = tf.Session()

# Set RNN Parameters
min_word_frequency = 5  # Trim the less frequent words off
rnn_size = 128  # RNN Model size
epochs = 200  # Number of epochs to cycle through data
batch_size = 100# Train on this many examples at once
learning_rate = 0.001  # Learning rate
training_seq_len = 50  # how long of a word group to consider
embedding_size = rnn_size  # Word embedding size
save_every = 500  # How often to save model checkpoints
eval_every = 50  # How often to evaluate the test sentences
max_sequence_length = 1

punctuation = string.punctuation
punctuation = ''.join([x for x in punctuation if x not in ['-', "'"]])

data = 'book.txt'
f = open(data, 'r')
x_data = f.read()
x_data1 = x_data.replace('\r\n', '')
x_data1 = x_data1.replace('\n', '')

# Clean text
print('Cleaning Text')
x_data1 = re.sub(r'[{}]'.format(punctuation), ' ', x_data1)
x_data1 = re.sub('\s+', ' ', x_data1).strip().lower()
word = []
length = len(x_data1)

def build_vocab(text, min_freq):
    word_counts = collections.Counter(text)
    # limit word counts to those more frequent than cutoff
    word_counts = {key: val for key, val in word_counts.items() if val > min_freq}
    # Create vocab --> index mapping
    words = word_counts.keys()
    vocab_to_ix_dict = {key: (i_x + 1) for i_x, key in enumerate(words)}
    # Add unknown key --> 0 index
    vocab_to_ix_dict['unknown'] = 0
    # Create index --> vocab mapping
    ix_to_vocab_dict = {val: key for key, val in vocab_to_ix_dict.items()}

    return ix_to_vocab_dict, vocab_to_ix_dict



ix2vocab, vocab2ix = build_vocab(x_data1, min_word_frequency)
vocab_size = len(ix2vocab) + 1
print('Vocabulary Length = {}'.format(vocab_size))
# Sanity Check
assert (len(ix2vocab) == len(vocab2ix))


text_processed = []
for ix, x in enumerate(x_data1):
    try:
        text_processed.append(vocab2ix[x])
    except KeyError:
        text_processed.append(0)
text_processed = np.array(text_processed)


data_length = len(text_processed)
#data_length = 400000

count = int(length / 40)
data = np.zeros((int(data_length / 5), 40))
data_label = np.zeros((int(data_length / 5), 1))
# word = np.zeros((40,count))

c = 0
i = 0
while i + 40 < data_length:
    data[c, :] = text_processed[i:i + 40].transpose()
    data_label[c] = text_processed[i + 40:i + 41].transpose()
    i = i + 5
    c = c + 1

x_data = tf.placeholder(tf.int32, [None, 40])
y_output = tf.placeholder(tf.int32, [None])

embedding_mat = tf.Variable(tf.random_uniform([vocab_size, 10], -1.0, 1.0))
identity_mat = tf.diag(tf.ones(shape=[vocab_size]))
embedding_output = tf.nn.embedding_lookup(embedding_mat, x_data)

# Define the RNN cell
# tensorflow change >= 1.0, rnn is put into tensorflow.contrib directory. Prior version not test.
if tf.__version__[0] >= '1':
    cell = tf.contrib.rnn.GRUCell(num_units=rnn_size)
else:
    cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_size)

output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32)
# output = tf.nn.dropout(output, dropout_keep_prob)

# Get output of RNN sequence
output = tf.transpose(output, [1, 0, 2])
last = tf.gather(output, int(output.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([rnn_size, vocab_size-1], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[vocab_size-1]))
logits_out = tf.matmul(last, weight) + bias

# Loss function
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=y_output)
loss = tf.reduce_mean(losses)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out, 1), tf.cast(y_output, tf.int64)), tf.float32))

optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)
length = len(data_label)
lo = []
lo1 = []
for epoch in range(epochs):

    # Shuffle training data
    shuffled_ix = np.random.permutation(np.arange(len(data)))
    x_train = data[shuffled_ix]
    y_train = data_label[shuffled_ix]
    num_batches = int(len(x_train)/batch_size) + 1
    # TO DO CALCULATE GENERATIONS ExACTLY
    for i in range(num_batches):
        # Select train data
        min_ix = i * batch_size
        max_ix = np.min([len(x_train), ((i+1) * batch_size)])
        x_train_batch = x_train[min_ix:max_ix]
        y_train_batch = y_train[min_ix:max_ix]
        y_train_batch = np.squeeze(y_train_batch)
        # Run train step
        train_dict = {x_data: x_train_batch, y_output: y_train_batch}
        sess.run(train_step, feed_dict=train_dict)
        
        temp = sess.run(loss,feed_dict=train_dict)
        lo1.append(temp)
        temp1 = sess.run(logits_out,feed_dict=train_dict)
        temp1 = np.argmax(temp1,axis=1)
        lo.append(temp1)
        #print('loss = ',temp1)
        print("\n epoch = %i , batch_number = %i , loss = %g "%(epoch,i, temp))
    
    # Run loss and accuracy for training


print('\n')
data_test = data[0, :]

text = []
for ix, x in enumerate(data_test):
    try:
        text.append(ix2vocab[x])
    except KeyError:
        text.append(0)

print('test sequence:\n')
print(text)


data_test = np.reshape(data_test, [1, 40])
data_l = data_label[0]
temp = np.array((40, 1))
temp = np.reshape(data_test, [40, 1])
script = []
out=np.zeros((400,1))

for i in range(400):
    x = np.reshape(temp, [1, 40])
    train_dict = {x_data: x, y_output: data_l}
    e = sess.run(logits_out, feed_dict=train_dict)
    out[i] = np.argmax(e, axis=1)
    temp = np.roll(temp, -1)
    temp[39] = out[i]
    script.append(out[i])

script = np.array(script)
script_f = []
script = np.squeeze(script)

sentence = ''

for m in range(1, 400):
    d = script[m]
    # d=np.squeeze(d)
    d1 = ix2vocab[d]

    sentence = sentence + d1
print('\n generated script:\n')
print(sentence)

"""
s = script[10]
s2 = ix2vocab[s]    
"""





