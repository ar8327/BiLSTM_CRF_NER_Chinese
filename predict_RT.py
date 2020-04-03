import model
import tensorflow as tf
import config
import util
import time
import numpy as np
import os
import jieba
import config

C = config.Config()
jieba.load_userdict('data/coal_dict.txt')

model = model.Model()
labels = util.labelGenerator()
compute_graph = tf.get_default_graph()

with compute_graph.as_default():
    model.lstm_crf_predict() #Define graph
    model.load_word2vec() #Initialize all variables
    model.restore()
    trainHelper = util.trainHelper(model.word2vector) #Train helper do the padding
    # for x,y in batchLoader:
    #     x_raw = x.copy()
    #     x,y,sequence_length,seq_max_len = trainHelper.generateBatch(x,y)
    #     viterbi_sequence = model.predict(x,sequence_length,seq_max_len)
    #     print(x_raw[0])
    #     print(labels.ID2label(viterbi_sequence[0][0]))
    while True:
        seg_X = input("Input your sentence :").replace(' ', '').replace('\n', '')
        x = list(jieba.cut(seg_X))
        x_raw = x.copy()
        x, sequence_length, seq_max_len = trainHelper.generateData4Predict(x)
        x = np.reshape(x,(1,x.shape[0]))
        viterbi_sequence = model.predict(x, sequence_length, seq_max_len)
        print("Raw sentence after segmentation -> ")
        print(x_raw)
        print("Model output ->")
        print(labels.ID2label(viterbi_sequence[0][0]))
