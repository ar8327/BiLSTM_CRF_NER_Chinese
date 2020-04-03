import model
import tensorflow as tf
import config
import util
import time
import numpy as np
import os

C = config.Config()
model = model.Model()
labels = util.labelGenerator()
dataLoader = util.dataLoader(labels.label2id)
compute_graph = tf.get_default_graph()

X_test = np.load(os.path.join(C.numpy_serialized_path, C.x_test_serialized))
y_test = np.load(os.path.join(C.numpy_serialized_path, C.y_test_serialized))

batchLoader = util.batchGenerator(X_test,y_test,barchsize=1)
batchIter = iter(batchLoader) #batchLoader is an endless iterator providing X,Y

with compute_graph.as_default():
    model.lstm_crf_predict() #Define graph
    model.load_word2vec() #Initialize all variables
    model.restore()
    trainHelper = util.trainHelper(model.word2vector) #Train helper do the padding
    for x,y in batchLoader:
        x_raw = x.copy()
        x,y,sequence_length,seq_max_len = trainHelper.generateBatch(x,y)
        viterbi_sequence = model.predict(x,sequence_length,seq_max_len)
        print(x_raw[0])
        print(labels.ID2label(viterbi_sequence[0][0]))
