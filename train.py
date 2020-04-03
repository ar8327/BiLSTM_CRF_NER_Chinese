import model
import tensorflow as tf
import config
import util
import time
import os
import numpy as np
from sklearn.model_selection import train_test_split

C = config.Config()
model = model.Model()

if os.path.exists(C.labelMapping):
    labels = util.labelGenerator()
else:
    labels = util.labelGenerator(C.labelList, C.labelPrefix)
    labels.serialize()  # Save labels

dataLoader = util.dataLoader(labels.label2id)
compute_graph = tf.get_default_graph()

#Load train/test split if exists , else generate
if os.path.exists(os.path.join(C.numpy_serialized_path,C.x_train_serialized)):
    X_train = np.load(os.path.join(C.numpy_serialized_path,C.x_train_serialized),allow_pickle=True)
    y_train = np.load(os.path.join(C.numpy_serialized_path,C.y_train_serialized),allow_pickle=True)
    X_test = np.load(os.path.join(C.numpy_serialized_path,C.x_test_serialized),allow_pickle=True)
    y_test = np.load(os.path.join(C.numpy_serialized_path,C.y_test_serialized),allow_pickle=True)
else:
    X,Y = dataLoader.getAll()
    X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.1)
    np.save(os.path.join(C.numpy_serialized_path,C.x_train_serialized),X_train)
    np.save(os.path.join(C.numpy_serialized_path,C.y_train_serialized),y_train)
    np.save(os.path.join(C.numpy_serialized_path,C.x_test_serialized),X_test)
    np.save(os.path.join(C.numpy_serialized_path,C.y_test_serialized),y_test)


batchLoader = util.batchGenerator(X_train,y_train)
batchIter = iter(batchLoader) #batchLoader is an endless iterator providing X,Y

with compute_graph.as_default():
    model.lstm_crf() #Define graph
    model.load_word2vec() #Initialize all variables
    try:
        model.restore()
        print(C.info.format(str(time.time()), "Train", "Load pretrained weight."))
    except:
        print(C.info.format(str(time.time()), "Train", "Failed to load pretrained weight."))
    trainHelper = util.trainHelper(model.word2vector) #Train helper do the padding
    epochCount = 0
    avgloss = 0
    for x,y in batchLoader:
        x_raw = x.copy()
        x,y,sequence_length,seq_max_len = trainHelper.generateBatch(x,y)
        loss,viterbi_sequence = model.train(x,y,sequence_length,seq_max_len)
        avgloss += loss
        epochCount += 1
        if epochCount % C.train_report_step == 0:
            epochCount = 0
            print(C.info.format(str(time.time()), "Train", "Avarage loss = "+str(avgloss/C.train_report_step)))
            avgloss = 0
            model.save()
            # print("*******")
            # for i in range(x_raw.shape[0]):
            #     print(x_raw[i])
            #     print(labels.ID2label(viterbi_sequence[i]))
            # print("*******")
