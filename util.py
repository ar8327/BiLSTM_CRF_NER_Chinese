import config
import numpy as np
import time
import json


C = config.Config()

class dataLoader():
    def __init__(self,label2id):
        self.dataFile = open(C.traindata,'r')
        self.label2id = label2id

    def getAll(self):
        self.X = []
        self.Y = []
        self.tempX = []
        self.tempY = []

        for line in self.dataFile.readlines():
            if line != "\n":
                line = line.split()
                if len(line) == 1:
                    self.tempX = []
                    self.tempY = []
                    continue
                self.tempX.append(line[0])
                self.tempY.append(self.label2id[line[1]])
            else:
                if self.tempX != []:
                    self.X.append(self.tempX)
                    self.Y.append(np.array(self.tempY))
                    self.tempX = []
                    self.tempY = []

        if self.tempX != []:
            self.X.append(self.tempX)
            self.Y.append(np.array(self.tempY))
        print(C.info.format(str(time.time()), "dataLoader", "Load data from "+C.traindata))
        return np.array(self.X),np.array(self.Y)

class labelGenerator():
    def __init__(self,labels=None,prefix=None):
        if labels == None and prefix == None:
            self.labels_saved = json.load(open(C.labelMapping, 'r'))
            self.label2id = self.labels_saved['label2id']
            self.id2label = self.labels_saved['id2label']
            return
        else:
            self.labels = labels
            self.label2id = dict()
            self.prefix = prefix
            self.id = 0
            self.id2label = []
            for label in self.labels:
                truelabel = label
                for p in prefix:
                    if truelabel != 'O':
                        label = p+'-'+truelabel
                    else:
                        label = truelabel
                    if self.label2id.get(label,-1) == -1:
                        self.label2id.update({label:self.id})
                        self.id2label.append(label)
                        self.id += 1
                    else:
                        continue
            return

    def serialize(self):
        self.labels_to_save = dict()
        self.labels_to_save.update({'label2id':self.label2id})
        self.labels_to_save.update({'id2label':self.id2label})
        json.dump(self.labels_to_save,open(C.labelMapping,'w'))

    def ID2label(self,pred):
        result = []
        for i in range(pred.shape[0]):
            if pred[i] != C.empty_class:
                result.append(self.id2label[pred[i]])
            else:
                result.append('EMPTY')
        return result

class trainHelper():
    def __init__(self,word2vec_keyed_vectors):
        self.word2vec = word2vec_keyed_vectors

    def word2id(self,word):
        return self.word2vec.vocab[word].index

    def id2word(self,id):
        return self.word2vec.index2word[id]

    def id2sentence(self,sentence):
        result = []
        for wordid in sentence:
            if wordid != C.nullid:
                result.append(self.id2word(wordid))

    def padlabel(self,label,pad_length):
        padding = np.array([C.empty_class]*pad_length)
        return np.concatenate((label,padding))

    def generateData4Predict(self,sentence):
        maxL = len(sentence)
        return self.sentence2id(sentence,maxL),np.array([maxL]),maxL

    def generateBatch(self,sentences,labels):
        #We maintain a table for recording real length of each sentence
        real_len = []

        #Get max length of this batch Notice that the result is dense.
        maxL = -1
        for sentence in sentences:
            real_len.append(len(sentence))
            if len(sentence) > maxL:
                maxL = len(sentence)
        result = []
        result_y = []
        for sentence in sentences:
            result.append(self.sentence2id(sentence,maxL)) #Sentence is padded
        for label in labels:
            pad_length = abs(label.shape[0]-maxL)
            result_y.append(self.padlabel(label,pad_length))
        return np.array(result),np.array(result_y),np.array(real_len),maxL

    def sentence2id(self,sentence,maxL):
        result = []
        count = 0
        for word in sentence:
            try:
                result.append((self.word2id(word)))
            except:
                result.append(C.UNK_char_id)
            count += 1
        #Do padding
        for i in range(maxL-count):
            result.append(C.nullid)
        return np.array(result)

class batchGenerator:
    def __init__(self,X,Y,barchsize=32,shuffle=True):
        if shuffle:
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            self.X = X[indices]
            self.Y = Y[indices]
        else:
            self.X = X
            self.Y = Y
        self.batchsize = barchsize
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current + self.batchsize <= self.X.shape[0]:
            r = self.X[self.current:self.current+self.batchsize]
            l = self.Y[self.current:self.current+self.batchsize]
        else:
            print(C.info.format(str(time.time()),"batchGenerator","Hit next epoch"))
            self.current = 0
            r = self.X[self.current:self.current+self.batchsize]
            l = self.Y[self.current:self.current+self.batchsize]
        self.current += self.batchsize
        return r,l
