import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #设置显卡


class Config():
    def __init__(self):
        self.EMB_WORDCOUNT = 356063
        self.PATH_EMBEDDINGS = 'sgns.renmin.bigram-char'
        self.EMBEDDING_DIM = 300
        self.EMBEDDING_VOCAL_SIZE = 356053
        self.num_units = 128 #Hidden unit in LSTM
        self.batch_size = 16
        self.nullid = self.EMBEDDING_VOCAL_SIZE+1
        self.UNK_char_id = self.EMBEDDING_VOCAL_SIZE
        self.labelList = ['product_name','time','org_name','person_name','location','company_name','O']
        self.labelPrefix = ['B','E','M']
        self.labelBG = 'O'
        self.traindata = 'data-example.txt'
        self.data_to_label = '/home/ar8327/PycharmProjects/BLSTM_CRF_NER/data/unlabeled.txt'
        self.num_class = (len(self.labelList)-1)*len(self.labelPrefix)+2 #Class count
        self.empty_class = self.num_class-1
        self.info = "At {0} from module {1} : {2}"
        self.train_report_step = 100
        self.modelpath = 'models/BLSTM_NER.ckpt'
        self.labelMapping = 'labels.json'
        self.numpy_serialized_path = 'npd/'
        self.x_train_serialized = 'x_train.npy'
        self.y_train_serialized = 'y_train.npy'
        self.x_test_serialized = 'x_test.npy'
        self.y_test_serialized = 'y_test.npy'