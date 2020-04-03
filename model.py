import tensorflow as tf
import config
import gensim
import numpy as np
C = config.Config()
class Model():
    def __init__(self):
        self.graph = tf.get_default_graph()
        self.sess = tf.Session()

    def lstm_crf(self):
        with self.graph.as_default():
            self.seq_max_len = tf.placeholder(tf.int32)
            self.X = tf.placeholder(tf.int32,shape=[None,None]) # shape = (batch_size,max_sentence_length)
            self.Y = tf.placeholder(tf.int32,shape=[None,None]) #shape = (batch_size,max_result_sentence_length)
            self.embedding_placeholder = tf.placeholder(tf.float32, [C.EMBEDDING_VOCAL_SIZE+2, C.EMBEDDING_DIM]) #For feeding pretrained weights , +2 for UNK word embedding,padding embedding
            self.embeddings = tf.Variable(tf.constant(0.0, shape=[C.EMBEDDING_VOCAL_SIZE+2, C.EMBEDDING_DIM]),
                            trainable=True, name="W_embeddings") #Real embedding matrix
            self.X_embed = tf.nn.embedding_lookup(self.embeddings,self.X) #shape = (batch_size,max_sentence_length,embed_dim)
            self.sequence_length = tf.placeholder(tf.int32,shape=[None]) #Real seq length for each sentence per batch


            with tf.name_scope("BiLSTM"):
                with tf.variable_scope('forward'):
                    self.lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(C.num_units, forget_bias=1.0, state_is_tuple=True)
                with tf.variable_scope('backward'):
                    self.lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(C.num_units, forget_bias=1.0, state_is_tuple=True)
                (self.output_fw, self.output_bw), self.states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.lstm_fw_cell,
                                                                                 cell_bw=self.lstm_bw_cell,
                                                                                 inputs=self.X_embed,
                                                                                 sequence_length=self.sequence_length,
                                                                                 dtype=tf.float32,
                                                                                 scope="BiLSTM")
            self.outputs = tf.concat([self.output_fw, self.output_bw], axis=2)
            output_flat = tf.reshape(self.outputs,[-1,2*C.num_units])
            #Fully connected layer
            self.W = tf.get_variable('W',shape=[2*C.num_units,C.num_class])
            self.b = tf.get_variable('b',shape=[C.num_class],initializer=tf.zeros_initializer)
            self.pred = tf.matmul(output_flat,self.W)+self.b
            self.scores = tf.reshape(self.pred,[-1,self.seq_max_len,C.num_class])

            # Linear-CRF.
            self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.scores, self.Y, self.sequence_length)

            # Decoder for prediction
            self.viterbi_sequence, self.viterbi_score = tf.contrib.crf.crf_decode(self.scores, self.transition_params,
                                                                        self.sequence_length)

            self.loss = tf.reduce_mean(-self.log_likelihood)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=3e-04)


            self.train_op = self.optimizer.minimize(self.loss)
            self.saver = tf.train.Saver()


    def lstm_crf_predict(self):
        with self.graph.as_default():
            self.seq_max_len = tf.placeholder(tf.int32)
            self.X = tf.placeholder(tf.int32,shape=[None,None]) # shape = (batch_size,max_sentence_length)
            self.Y = tf.placeholder(tf.int32,shape=[None,None]) #shape = (batch_size,max_result_sentence_length)
            self.embedding_placeholder = tf.placeholder(tf.float32, [C.EMBEDDING_VOCAL_SIZE+2, C.EMBEDDING_DIM]) #For feeding pretrained weights , +2 for UNK word embedding,padding embedding
            self.embeddings = tf.Variable(tf.constant(0.0, shape=[C.EMBEDDING_VOCAL_SIZE+2, C.EMBEDDING_DIM]),
                            trainable=False, name="W_embeddings") #Real embedding matrix
            self.X_embed = tf.nn.embedding_lookup(self.embeddings,self.X) #shape = (batch_size,max_sentence_length,embed_dim)
            self.sequence_length = tf.placeholder(tf.int32,shape=[None]) #Real seq length for each sentence per batch


            with tf.name_scope("BiLSTM"):
                with tf.variable_scope('forward'):
                    self.lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(C.num_units, forget_bias=1.0, state_is_tuple=True)
                with tf.variable_scope('backward'):
                    self.lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(C.num_units, forget_bias=1.0, state_is_tuple=True)
                (self.output_fw, self.output_bw), self.states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.lstm_fw_cell,
                                                                                 cell_bw=self.lstm_bw_cell,
                                                                                 inputs=self.X_embed,
                                                                                 sequence_length=self.sequence_length,
                                                                                 dtype=tf.float32,
                                                                                 scope="BiLSTM")
            self.outputs = tf.concat([self.output_fw, self.output_bw], axis=2)
            output_flat = tf.reshape(self.outputs,[-1,2*C.num_units])
            #Fully connected layer
            self.W = tf.get_variable('W',shape=[2*C.num_units,C.num_class])
            self.b = tf.get_variable('b',shape=[C.num_class],initializer=tf.zeros_initializer)
            self.pred = tf.matmul(output_flat,self.W)+self.b
            self.scores = tf.reshape(self.pred,[-1,self.seq_max_len,C.num_class])

            # Linear-CRF.
            self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.scores, self.Y, self.sequence_length)

            # Decoder for prediction
            self.viterbi_sequence, self.viterbi_score = tf.contrib.crf.crf_decode(self.scores, self.transition_params,
                                                                        self.sequence_length)
            self.saver = tf.train.Saver()

    def load_word2vec(self):
        self.sess.run(tf.global_variables_initializer())
        self.word2vector = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format(C.PATH_EMBEDDINGS).wv
        self.pretrained_embeddings = self.word2vector.vectors
        self.vec_unk = np.random.random((2,C.EMBEDDING_DIM))
        self.pretrained_embeddings = np.row_stack((self.pretrained_embeddings,self.vec_unk))
        self.embeddings.load(self.pretrained_embeddings,session=self.sess)

    def train(self,X,Y,sequence_length,seq_max_len):
        tf_viterbi_sequence, _ ,loss= self.sess.run([self.viterbi_sequence, self.train_op,self.loss],
                                             feed_dict={self.X: X,
                                                        self.Y: Y,
                                                        self.sequence_length: sequence_length,
                                                        self.seq_max_len: seq_max_len})
        return loss,tf_viterbi_sequence

    def predict(self,X,sequence_length,seq_max_len):
        tf_viterbi_sequence = self.sess.run([self.viterbi_sequence],
                                             feed_dict={self.X: X,
                                                        self.sequence_length: sequence_length,
                                                        self.seq_max_len: seq_max_len})
        return tf_viterbi_sequence

    def save(self):
        self.saver.save(sess=self.sess, save_path=C.modelpath)

    def restore(self):
        self.saver.restore(sess=self.sess, save_path=C.modelpath)