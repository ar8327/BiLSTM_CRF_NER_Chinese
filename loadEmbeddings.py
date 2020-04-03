import tensorflow as tf
import gensim
import config
C = config.Config()

word2vector = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format(C.PATH_EMBEDDINGS).wv
embeddings = word2vector.vectors

W = tf.Variable(tf.constant(0.0, shape=[C.EMBEDDING_VOCAL_SIZE, C.EMBEDDING_DIM]),
                trainable=False, name="W")

embedding_placeholder = tf.placeholder(tf.float32, [C.EMBEDDING_VOCAL_SIZE, C.EMBEDDING_DIM],name="embedding_placeholder")
embedding_init = W.assign(embedding_placeholder)

# ...
sess = tf.Session()

sess.run(embedding_init, feed_dict={embedding_placeholder: embeddings})

tf.nn.embedding_lookup()