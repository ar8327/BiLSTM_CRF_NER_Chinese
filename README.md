## BLSTM_CRF

BLSTM_CRF for sequential classification

###Model description:
+ Input:
  - 新中国今天成立了
+ Output:
  - 1 : [新 , 中国 , 今天 , 成立 , 了]
  - 2 : [O,LOCATION,TIME,O,O]
  + Where 1 is the word segmentation result and 2 is classification result of each word in 1

###Usage:
+ Download pre-trained word-embeddings for classification task : 
  - http://ar8327k.top/bin/sgns.renmin.bigram-char.bz2
+ Extract the embedding under the root directory of this project
+ Run `train.py`
+ After trainning , use `predict.py` to perform evaluation or use `predict_RT.py` to perform real-time prediction