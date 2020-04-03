## BLSTM_CRF

BLSTM_CRF for sequential classification

### Model description:
+ Input:
  - 新中国今天成立了
+ Output:
  - 1 : [新 , 中国 , 今天 , 成立 , 了]
  - 2 : [O,LOCATION,TIME,O,O]
  + Where 1 is the word segmentation result and 2 is classification result of each word in 1

### Usage:
+ Download pre-trained word-embeddings for classification task : 
  - http://ar8327k.top/bin/sgns.renmin.bigram-char.bz2
+ Extract the embedding under the root directory of this project
+ Run `train.py`
+ After trainning , use `predict.py` to perform evaluation or use `predict_RT.py` to perform real-time prediction
### Details (In Chinese)
+ 本模型使用了第三方开源的预训练word-embedding向量，表示感谢！
+ 本模型使用了第三方开源的标注语料（2k条，在`data/origindata.txt`下），经过`format_transfer.py`转化为本程序读取的格式，表示感谢！
+ 若使用其他训练语料（如，增加要分类的类别等），需要更改`config.py`中的设置！
+ 预计会在之后会有另一个版本，为了深究原理手工实现CRF层，其他不变！
