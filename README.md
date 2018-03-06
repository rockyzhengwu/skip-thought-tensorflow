# skip-thought-tensorflow
Skip-Thought Vectors implement by tensorflow


代码从tensorflow 官网copy来
做了如下改变:

1. 修改部分代码,为了能在中文场景下跑起来
2. 适用Python3.x 的代码 官方的代码在Python3.x下存在部分问题
3. 去掉bazel的编译,直接用shell 主要对bazel不熟，加上国内环境的限制不用

## 准备训练数据
 训练数据格式:一行一个句子，每个段落用```\n```分割
中文诗歌例子见data/train.txt

有了训练数据,先转换成tfrecord格式，具体方式如下
```bash
cd skip_thoughts/data/
./pre_data.sh
```

会产生在data/tfrecord 文件夹下产生 vocab.txt(字映射成id) 和word_counts.txt(词频文件) 和很多tfrecord　文件


## train

训练

````bash

cd skip_thoughts
./train.sh

````

## expansion

模型基于encoder-decoder的方式，vocab代码中做了限制是20000。我理解主要是在decoder的时候不能有太大的vocab,word2vec用了层次softmax和负采样来解决的
为了有很多未登录词作者用了线性映射的方法：在word2vec和skip-thoughts的词向量直接做了个线性回归，把word2vec的结果映射到skip-thoughts


我在古诗上做的基于字的训练，字相对很少，所以暂时不需要,所以```vocab_expansion.sh```中的word2vec被我去掉了，只是单纯的产生了词向量存储出来

## 句子向量

利用模型的encoder部分就可以得到任意句子的向量
```bash

```
