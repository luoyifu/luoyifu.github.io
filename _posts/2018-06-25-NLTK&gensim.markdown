---
layout:     post
title:      "自然语言处理与Gensim包"
subtitle:   "python 自然语言处理与LDA的gensim实现"
date:       2018-06-25 8:00:00
author:     "Luo Yifu"
header-img: "img/post-bg-2015.jpg"
tags:
    - NLTK
    - gensim
    - LDA
---
# 自然语言处理与gensim包



## Gensim包简介

Gensim是python开源工具包，用于处理非结构化文本，学习隐藏的主题表达。支持主题算法包括TF-IDF, LSA, LDA, word2vec。

### Gensim的基本概念

* 语料（corpus）：一组原始文本的集合。
* 模型（model）：定义了两个想来那个空间的变换（从文本的一种向量表示变换成另一种）



## Gensim包处理自然语言语料库

#### Step1 语料预处理

预处理过程是将原始语料转换成gensim能够处理的稀疏向量。原始语料仓库是一些文档集合，每个文档是一些字符集合。**我们先对原始文档进行分词、去除停用词等操作，得到每一篇文档的特征列表。**

```python
# 导入gensim包
from gensim import corpora, models

# tests文件是文档的list，每行是分词后的文档
# 例如：tests = [['发作性','咳嗽'...],['反复','咳痰'...],[...]]

# 建立语料特征索引词典，并将文本特征的原始表达转化成词袋模型对应的稀疏向量表达。
dictionary = corpora.Dictionary(texts)

# 将文本的原始表达转换成词袋模型的稀疏向量表达。
# corpus的每一个元素对应一篇文档
corpus = []
for text in texts:
    corpus.append(dictionary.doc2bow(text))
```

至此，语料预处理工作完成。得到了语料仓库中每一篇文档对应的稀疏向量，向量的每个元素代表一个“词”在这篇文档中出现的次数。



##### 文档的流式处理

为了优化内存，可以对文档进行流式处理，即每次迭代返回一个稀疏向量。

```python
class MyCorpus(object):
	def __iter__(self):
		for line in open('mycorpus.txt'):
			yield dictionary.doc2bow(line.lower().split())
```



### 主题向量的变换

对文本向量的变换是gensim的核心，**每个变换对应一个主题模型**，例如词袋模型`doc2bow`变换。每个模型时一个标准的python对象。

