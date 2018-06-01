---
layout:     post
title:      "TensorFlow 实现线性回归"
subtitle:   "TensorFlow for Regression"
date:       2018-06-01 16:00:00
author:     "Luo Yifu"
header-img: "img/post-bg-2015.jpg"
tags:
    - tensorflow
    - deeplearning
---

# TensorFlow实现线性回归分析
线性回归是机器学习中的一个基础问题。

问题: 希望能够找到一个城市中纵火案和盗窃案之间的关系，纵火案的数量是X，盗窃案的数量是Y，我们建设存在如下线性关系，Y = wX + b。

首先定义输入X和目标Y的占位符(placeholder)
```
X = tf.placeholder(tf.float32, shape=[], name='input')
Y = tf.placeholder(tf.float32, shape=[], name='label')
```
这里shape=[]表示标量(scalar)
然后定义需要更新和学习的参数w和b

```
w = tf.get_variable(
'weight', shape=[], initializer=tf.truncated_normal_initializer())
b = tf.get_variable('bias', shape=[], initializer=tf.zeros_initializer())
```

<br>参考资料：
[TensorFlow实现线性回归](https://zhuanlan.zhihu.com/p/28924642)

[TensorFlow结构化模型](https://zhuanlan.zhihu.com/p/29598122)