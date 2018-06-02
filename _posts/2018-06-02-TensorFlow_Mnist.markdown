---
layout:     post
title:      "TensorFlow Mnist"
subtitle:   "TensorFlow 的一个简单应用"
date:       2018-06-02 10:00:00
author:     "Luo Yifu"
header-img: "img/post-bg-2015.jpg"
tags:
    - tensorflow
    - deeplearning
---

# TensorFlow Mnist

导入数据（连接不上google非常麻烦）
```
mnist = input_data.read_data_sets('data/', one_hot=True)
# input_data.py这个文件在官网上无法下载，见我的附录
```

**注意：**在`input_data.py`文件最后一行：
```
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
```
这个函数可以顺着路径找到，更改其中的`DEFAULT_SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'`（官方的default source url还是连接不上）


首先定义数据结构：
输入数据暂时不知道是什么，我们用占位符placeholder，权重w和偏置量b都是需要拟合的，用变量varialbe来定义：
```
# 希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784]
x = tf.placeholder("float", [None, 784])
# 用全为零的张量来初始化W和b。因为我们要学习W和b的值，它们的初值可以随意设置。
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
```

使用softmax来实现模型。这里的x是输入的图片，y是通过softmax回归后的输出。我们需要学习w和b，使得输出y和实际输出（就是实际上是数字1，2，3……这些）尽量相同。所以，我们还需要定义一个实际值，也用placeholder定义。
```
# 用tf.matmul(​​X，W)表示x乘以W
y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder("float", [None,10])
```

接下来，定义代价函数，使用交叉熵代价函数`cross_entropy`：
```
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
```
> 首先，用 tf.log 计算 y 的每个元素的对数。接下来，我们把 y_ 的每一个元素和 tf.log(y_) 的对应元素相乘。最后，用 tf.reduce_sum 计算张量的所有元素的总和。（注意，这里的交叉熵不仅仅用来衡量单一的一对预测和真实值，而是所有100幅图片的交叉熵的总和。对于100个数据点的预测表现比单一数据点的表现能更好地描述我们的模型的性能。

下面，使用反向传播算法优化，使得代价函数最小
```
# 用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
```

至此，模型已经设计完毕。下面，初始化并部署模型：
```
# 初始化变量
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 这里我们让模型循环训练1000次
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

模型评估：
如何评估模型？首先要找到预测正确的那些图片
> tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签，比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，而 tf.argmax(y_,1) 代表正确的标签，我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。

定义预测正确函数：
```
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # 这行代码会给我们一组布尔值。
# 为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。
# 例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
```

最后，我们计算所学习到的模型在测试数据集上面的正确率。
```
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
```


详情参见：[官方文档](http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html)

## 附录：

input_data.py
```
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
```
