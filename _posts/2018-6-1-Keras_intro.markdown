---
layout:     post
title:      "Keras入门"
subtitle:   ""
date:       2018-06-01 12:00:00
author:     "Luo Yifu"
header-img: "img/post-bg-2015.jpg"
tags:
    - Keras
    - deeplearning
---

# Keras 入门
## Keras是什么

tensorflow已经是一个封装好的框架，但是我们发现写一个简单的神经网络也需要很多行才能够写完，这个时候，就有很多的第三方插架来帮助你写网络，也就是说你用tensorflow要写10行，第三方插架帮你封装了一个函数，就是把这10行集合在这个函数里面，那么你用1行，传入相同的参数，就能够达到10行相同的效果，如此简便并且节约时间，可以帮助我们很快的实现我们的想法。
<br>Keras是一个兼容Theano和TensorFlow的高级包，用它来组建一个神经网络只需要几条语句就搞定。

**Keras中有两类模型：Sequential 顺序模型 和 使用函数式 API 的 Model 类模型**。

## 顺序模型Sequential
Keras 的核心数据结构是`model`，一种组织网络层的方式。最简单的模型是`Sequential`顺序模型，它是由多个网络层线性堆叠的栈。

引入sequential，这个就是一个空的网络结构，并且这个结构是一个顺序的序列，所以叫Sequential，Keras里面还有一些其他的网络结构。

### 1. 构建sequential模型

你可以通过将层的列表传递给`Sequential`的构造函数，来创建一个`Sequential`模型：

```
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
```
或者一层一层添加：
```
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
```



### 2. 指定输入尺寸
模型需要知道它所期望的输入的尺寸。出于这个原因，顺序模型中的第一层（只有第一层，因为下面的层可以自动地推断尺寸）需要接收关于其输入尺寸的信息。有几种方法来做到这一点：
* 传递一个`input_shape`参数给第一层。它是一个表示尺寸的元组 (一个整数或`None`的元组，其中`None`表示可能为任何正整数)。在`input_shape`中不包含数据的`batch`大小。
* 某些 2D 层，例如`Dense`，支持通过参数`input_dim`指定输入尺寸，某些 3D 时序层支持`input_dim`和`input_length`参数。
* 如果你需要为你的输入指定一个固定的`batch`大小（这对 stateful RNNs 很有用），你可以传递一个`batch_size`参数给一个层。如果你同时将`batch_size=32`和`input_shape=(6, 8)`传递给一个层，那么每一批输入的尺寸就为 (32，6，8)。

下面两种输入方式等价：
```
model = Sequential()
model.add(Dense(32, input_shape=(784,)))
```
```
model = Sequential()
model.add(Dense(32, input_dim=784))
```

### 3. 编译
在训练模型之前，需要配置学习过程，这是通过`compile`方法完成的。它接收三个参数：
* 优化器`optimizer`。它可以是现有优化器的字符串标识符，如`rmsprop`或`adagrad`，也可以是`Optimizer`类的实例。详见：[optimizers](https://keras.io/zh/optimizers/)。
* 损失函数`loss`，模型试图最小化的目标函数。它可以是现有损失函数的字符串标识符，如`categorical_crossentropy`或 `mse`，也可以是一个目标函数。详见：[losses](https://keras.io/zh/losses/)。
* 评估标准`metrics`。对于任何分类问题，你都希望将其设置为`metrics = ['accuracy']`。评估标准可以是现有的标准的字符串标识符，也可以是自定义的评估标准函数

`compile`方法：
```
compile(self, optimizer, loss, metrics=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)

# metrics: 在训练和测试期间的模型评估标准。通常你会使用 metrics = ['accuracy']。 要为多输出模型的不同输出指定不同的评估标准，还可以传递一个字典，如 metrics = {'output_a'：'accuracy'}。
# sample_weight_mode: 如果你需要执行按时间步采样权重（2D权重），请将其设置为 temporal。 默认为 None，为采样权重（1D）。如果模型有多个输出，则可以通过传递 mode 的字典或列表，以在每个输出上使用不同的  sample_weight_mode。
# weighted_metrics: 在训练和测试期间，由 sample_weight 或 class_weight 评估和加权的度量标准列表。
# target_tensors: 默认情况下，Keras 将为模型的目标创建一个占位符，在训练过程中将使用目标数据。相反，如果你想使用自己的目标张量（反过来说，Keras 在训练期间不会载入这些目标张量的外部 Numpy 数据），可以通过 target_tensors 参数指定它们。它应该是单个张量（对于单输出 Sequential 模型）。
# **kwargs: 当使用 Theano/CNTK 后端时，这些参数被传入 K.function。当使用 TensorFlow 后端时，这些参数被传递到  tf.Session.run。

```

### 4. 训练
Keras 模型在输入数据和标签的 Numpy 矩阵上进行训练。为了训练一个模型，你通常会使用 fit 函数。
`fit`:以固定数量的轮次（数据集上的迭代）训练模型。
```
fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
```

[Keras中文文档](https://keras.io/zh/)