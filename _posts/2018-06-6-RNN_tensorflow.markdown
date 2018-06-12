---
layout:     post
title:      "TensorFlow 循环神经网络（RNN）"
subtitle:   ""
date:       2018-06-06 10:00:00
author:     "Luo Yifu"
header-img: "img/post-bg-2015.jpg"
tags:
    - tensorflow
    - deeplearning
---
# TensorFlow循环神经网络（RNN）

## RNN

### RNN思想：

无论是 NN 还是 CNN，它们处理的数据都是相对固定的。NN 自不用提，一旦网络结构确定下来之后，输入数据的维度便随之确定了下来；而虽说 CNN 能够通过 RoI Pooling 等手段来接受不同长宽的图片输入，但它们大多只是在最后一步做出了调整、并没有特别根本地解决了问题。而循环神经网络（Recurrent Neural Network，常简称为 RNN）则通过非常巧妙的形式、让模型能用同一套结构非常自然地处理不同的输入数据，这使得 RNN 在处理序列问题（比如各种 NLP
Tasks）时显得得心应手（注：这并非是绝对的结论，只能说从设计的理念上来看确实更为自然；事实上在特定的序列问题上 CNN 能够表现得比 RNN 更好，比如 Facebook FAIR 团队最近弄出的那个 CNN 翻译模型……）

* 与 CNN 类似，RNN 也可以说是 NN 的一种拓展，但从思想上来说，RNN 和 NN、CNN 相比已经有了相当大的不同：
* NN 只是单纯地接受数据，认为不同样本之间是独立的 CNN 注重于挖掘单个结构性样本（比如说图像）相邻区域之间的关联RNN 注重于挖掘样本与样本之间的序关联（这是我瞎掰的一个词 ( σ'ω')σ）

CNN 是通过局部连接和权值共享来做到相邻区域的特征提取的，那么 RNN 是如何做到提取序特征的呢？关键就在于“状态（State）”的引入。换句话说，在 RNN 中，输入不是简单地通过权值矩阵（NN 的做法）或卷积（CNN 的做法）来得到输出，而是要先得出一个 State、然后再由这个 State 得到输出。这样一来，只需让 State 之间能够“通信”，那么当前样本就能够影响到下一个、乃至于下 n 个样本的 State；由于输出是由 State 决定的，所以影响到了 State 意味着能够影响输出，亦即当前样本能够影响到下一个、乃至于下n 个样本的输出，从而能使结果反映出序特征。这就是 RNN 挖掘序关联的一般性思路。事实上，我们可以把 State 理解为网络结构的“记忆”——它能帮助网络“记住”之前看到过的样本的某些信息，并结合最新的样本所带来的信息来进行决策


* 传统的神经网络是层与层之间是全连接的，但是每层之间的神经元是没有连接的（其实是假设各个数据之间是独立的）
> 这种结构不善于处理序列化的问题。比如要预测句子中的下一个单词是什么，这往往与前面的单词有很大的关联，因为句子里面的单词并不是独立的。
* RNN 的结构说明当前的的输出与前面的输出也有关，即隐层之间的节点不再是无连接的，而是有连接的

基本的结构如图，可以看到有个循环的结构，将其展开就是右边的结构：
![RNN结构图](/img/in-post/rnn.jpg)

如上图：
* 输入单元(inputs units): $\{ {x_0},{x_1}, \cdots \cdots ,{x_t},{x_{t + 1}}, \cdots \cdots \}$,
* 输出单元(output units)为：$\{ {o_0},{o_1}, \cdots \cdots ,{o_t},{o_{t + 1}}, \cdots \cdots \}$,
* 隐藏单元(hidden units)输出集: $\{ {s_0},{s_1}, \cdots \cdots ,{ost},{s_{t + 1}}, \cdots \cdots \}$
* 时间 t 隐层单元的输出为：${s_t} = f(U{x_t} + W{s_{t - 1}})$
> f就是激励函数，一般是sigmoid,tanh, relu等
> 计算${s_{0}}$时，即第一个的隐藏层状态，需要用到${s_{-1}}$，但是其并不存在，在实现中一般置为0向量
> （如果将上面的竖着立起来，其实很像传统的神经网络，哈哈）
* 时间 t 的输出为：${o_t}=Softmax(V{s_t})$
> 可以认为隐藏层状态${s_t}$是网络的记忆单元. ${s_t}$包含了前面所有步的隐藏层状态。而输出层的输出${o_t}$只与当前步的${s_t}$有关。
> （在实践中，为了降低网络的复杂度，往往${s_t}$只包含前面若干步而不是所有步的隐藏层状态）
* 在RNNs中，每输入一步，每一层都共享参数U,V,W，（因为是将循环的部分展开，天然应该相等）
* RNNs的关键之处在于隐藏层，隐藏层能够捕捉序列的信息。


![朴素 RNN 的结构](/img/in-post/rnn2.jpg)
其中，x_{t-1},x_t,x_{t+1}、o_{t-1},o_t,o_{t+1}、s_{t-1},s_t,s_{t+1}可以分别视为第t-1,t,t+1“时刻”的输入、输出与 State。不难看出对于每一个时刻而言，朴素的 RNN 都可视为一个普通的神经网络：

## 循环神经网络（RNN）与递归神经网络（RNN）
RNN（Recurrent neural network，循环神经网络)是一系列能够处理序列数据的神经网络的总称。这里要注意循环神经网络和递归神经网络（Recursive neural network）的区别。

* recurrent: 时间维度的展开，代表信息在时间维度从前往后的的传递和积累，可以类比markov假设，后面的信息的概率建立在前面信息的基础上，在神经网络结构上表现为后面的神经网络的隐藏层的输入是前面的神经网络的隐藏层的输出；
* recursive: 空间维度的展开，是一个树结构，比如nlp里某句话，用recurrent neural network来建模的话就是假设句子后面的词的信息和前面的词有关，而用recurxive neural network来建模的话，就是假设句子是一个树状结构，由几个部分(主语，谓语，宾语）组成，而每个部分又可以在分成几个小部分，即某一部分的信息由它的子树的信息组合而来，整句话的信息由组成这句话的几个部分组合而来。

只是recurrent是时间递归（常用），而recursive是指结构递归神经网络

## LSTM
LSTM算法全称为Long short-term memory，最早由 Sepp Hochreiter和Jürgen Schmidhuber于1997年提出.是一种特别的 RNN，比标准的 RNN 在很多的任务上都表现得更好。

LSTMs 的 cell 的时间通道有两条。
LSTM 的关键就是细胞状态，水平线在图上方贯穿运行。
* 上方的时间通道（h(old)→h(new)）仅包含了两个代数运算,这意味着它信息传递的方式会更为直接h(new)=h(old)∗r1+r2
* 位于下方的时间通道（s(old)→s(new)）则运用了大量的层结构,在 LSTMs 中，我们通常称这些层结构为门（Gates）

> ϕ1是sigmoid函数，ϕ2是tanh函数
> *表示 element wise 乘法(就是点乘)，使用X表示矩阵乘法
一个经典的cell结构如下图
![LSTM结构](/img/in-post/RNN_LSTM.png)

## TensorFlow代码实现

### 1. RNNCell构建
RNNCell是TensorFlow中实现RNN的基本单元，每个RNNCell都有一个`call`方法，使用方式是：`(output, next_state) = call(input, state)`
<br>**RNNCell的Call方法的基本功能：**每调用一次RNNCell的`call`方法，就相当于在时间上“推进了一步”，如下图所示：
假设我们有一个初始状态h0，还有输入x1，调用`call(x1, h0)`后就可以得到`(output1, h1`)：
![first step](/img/in-post/rnn01.jpg)
再调用一次`call(x2, h1)`就可以得到`(output2, h2)`：
![second step](/img/in-post/rnn02.jpg)

RNNCell只是一个抽象类，我们用的时候都是用的它的两个子类`BasicRNNCell`和`BasicLSTMCell`。顾名思义，前者是RNN的基础类，后者是LSTM的基础类

对于RNNCell，还有两个类属性比较重要：
```
# 隐藏层大小
state_size
# 输出大小
output_size
```
比如我们通常是将一个batch送入模型计算，**设输入数据的形状为`(batch_size, input_size)`，那么计算时得到的隐层状态就是`(batch_size, state_size)`，输出就是`(batch_size, output_size)`。**
> state_size对应上图中h
```
import tensorflow as tf
import numpy as np

cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128) # state_size = 128
print(cell.state_size) # 128

inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
h0 = cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size)

output, h1 = cell.call(inputs, h0) #调用call函数

print(h1.shape) # (32, 128)
```
对于`batch_size`和`num_step`参数不理解的可以参见下图：
![参数解析](/img/in-post/rnn_data.jpg)
> 下图中每一行为一个batch，可以看到这里的batch_size = 3,每一列为一个num_step,下图中的num_steps为3

对于`BasicLSTMCell`，情况有些许不同，因为LSTM可以看做有两个隐状态h和c，对应的隐层就是一个Tuple，每个都是`(batch_size, state_size)`的形状：
```
import tensorflow as tf
import numpy as np

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)

inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
h0 = lstm_cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态
output, h1 = lstm_cell.call(inputs, h0)

print(h1.h)  # shape=(32, 128)
print(h1.c)  # shape=(32, 128)
```

### 2. 一次执行多步学习tf.nn.dynamic_rnn
RNNCell的`call`方法只是在序列时间上前进了一步。比如使用x1、h0得到h1，通过x2、h1得到h2等。这样的h话，如果我们的序列长度为10，就要调用10次call函数，比较麻烦。对此，TensorFlow提供了一个tf.nn.dynamic_rnn函数，使用该函数就相当于调用了n次call函数。即通过{h0,x1, x2, …., xn}直接得{h1,h2…,hn}。

具体来说，设我们输入数据的格式为`(batch_size, time_steps, input_size)`，其中`time_steps`表示序列本身的长度，如在Char RNN中，长度为10的句子对应的`time_steps`就等于10。最后的`input_size`就表示输入数据单个序列单个时间维度上固有的长度。另外我们已经定义好了一个RNNCell，调用该RNNCell的`call`函数`time_steps`次，对应的代码就是：
```
# inputs: shape = (batch_size, time_steps, input_size)
# cell: RNNCell
# initial_state: shape = (batch_size, cell.state_size)。初始状态。一般可以取零矩阵

outputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
```
此时，得到的outputs就是time_steps步里所有的输出。它的形状为`(batch_size, time_steps, cell.output_size)`。state是最后一步的隐状态，它的形状为`(batch_size, cell.state_size)`。

### 3. 多层RNN
将x输入第一层RNN的后得到隐层状态h，这个隐层状态就相当于第二层RNN的输入，第二层RNN的隐层状态又相当于第三层RNN的输入，以此类推。在TensorFlow中，可以使用`tf.nn.rnn_cell.MultiRNNCell`函数对RNNCell进行堆叠
```
import tensorflow as tf
import numpy as np

# 每调用一次这个函数就返回一个BasicRNNCell
def get_a_cell():
   return tf.nn.rnn_cell.BasicRNNCell(num_units=128)

# 用tf.nn.rnn_cell MultiRNNCell创建3层RNN
cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(3)]) # 3层RNN

# 得到的cell实际也是RNNCell的子类
# 它的state_size是(128, 128, 128)
# (128, 128, 128)并不是128x128x128的意思
# 而是表示共有3个隐层状态，每个隐层状态的大小为128

print(cell.state_size) # (128, 128, 128)

# 使用对应的call函数
inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
h0 = cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态
output, h1 = cell.call(inputs, h0)

print(h1) # tuple中含有3个32x128的向量
```

参考资料
<br>[完全图解RNN](https://zhuanlan.zhihu.com/p/28054589)
<br>[RNN基础知识](http://lawlite.me/2017/06/14/RNN-%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%92%8CLSTM-01%E5%9F%BA%E7%A1%80/)
<br>[理解LSTM网络(译)](https://www.jianshu.com/p/9dc9f41f0b29)
<br>[TensorFlow中RNN的正确打开方式](https://blog.csdn.net/starzhou/article/details/77848156)