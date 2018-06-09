---
layout:     post
title:      "TensorFLow卷积神经网络"
subtitle:   "使用TensorFLow构建卷积神经网络"
date:       2018-06-02 19:00:00
author:     "Luo Yifu"
header-img: "img/post-bg-2015.jpg"
tags:
    - 卷积神经网络(CNN)
    - deeplearning
    - TensorFlow
---
# TensorFlow构建卷积神经网络

卷积神经网络（CNN）由输入层、卷积层、激活函数、池化层、全连接层组成，即INPUT-CONV-RELU-POOL-FC

* 卷积层: 做特征的提取，输出对应得feature map
* 池化层: 就是对数据进行下采样，减少数据处理量同时保留有用信息
* 全连接层: 就是对提取特征进行组织综合，输出识别物体的分类情况

## 1. 重要概念
### 卷积
卷积的本质就是加权叠加。卷积核与一个连接观察窗口的全连接神经元类似，因此我们用它来代替我们上文提到的特征观察神经元，并最后通过窗口滑动观察整个输入，输出一个 feature map

### ReLu激活函数
最近几年卷积神经网络中，激活函数往往不选择sigmoid或tanh函数，而是选择relu函数。
Relu函数作为激活函数，有下面几大优势：
* 速度快：和sigmoid函数需要计算指数和倒数相比，relu函数其实就是一个max(0,x)，计算代价小很多。
* 减轻梯度消失问题 回顾计算梯度的公式∇=σ′δx。其中，σ′是sigmoid函数的导数。在使用反向传播算法进行梯度计算时，每经过一层sigmoid神经元，梯度就要乘上一个σ′。从下图可以看出，σ′函数最大值是1/4。因此，乘一个会导致梯度越来越小，这对于深层网络的训练是个很大的问题。而relu函数的导数是1，不会导致梯度变小。当然，激活函数仅仅是导致梯度减小的一个因素，但无论如何在这方面relu的表现强于sigmoid。使用relu激活函数可以让你训练更深的网络。
* 稀疏性 通过对大脑的研究发现，大脑在工作的时候只有大约5%的神经元是激活的，而采用sigmoid激活函数的人工神经网络，其激活率大约是50%。有论文声称人工神经网络在15%-30%的激活率时是比较理想的。因为relu函数在输入小于0时是完全不激活的，因此可以获得一个更低的激活率

通过ReLu激活函数，可以对卷积层进行过滤，效果如图：
![ReLu激活函数](/img/in-post/ReLu.png)

### 卷积核
卷积核虽然模拟的是一个特征观察神经元，但它并不属于卷积层，它相当于一个特征过滤器（或说是一个权重矩阵）。它将符合自己要求的特征输出到feature map上。
* 不同卷积核提取图片中的不同信息
* 多个卷积层对特征进行提取组合后，可以得到比较复杂的特征

至于卷积层为什么可以提取特征并取得很好的效果，可以看下面表示特征的卷积核与输入图片进行运算后提取的feature map 

输入为28*28的图像，经过5*5的卷积之后，得到一个(28-5+1)*(28-5+1) = 24*24、的map。卷积层的每个map是不同卷积核在前一层每个map上进行卷积，并将每个对应位置上的值相加然后再加上一个偏置项。
![卷积核](/img/in-post/cnn_juanji.png)
每次用卷积核与map中对应元素相乘，然后移动卷积核进行下一个神经元的计算。如图中矩阵C的第一行第一列的元素2，就是卷积核在输入map左上角时的计算结果。在图中也很容易看到，输入为一个4*4的map，经过2*2的卷积核卷积之后，结果为一个(4-2+1) *(4-2+1) = 3*3的map。

## 2. 输入层和卷积层

卷积层，进行特征提取，通常会使用多层卷积层来得到更深层次的特征图。

卷积层的参数：

设置几个参数：filters大小：F, filters数量；K, 滑动步长（stride size）:S, 边距（pad）:P
输入数据尺寸为；W1*H1*D1
输出数据尺寸为：
W2=[(W1-F+2P)/S]+1
H2=[(H1-F+2P)/S]+1
D2=K

> 一般有:F=3 => zero pad with 1
> F=5 => zero pad with 2
> F=7=> zero pad with 3



## 3. 池化层（pooling layer）
池化层：对输入的特征图进行压缩，一方面使特征图变小，简化网络计算复杂度；一方面进行特征压缩，提取主要特征。
池化操作一般有两种，一种是Avy Pooling,一种是max Pooling:

![max pooling](/img/in-post/cnn_max_pooling.png)
同样地采用一个2*2的filter,max pooling是在每一个区域中寻找最大值，这里的stride=2,最终在原特征图中提取主要特征得到右图。

（Avy pooling现在不怎么用了，方法是对每一个2*2的区域元素求和，再除以4，得到主要特征），而一般的filter取2*2,最大取3*3,stride取2，压缩为原来的1/4.

注意：这里的pooling操作是特征图缩小，有可能影响网络的准确度，因此可以通过增加特征图的深度来弥补（这里的深度变为原来的2倍）。

池化层里我们用的maxpooling，将主要特征保留，舍去多余无用特征,这样就可以实现信息压缩

## 全连接层
连接所有的特征，将输出值送给分类器（如softmax分类器），如图：
![全连接层](/img/in-post/quanlianjie.png)
输出不同分类的概率：
![输出不同分类](/img/in-post/shuchu.png)
## 整体概览
总的一个cnn网络结构大致如下：
![example of cnn](/img/in-post/cnn_example.png)
另外：CNN网络中前几层的卷积层参数量占比小，计算量占比大；而后面的全连接层正好相反，大部分CNN网络都具有这个特点。因此我们在进行计算加速优化时，重点放在卷积层；进行参数优化、权值裁剪时，重点放在全连接层。

> * 最左边是数据输入层：对数据做一些处理，比如去均值（把输入数据各个维度都中心化为0，避免数据过多偏差，影响训练效果）、归一化（把所有的数据都归一到同样的范围）、PCA/白化等等。CNN只对训练集做“去均值”这一步。
> * 中间是:
> CONV：卷积层，线性乘积 求和。
> RELU：激励层，使用relu做卷积层的激活函数。
> POOL：池化层，简言之，即取区域平均或最大。

> * 最右边是：FC：全连接层

## CNN的特点：

1、局部感知：一般认为图像的空间联系是局部的像素联系比较密切，而距离较远的像素相关性较弱，因此，每个神经元没必要对全局图像进行感知，只要对局部进行感知，然后在更高层将局部的信息综合起来得到全局信息。利用卷积层实现：（特征映射，每个特征映射是一个神经元阵列）：从上一层通过局部卷积滤波器提取局部特征。卷积层紧跟着一个用来求局部平均与二次提取的计算层，这种二次特征提取结构减少了特征分辨率。

2、参数共享：在局部连接中，每个神经元的参数都是一样的，即：同一个卷积核在图像中都是共享的。（理解：卷积操作实际是在提取一个个局部信息，而局部信息的一些统计特性和其他部分是一样的，也就意味着这部分学到的特征也可以用到另一部分上。所以对图像上的所有位置，都能使用同样的学习特征。）卷积核共享有个问题：提取特征不充分，可以通过增加多个卷积核来弥补，可以学习多种特征。

![example of cnn](/img/in-post/cnn_example1.png)
如没有这个原则，则特征图由10个32*32*1的特征图组成，每个特征图上有32*32=1024个神经元，每个神经元对应输入图像上一块5*5*3的区域，即一个神经元和输入图像的这块区域有75个连接，即75个权值参数，则所有特征图共有75*1024*10=768000个权值参数，这是非常复杂的
因此卷积神经网络引入“权值”共享原则，即一个特征图上每个神经元对应的75个权值参数被每个神经元共享（这张特征图所有神经元对应的参数都一样），这样则只需75*10=750个权值参数，而每个特征图的阈值也共享，即需要10个阈值，则总共需要750+10=760个参数。

3、采样(池化)层：在通过卷积得到特征后，希望利用这些特征进行分类。基于局部相关性原理进行亚采样，在减少数据量的同时保留有用信息。（压缩数据和参数的量，减少过拟合）（max-polling 和average-polling）


## 4. InteractiveSession类

`InteractiveSession`类，它能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)构成的。这对于工作在交互式环境中的人们来说非常便利，比如使用IPython。如果你没有使用InteractiveSession，那么你需要在启动session之前构建整个计算图，然后启动该计算图。
与Session()类的主要区别：
* 在没有指定Session变量的情况下运行变量（which allows us to run variables without needing to constantly refer to the session object (less typing!)）

* Session（）使用`with..as..`后可以不使用`close`关闭对话，而调用InteractiveSession需要在最后调用`close`

TensorFlow在卷积和池化上有很强的灵活性。我们怎么处理边界？步长应该设多大？在这个实例里，我们会一直使用vanilla版本。我们的卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小。我们的池化用简单传统的2x2大小的模板做max pooling。为了代码更简洁，我们把这部分抽象成一个函数。
```
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
```


卷积层定义函数：
```
def conv2d(x, W):
  # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
```


给定一个input的张量[batch, in_height, in_width, in_channels]和一个过滤器 / 内核张量 [filter_height, filter_width, in_channels, out_channels]后，执行以下操作：

> 1. 展平filter为一个形状为[filter_height * filter_width * in_channels, output_channels]的二维矩阵。
> 2. 从input中按照filter大小提取图片子集形成一个大小为[batch, out_height, out_width, filter_height * filter_width * in_channels]的虚拟张量。
> 3. 循环每个图片子集，右乘filter矩阵。



参考资料：
<br>[非常棒的卷积神经网络的基础知识，动图让理论更生动](https://blog.csdn.net/qq_31456593/article/details/76083091)
<br>[卷积和池化](https://www.cnblogs.com/believe-in-me/p/6645402.html)
<br>[tf.nn.conv2d函数理解](https://www.cnblogs.com/qggg/p/6832342.html)