---
layout:     post
title:      "CNN网络加入自动编码器"
subtitle:   "自动编码器对模型结果影响的一次实验"
date:       2018-06-24 10:00:00
author:     "Luo Yifu"
header-img: "img/post-bg-2015.jpg"
tags:
    - tensorflow
    - deeplearning
    - cnn
    - autoencoder
---

自动编码器起到增强模型鲁棒性，防止模型过度拟合的作用。但自动编码器在模型中作用如何，如何调试，以及参数设定等问题都需要我们一个个思考解决。

## 1. 在模型中加入自动编码器的效果
按照相关理论，自动编码器可以提升模型的鲁棒性。其原理实际就是在自动编码器训练的时候，人为加入噪声（比如服从正态分布的噪声）。

在tensorflow入门训练中，我们选择了mnist数据进行手写数字识别。基础的模型是一个输入层，一个Softmax层，然后就是输出层。
模型构建如图：

### 1.1 加入自动编码器后对模型结果的影响
在输入数据后，加入一层降噪自动编码器后，模型如图：
![有降噪自动编码器](/img/in-post/basic_mnist.png)
然后，经过5次实验，加入降噪编码器后，训练准确率为 **mean_accuracy = 0.8778**（进行了5次实验，结果分别为：0.8803,0.8562,0.8738,0.8955,0.8831），与没加入降噪编码器的结果(0.9168)相比，还略有不如。
> 自动编码器构建和训练：
> 这个自动编码器训练参考最后的“自动编码器训练”部分，参数设定为：`training_epochs=20`，`batch_size=128`

下面尝试用在CNN网络当中：
构建2层CNN网络，代码见文末代码部分，模型结构如图（使用tensorboard显示）。
模型构建参见[卷积神经网络TensorFlow实现](https://luoyifu.github.io/luoyifu.github.io/2018/06/02/TensorFlow_Mnist/)的TensorFlow代码实现部分。
模型训练结果：**mean_accuracy = 0.9439**（进行了三次实验，结果分别为：0.9362，0.9453，0.9502）
![无降噪自动编码器](/img/in-post/tensorboard_cnn.png)

加入降噪自动编码器后，模型训练结果为 **mean_accuracy = 0.9173**（进行了三次实验，结果分别为：0.9088，0.9129，0.9301）
> 还是使用刚刚训练得到的自动编码器（参数设定为：`training_epochs=20`，`batch_size=128`）

从这两次实验中看来，**自动编码器的加入，并没有改善训练结果，反而导致训练结果变差了一些**。

那么问题出在哪里？

### 1.2 模型训练结果和降噪自动编码器的关系

## 2. 自动编码器的训练问题
我们发现之前加入的自动编码器并不能优化CNN模型的结果。怀疑问题处在自动编码器上。重新回来训练自动编码器。决定自动编码器的效果的参数，看输入数据和经过自动编码再解码的输出数据的差值。在代码部分，就是`autoencoder.cost()`方法所获得的结果。

影响自动编码器结果的一个主要参数时隐藏节点数目（`n_hidden`）。隐藏节点数目少，误差就大。

## 2. CNN模型加入降噪自动编码器代码
### 2.1 CNN网络
```
# 加入自动编码器的CNN
import tensorflow as tf
import numpy as np 
import input_data
import AutoEncoder as AuEn

print('download and extract MNIST datasets')
mnist = input_data.read_data_sets('data/', one_hot=True)

# 加入提前训练好的自动编码器层
# 因为自动编码器训练比较慢，所以提前对其训练，在这个模型中直接导入。
with tf.name_scope('AutoEncoder'):
    autoencoder=AuEn.AdditiveGaussianNoiseAutoencoder(
        n_input=784,n_hidden=200,transfer_function=tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),scale=0.01)

    # 读取训练好的自动编码器：
    saver = tf.train.Saver()
    saver.restore(autoencoder.sess,'model/mnist_au_model.ckpt')

# 输入数据层
with tf.name_scope('Input'):
    x = tf.placeholder("float", shape=[None, 784],name='x_input')
    y_ = tf.placeholder("float", shape=[None, 10],name='y_input')

# 初始化偏置项
def weight_variable(shape):
  # 这是一个截断的产生正太分布的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成。
  # 和一般的正太分布的产生随机数据比起来，这个函数产生的随机数与均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的。
  initial = tf.truncated_normal(shape, stddev=0.1,name='weight')
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape,name='bias')
  return tf.Variable(initial)

def conv2d(x, W):
  # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
  # W 是权重也是过滤器/内核张量
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 构建第一层卷积
# 把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。
with tf.name_scope('First_Layer'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    # 把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数
    # (因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


with tf.name_scope('Second_Layer'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    # 构建第二层卷积
    # 为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个5x5的patch会得到64个特征。
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


with tf.name_scope('Full_acess'):
    # 现在，图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。
    # 我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    # 我们通过tf.reshape()将h_pool2的输出值从一个三维的变为一维的数据, -1表示先不考虑输入图片例子维度, 将上一个输出结果展平.
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 为了减少过拟合，我们在输出层之前加入dropout
# 用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
# 这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。 
# TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale。
# 所以用dropout的时候可以不用考虑scale。
with tf.name_scope('Drop_OUt'):
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 最后，我们添加一个softmax层，就像前面的单层softmax regression一样
# 输出层
with tf.name_scope('Output'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# 训练和评估
with tf.name_scope('cross_entropy'):
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('Evaluate'):
    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 输出summary，以便可视化
    summary_writer = tf.summary.FileWriter("cnn-logs/", sess.graph)

    for i in range(500):
        batch_xs, batch_ys = mnist.train.next_batch(50)
        batch_xs_1 = autoencoder.reconstruct( batch_xs )
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
            print ("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch_xs_1, y_: batch_ys, keep_prob: 1.0})
    
    print ("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```
### 2.2 自动编码器训练
CNN模型处理Mnist数据的自动编码器训练模块：
自动编码器代码参见[深度学习-自动编码器（AutoEnccoder）](https://luoyifu.github.io/luoyifu.github.io/2018/06/11/AutoEncoder_tensorflow/)
```
import tensorflow as tf
import numpy as np
import sklearn.preprocessing as prep
import AutoEncoder as AuEn

from tensorflow.examples.tutorials.mnist import input_data  
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  

'''自编码器中会使用一种参数初始化方法xavier initialization，它的特点是会根据某一层网络的输入，输出节点数量自动调整最合适的分布。
如果深度学习模型的权重初始化得太小，那信号将在每层间传递时逐渐缩小而难以产生作用，但如果权重初始化得太大，
那信号将在每层间传递时逐渐放大并导致发散和失效。而Xaiver初始化器做的事情就是让权重被初始化得不大不小，正好合适。
即让权重满足0均值，同时方差为2／（n（in）+n(out)），分布可以用均匀分布或者高斯分布。
下面fan_in是输入节点的数量，fan_out是输出节点的数量。'''

# 定义一个对训练、测试数据进行标准化处理的函数
# 标准化即让数据变成0均值且标准差为1的分布。方法就是先减去均值，再除以标准差。
def standard_scale(X_train,X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train=preprocessor.transform(X_train)
    X_test=preprocessor.transform(X_test)
    return X_train,X_test

# 再定义一个获取随机block数据的函数：
# 取一个从0到len(data)-batch_size之间的随机整数，
# 再以这个随机数作为block的起始位置，然后顺序取到一个batch size的数据。
# 要注意的是，这属于不放回抽样，可以提高数据的利用效率
def get_random_block_from_data(data,batch_size):
    start_index = np.random.randint(0,len(data) - batch_size)
    return data[start_index:(start_index+batch_size)]

# 用之前定义的standard_scale函数对训练集、测试机进行标准化变换

# 创建一个自编码器的实例，定义模型输入节点数n_input为784，
# 自编码器的隐含层点数n_hidden为200，隐含层的激活函数transfer_function为softplus，优化器optimizer为Adam
# 且学习速率为0。001，同时将噪声的系数设为0.01
with tf.name_scope('AutoEncoder'):
    autoencoder=AuEn.AdditiveGaussianNoiseAutoencoder(
        n_input=784,n_hidden=200,transfer_function=tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),scale=0.01)

# 定义是重新训练模型还是读取已有模型，如果is_Trained = True， 则表示模型已经训练过了，直接读取模型
is_Trained = False
saver = tf.train.Saver()

if is_Trained :
    saver.restore(autoencoder.sess,'model/mnist_au_model.ckpt')
else:
    X_train,X_test=standard_scale(mnist.train.images,mnist.test.images)
    # 下面定义几个常用参数，总训练样本数，最大训练的轮数(traning_epochs)设为20，
    # batch_size设为128，并设置每隔一轮(epoch)就显示一次损失cost
    # 下面开始训练过程，在每一轮(epoch)循环开始时，将平均损失avg_cost设为0，
    # 并计算总共需要的batch数（通过样本总数除以batch大小），
    # 在每一轮迭代后，显示当前的迭代数和这一轮迭代的平均cost。
    n_samples=int(mnist.train.num_examples)
    training_epochs=160
    batch_size=128
    display_step=1
    for epoch in range(training_epochs):
        total_batch = int(n_samples/batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)
            cost = autoencoder.partial_fit(batch_xs)
        if epoch%10 == 0:
            print ("step %d, training cost %g"%(epoch, cost))
    # 将训练好的模型保存起来
    saver.save(autoencoder.sess,'model/mnist_au_model.ckpt') 

print('The model is ready!')   
```