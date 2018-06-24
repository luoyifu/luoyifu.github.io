---
layout:     post
title:      "深度学习-自动编码器"
subtitle:   "自动编码器&降噪自动编码器&TensorFlow"
date:       2018-06-11 10:00:00
author:     "Luo Yifu"
header-img: "img/post-bg-2015.jpg"
tags:
    - tensorflow
    - deeplearning
---
# 深度学习-自动编码器（AutoEnccoder）
如果给定一个神经网络，我们假设其输出与输入是相同的，然后训练调整其参数，得到每一层中的权重。自然地，我们就得到了输入I的几种不同表示（每一层代表一种表示），这些表示就是特征。

自动编码器就是一种尽可能复现输入信号的神经网络。为了实现这种复现，自动编码器就必须捕捉可以代表输入数据的最重要的因素（类似主成分分析PCA，找到主要特征）。

**使用自动编码器的目的：**在研究中可以发现，如果在原有的特征中加入这些自动学习得到的特征（自动编码器习得的特征）可以大大提高精确度，甚至在分类问题中比目前最好的分类算法效果还要好！


## 1. 自动编码器——无标签数据非监督学习特征
Auto-Encoder(AE)是20世纪80年代晚期提出的，它是一种无监督学习算法，使用了反向传播算法，让目标值等于输入值。基本的AE可视为一个三层神经网络结构：一个输入层、一个隐藏层和一个输出层，其中输出层与输入层具有相同的规模。
### 1.1 编码解码过程
在监督学习的情况下，输入的样本是有标签的（input, target），这样可以根据当前输出和target（label）之间的差去改变前面各层的参数，学习直到收敛。如下图（左）：
![有标签数据的学习](/img/in-post/autoencoder01.jpg)
但对于无标签数据，左图的学习过程就无法实现了。这样，我们用下图所示方式进行学习：
![无标签数据的学习](/img/in-post/autoencoder02.jpg)
首先对输入数据输入编码器进行编码（encoder），得到一个code，这个code也就是输入的一个表示。为了验证编码的这个code代表输入数据，再对code进行解码（decoder）。如果输出数据和一开始的输入数据是很像的（理想状态下完全一致），那么这个编码得到的code是可信的。所以，学习过程就是调整参数，使得输出和输入相似。训练结束后我们就得到了输入数据的一个编码表示。误差的来源就是直接重构后与原输入相比得到。

### 1.2 多层训练
通过编码和解码训练，我们得到了一个编码code。这个编码可以看做输入数据的另一种表示（因为code解码后和原始输入很相似，虽然表达不同，但信息是一致的）。接下来，再构建第二层编码器，和第一层构建方式一样。这里将第一层得到的code作为第二层的输入，训练后得到第二层的code（也即原始数据的第二种表示方式）。
其他层就同样的方法炮制就行了（训练这一层，前面层的参数都是固定的，并且他们的decoder已经没用了，都不需要了）

### 1.3 有监督的微调
经过多层编码器的训练，得到很多层编码code。至于需要多少层（或者深度需要多少，这个目前本身就没有一个科学的评价方法）需要自己试验调了。每一层都和原始输入不同的表示，可以想象成人的视觉识别，每一层都更加抽象。
至此，AutoEncoder学会了如何复现输入，得到输入的其他表现方式（或者说，它只是学习获得了一个可以良好代表输入的特征，这个特征可以最大程度上代表原输入信号）。

为了实现分类，我们就可以在AutoEncoder的最顶的编码层添加一个分类器（例如logistic回归、SVM等），然后通过标准的多层神经网络的监督训练方法（梯度下降法）去训练。

也就是说，这时候，我们需要将最后层的特征code输入到最后的分类器，通过有标签样本，通过监督学习进行微调，这也分两种，一个是只调整分类器（黑色部分）：
![微调分类器](/img/in-post/autoencoder03.jpg)
另一种：通过有标签样本，微调整个系统：（如果有足够多的数据，这个是最好的。end-to-end learning端对端学习）
![微调整个系统](/img/in-post/autoencoder04.jpg)

一旦监督训练完成，这个网络就可以用来分类了。神经网络的最顶层可以作为一个线性分类器，然后我们可以用一个更好性能的分类器去取代它。

在研究中可以发现，如果在原有的特征中加入这些自动学习得到的特征可以大大提高精确度，甚至在分类问题中比目前最好的分类算法效果还要好！

## 2. 降噪自动编码器Denoising AutoEncoder
降噪自动编码器DenoisingAutoEncoder是在自动编码器的基础上，训练数据加入噪声，所以自动编码器必须学习去去除这种噪声而获得真正的没有被噪声污染过的输入。因此，这就迫使编码器去学习输入信号的更加鲁棒的表达，这也是它的泛化能力比一般编码器强的原因。DA可以通过梯度下降算法去训练。

### 2.1 为什么要构建降噪自动编码器
在模型的复杂度和数据量都已经确定的前提下，防止过拟合的一种办法是减少数据中的噪音数量，即对训练集数据做清洗操作。然而，如果我们无法检测并删除掉数据中的噪音。另一种防止过拟合的办法就是**给数据中增加噪音**，这看似与之前的结论矛盾，但却是增强模型鲁棒性的一种有效方式。以手写数字识别为例，假如现在我们输入的是一副含有一定噪音的图片，例如图片中有污点，图片中的数字倾斜等，并且我们仍然希望解码后的图片是一副干净正确的图片，这就需要编码器不仅有编码功能，还得有去噪音的作用，通过这种方式训练出的模型具有更强的鲁棒性。

Denoising Autoencoder（降噪自动编码器）就是在Autoencoder的基础之上，为了防止过拟合问题而对输入的数据（网络的输入层）加入噪音，使学习得到的编码器W具有较强的鲁棒性，从而增强模型的泛化能力。Denoising Autoencoder是Bengio在08年提出的
关于Denoising Autoencoder的示意图如下，其中x是原始的输入数据，Denoising Autoencoder以一定概率把输入层节点的值置为0，从而得到含有噪音的模型输入xˆ。
![降噪自动编码器](/img/in-post/autoencoder05.jpg)

* Denoising Autoencoder与人的感知机理类似，比如人眼看物体时，如果物体某一小部分被遮住了，人依然能够将其识别出来。
* 人在接收到多模态信息时（比如声音，图像等），少了其中某些模态的信息有时也不会造成太大影响。
* Autoencoder的本质是学习一个相等函数，即网络的输入和重构后的输出相等，这种相等函数的表示有个缺点就是当测试样本和训练样本不符合同一分布，即相差较大时，效果不好，而Denoising Autoencoder在这方面的处理有所进步。


## 3. tensorflow代码实现

降噪自动编码器
```
# --降噪自动编码器
# 参考 https://blog.csdn.net/qq_31531635/article/details/76158288
# 定义降噪自动编码器类

# 实现一个去噪自编码器和实现一个单隐含层的神经网络差不多，
# 只不过是在数据输入时做了标准化，并加上了一个高斯噪声，同时我们的输出结果不是数字分类结果，
# 而是复原的数据，因此不需要用标注过的数据进行监督训练。
# 自编码器作为一种无监督学习的方法，它与其它无监督学习的主要不同在于，它不是对数据进行聚类，
# 而是提取其中最有用，最频繁出现的高阶特征，根据这些高阶特征重构数据。
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt  
import sklearn.preprocessing as prep

'''自编码器中会使用一种参数初始化方法xavier initialization，它的特点是会根据某一层网络的输入，输出节点数量自动调整最合适的分布。
如果深度学习模型的权重初始化得太小，那信号将在每层间传递时逐渐缩小而难以产生作用，但如果权重初始化得太大，
那信号将在每层间传递时逐渐放大并导致发散和失效。而Xaiver初始化器做的事情就是让权重被初始化得不大不小，正好合适。
即让权重满足0均值，同时方差为2／（n（in）+n(out)），分布可以用均匀分布或者高斯分布。
下面fan_in是输入节点的数量，fan_out是输出节点的数量。'''

def xavier_init(fan_in,fan_out,constant=1):
    low = -constant * np.sqrt(6.0/(fan_in+fan_out))
    high = constant * np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):
    
    def __init__(self, 
                n_input, 
                n_hidden, 
                transfer_function =tf.nn.softplus, 
                optimizer = tf.train.AdamOptimizer(),  
                scale = 0.1):
        # scale参数表示噪声规模大小。构建hidden层时，给输入x增加了一个服从正态分布（高斯分布）的噪声
        # 噪声规模用scale修饰。未加规模参数的噪声取值[0,1]。
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # 定义网络结构，为输入x创建n_input长度的placeholder
        # 定义提取特征的隐藏层（hidden）：先将输入x加上噪声，然后用tf.matmul将加了噪声的输入与隐含层的权重相乘
        # 并使用tf.add加上隐含层的偏置，最后对结果进行激活函数处理。
        # 经过隐含层后，需要在输出层进行数据复原，重建操作(reconstruction)
        self.x=tf.placeholder(tf.float32,[None,self.n_input])
        
        self.hidden=self.transfer(tf.add(tf.matmul(self.x+scale * tf.random_normal((n_input,)),
                self.weights['w1']),self.weights['b1']))
        self.reconstruction=tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])
        
        # 接下来定义自编码器的损失函数，这里使用平方误差作为损失函数，
        # 再定义训练操作作为优化器对损失进行优化，最后创建Session并初始化自编码器的全部模型参数。
        self.cost=0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))
        self.optimizer=optimizer.minimize(self.cost)

        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)
    
    # 定义初始化参数的函数
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input],dtype= tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],dtype=tf.float32))
        return all_weights

    # 定义计算损失cost及执行一步训练的函数partial_fit。
    # 函数里只需让Session执行两个计算图的节点，分别是损失cost和训练过程optimizer，输入的feed_dict包括输入数据x，
    # 以及噪声的系数scale。函数partial_fit做的就是用一个batch数据进行训练并返回当前的损失cost。
    def partial_fit(self,X):
        cost,opt=self.sess.run((self.cost,self.optimizer),feed_dict={self.x:X,self.scale:self.training_scale})
        return cost

    # 下面为一个只求损失cost的函数，这个函数是在自编码器训练完毕后，在测试集上对模型性能进行评测时会用到的。
    def calc_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict={self.x:X,self.scale:self.training_scale})
    
    # 定义transform函数，返回自编码器隐含层的输出结果，
    # 它的目的是提供一个接口来获取抽象后的特征，自编码器的隐含层的最主要功能就是学习出数据中的高阶特征。
    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x:X,self.scale:self.training_scale})
    
    # 定义generate函数，将隐含层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据
    def generate(self,hidden=None):
        if hidden is None:
            hidden=np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})
    
    # 定义reconstruct函数，它整体运行一遍复原过程，包括提取高阶特征和通过高阶特征复原数据
    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,feed_dict={self.x:X,self.scale:self.training_scale})

    # 定义getWeights函数的作用是获取隐含层的权重w1
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    # 定义getBiases函数则是获取隐含层的偏置系数b1
    def getBiases(self):
        return self.sess.run(self.weights['b1'])

```
这是在网上看到的另外一份代码：
```
# -*- coding: utf-8 -*-  
import tensorflow as tf  
import numpy as np  
import Utils  
  
class Autoencoder(object):  
  
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer = tf.train.AdamOptimizer()):  
        # softplus: ln(exp(x) + 1) 它的倒数就是sigmoid函数，它限制左边始终大于0，右边没有限制  
        self.n_input = n_input  
        self.n_hidden = n_hidden  
        self.transfer = transfer_function  
  
        network_weights = self._initialize_weights()  
        self.weights = network_weights  
  
        # model  
        # n_input 指的是输入的维度  
        self.x = tf.placeholder(tf.float32, [None, self.n_input])  
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))  
        # 将隐藏层的输出限制在大于0的空间里面，  
        # 这就是一个非线性环节，只限制左端，并没有限制右端  
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])  
        # 在这里重构结果，只是隐藏环节的一个线性变换，  
        # 并没有加上任何非线性的环节，所以后面求重构误差的时候，  
        # 直接使用均方误差就行（因为这里并没有使用非线性环节进行归一化，要是使用了，可以利用交叉熵）  
  
        # cost  
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.sub(self.reconstruction, self.x), 2.0))  
        # 做减法，然后平方，这里使用的是重构误差最小，使用均方误差  
        self.optimizer = optimizer.minimize(self.cost)  
  
        init = tf.initialize_all_variables()  
        self.sess = tf.Session()  
        self.sess.run(init)  
  
  
    def _initialize_weights(self):  
        all_weights = dict()  
        all_weights['w1'] = tf.Variable(Utils.xavier_init(self.n_input, self.n_hidden))  
        # 在初始化的时候，就只是第一个权重矩阵是赋予一些随机值，其他的都是赋予全0矩阵  
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))  
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))  
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))  
        return all_weights  
  
    def partial_fit(self, X):  
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})  
        return cost  
  
    def calc_total_cost(self, X):  
        return self.sess.run(self.cost, feed_dict = {self.x: X})  
  
    def transform(self, X):  
        # 这个函数在整个自编码器训练好了之后，使用它可以将原始数据进行编码，得到的结果就是编码形式  
        return self.sess.run(self.hidden, feed_dict={self.x: X})  
  
    def generate(self, hidden = None):  
        # 这部分就是解码函数，用作解码用  
        if hidden is None:  
            hidden = np.random.normal(size=self.weights["b1"])  
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})  
  
    def reconstruct(self, X):  
        # 先进行编码，然后进行解码，得到的最后结果  
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})  
  
    def getWeights(self):  
        # 得到编码用的权值矩阵  
        return self.sess.run(self.weights['w1'])  
  
    def getBiases(self):  
        # 得到编码器的偏重值  
        return self.sess.run(self.weights['b1'])  
```

降噪自动编码器
```
emptyclass AdditiveGaussianNoiseAutoencoder(object):  
    # 在初始的数据中加入高斯噪声。在实现降噪自编码器的时候，  
    # 只是在输入加进去的时候，在输入上加上高斯噪声就行  
    # 其他的部分和基本自编码器一样  
    def __init__(self, n_input, n_hidden, transfer_function = tf.nn.softplus, optimizer = tf.train.AdamOptimizer(),  
                 scale = 0.1):  
        self.n_input = n_input  
        self.n_hidden = n_hidden  
        # n_input,n_hidden都是输入和隐藏层的维度  
        self.transfer = transfer_function  
        self.scale = tf.placeholder(tf.float32)  
        self.training_scale = scale  
        # scale 就是一个标量  
        network_weights = self._initialize_weights()  
        self.weights = network_weights  
  
        # model  
        self.x = tf.placeholder(tf.float32, [None, self.n_input])  
        # none不给定具体的值，它由输入的数目来决定  
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),  
                self.weights['w1']),  
                self.weights['b1']))  
        # 在输入的时候，在输入的数据上加上一些高斯噪声  
        # tf.random_normal((n_input,)) 默认  
        # 给的是一个均值为0，标准差是1的正态分布的随机数  
  
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])  
  
        # cost  
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.sub(self.reconstruction, self.x), 2.0))  
        self.optimizer = optimizer.minimize(self.cost)  
  
        init = tf.initialize_all_variables()  
        self.sess = tf.Session()  
        self.sess.run(init)  
          
              
class MaskingNoiseAutoencoder(object):  
    # 将有些数据直接忽略掉，即直接将一部分数据直接赋予0  
    def __init__(self, n_input, n_hidden, transfer_function = tf.nn.softplus, optimizer = tf.train.AdamOptimizer(),  
                 dropout_probability = 0.95):  
        self.n_input = n_input  
        self.n_hidden = n_hidden  
        self.transfer = transfer_function  
        self.dropout_probability = dropout_probability  
        self.keep_prob = tf.placeholder(tf.float32)  
  
        network_weights = self._initialize_weights()  
        self.weights = network_weights  
  
        # model  
        # 直接在输入数据上使用dropout来实现它，这样就实现了masking noise  
        self.x = tf.placeholder(tf.float32, [None, self.n_input])  
        self.hidden = self.transfer(tf.add(tf.matmul(tf.nn.dropout(self.x, self.keep_prob), self.weights['w1']),  
                                           self.weights['b1']))  
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])  
  
        # cost  
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.sub(self.reconstruction, self.x), 2.0))  
        self.optimizer = optimizer.minimize(self.cost)  
  
        init = tf.initialize_all_variables()  
        self.sess = tf.Session()  
        self.sess.run(init)  
```
## 4. 降噪自动个编码器的效果
我们使用mnist数据作为例子。加入降噪自动编码器后，原始数据和降噪自动编码后，将数据重构后的结果对比：
![训练轮数5](/img/in-post/autoencoder_1.png)
![训练轮数20](/img/in-post/autoencoder_2.png)
![训练轮数40](/img/in-post/autoencoder_3.png)
![训练轮数80](/img/in-post/autoencoder_4.png)
代码：
```
#----------------------------------------------------------------#
# --降噪自动编码器

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt  
import sklearn.preprocessing as prep
# AutoEncoder在上面的代码中有定义
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

# 创建一个自编码器的实例，定义模型输入节点数n_input为784，
# 自编码器的隐含层点数n_hidden为200，隐含层的激活函数transfer_function为softplus，优化器optimizer为Adam
# 且学习速率为0。001，同时将噪声的系数设为0.01
with tf.name_scope('AutoEncoder'):
    autoencoder=AuEn.AdditiveGaussianNoiseAutoencoder(
        n_input=784,n_hidden=200,transfer_function=tf.nn.softplus,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001),scale=0.01)

# 是否训练模型，还是读取已有模型
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
    training_epochs=80
    batch_size=128
    display_step=1
    examples_to_show = 10 
    for epoch in range(training_epochs):
        total_batch = int(n_samples/batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)
            cost = autoencoder.partial_fit(batch_xs)

    saver.save(autoencoder.sess,'model/mnist_au_model.ckpt') 

print('The model is ready!')   

au=[]
au = autoencoder.reconstruct(mnist.test.images[:examples_to_show])

f, a = plt.subplots(2, 10, figsize=(10, 2))  
for i in range(examples_to_show):  
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))  
    a[1][i].imshow(np.reshape(au[i], (28, 28)))  
plt.show() 
```