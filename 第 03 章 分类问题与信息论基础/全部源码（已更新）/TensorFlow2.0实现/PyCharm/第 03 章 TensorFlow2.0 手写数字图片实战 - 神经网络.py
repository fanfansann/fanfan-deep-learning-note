# 1. 导入模块，并设置图像参数
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.datasets as datasets

plt.rcParams['font.size'] = 4
plt.rcParams['font.family'] = ['STKaiti']
plt.rcParams['axes.unicode_minus'] = False


# 2. 导入数据，并对数据进行简单处理

def load_data() :
    # 加载 MNIST 数据集 元组tuple: (x, y), (x_val, y_val)
    (x, y), (x_val, y_val) = datasets.mnist.load_data()
    # 将 x 转换为浮点张量，并从 0 ~ 255 缩放到 [0, 1.] - 1 -> [-1, 1] 即缩放到 -1 ~ 1
    x = tf.convert_to_tensor(x, dtype = tf.float32) / 255. - 1
    # 转换为整形张量
    y = tf.convert_to_tensor(y, dtype = tf.int32)
    # one-hot 编码
    y = tf.one_hot(y, depth = 10)
    # 改变视图， [b, 28, 28] => [b, 28*28]
    x = tf.reshape(x, (-1, 28 * 28))
    # 构建数据集对象
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    # 批量训练
    train_dataset = train_dataset.batch(200)
    return train_dataset


# 3. 对神经网络参数进行初始化

def init_paramaters() :
    # 每层的张量需要被优化，使用 Variable 类型，并使用截断的正太分布初始化权值张量
    # 偏置向量初始化为 0 即可
    # 第一层参数
    w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev = 0.1))
    b1 = tf.Variable(tf.zeros([256]))
    # 第二层参数
    w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev = 0.1))
    b2 = tf.Variable(tf.zeros([128]))
    # 第三层参数
    w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev = 0.1))
    b3 = tf.Variable(tf.zeros([10]))
    return w1, b1, w2, b2, w3, b3


# 4. 模型训练

def train_epoch(epoch, train_dataset, w1, b1, w2, b2, w3, b3, lr=0.001):
    for step, (x, y) in enumerate( train_dataset ):
        with tf.GradientTape() as tape:
            # 第一层计算， [b, 784]@[784, 256] + [256] => [b, 256] + [256] => [b,256] + [b, 256]
            h1 = x @ w1 + tf.broadcast_to( b1, (x.shape[0], 256) )
            # 通过激活函数 relu
            h1 = tf.nn.relu( h1 )
            # 第二层计算， [b, 256] => [b, 128]
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu( h2 )
            # 输出层计算， [b, 128] => [b, 10]
            out = h2 @ w3 + b3

            # 计算网络输出与标签之间的均方差， mse = mean(sum(y - out) ^ 2)
            # [b, 10]
            loss = tf.square( y - out )
            # 误差标量， mean: scalar
            loss = tf.reduce_mean( loss )
            # 自动梯度，需要求梯度的张量有[w1, b1, w2, b2, w3, b3]
            grads = tape.gradient( loss, [w1, b1, w2, b2, w3, b3] )

        # 梯度更新， assign_sub 将当前值减去参数值，原地更新
        w1.assign_sub( lr * grads[0] )
        b1.assign_sub( lr * grads[1] )
        w2.assign_sub( lr * grads[2] )
        b2.assign_sub( lr * grads[3] )
        w3.assign_sub( lr * grads[4] )
        b3.assign_sub( lr * grads[5] )

        if step % 100 == 0:
            print( f"epoch:{epoch}, iteration:{step}, loss:{loss.numpy()}" )

    return loss.numpy()


def train(epochs) :
    losses = []
    train_dataset = load_data()
    w1, b1, w2, b2, w3, b3 = init_paramaters()
    for epoch in range(epochs) :
        loss = train_epoch(epoch, train_dataset, w1, b1, w2, b2, w3, b3, lr = 0.01)
        losses.append(loss)
    x = [i for i in range(0, epochs)]
    # 绘制曲线
    plt.plot(x, losses, color = 'cornflowerblue', marker = 's', label = '训练', markersize='5')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig('MNIST数据集的前向传播训练误差曲线.png')
    plt.show()
    plt.close()

if __name__ == '__main__' :
	# x 轴 0 ~ 50
    train(epochs = 50)