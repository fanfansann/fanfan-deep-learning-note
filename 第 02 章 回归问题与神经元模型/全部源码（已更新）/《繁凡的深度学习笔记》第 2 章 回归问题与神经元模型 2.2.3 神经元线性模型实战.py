import numpy as np
import math
from matplotlib import pyplot as plt
# cal y = 1.477x + 0.089 + epsilon,epsilon ~ N(0, 0.01^2)

# plt参数设置
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = ['STKaiti']
plt.rcParams['axes.unicode_minus'] = False

# 生成数据
def get_data():
    # 计算均方误差
    #保存样本集的列表
    data = []
    for i in range(100):
        x = np.random.uniform(-10., 10.)  # 随机采样 x
        # 高斯噪声
        eps = np.random.normal(0., 0.01) # 均值和方差
        # 得到模型的输出
        y = 1.477 * x + 0.089 + eps
        # 保存样本点
        data.append([x, y])
    # 转换为2D Numpy数组
    data = np.array(data)
    return data

# mse 损失函数
def mse(b, w, points) :
    totalError = 0
    # 根据当前的w，b参数计算均方差损失
    for i in range(0, len(points)) : # 循环迭代所有点
        # 获得 i 号点的输入 x
        x = points[i, 0]
        # 获得 i 号点的输出 y
        y = points[i, 1]
        # 计算差的平方，并累加
        totalError += (y - (w * x + b)) ** 2
    # 将累加的误差求平均，得到均方误差
    return totalError / float(len(points))


# 计算偏导数
def step_gradient(b_current, w_current, points, lr) :
    # 计算误差函数在所有点上的异数，并更新w，b
    b_gradient = 0
    w_gradient = 0
    # 总体样本
    M = float(len(points))
    for i in range(0, len(points)) :
        x = points[i, 0]
        y = points[i, 1]
        # 偏b
        b_gradient += (2 / M) * ((w_current * x + b_current) - y)
        # 偏w
        w_gradient += (2 / M) * x * ((w_current * x + b_current) - y)
    # 根据梯度下降算法更新的 w',b',其中lr为学习率
    new_b = b_current - (lr * b_gradient)
    new_w = w_current - (lr * w_gradient)
    return [new_b, new_w]


# 梯度更新
def gradient_descent(points, starting_b, starting_w, lr, num_iterations) :
    b = starting_b
    w = starting_w
    MSE = []
    Epoch = []
    for step in range(num_iterations) :
        b, w = step_gradient(b, w, np.array(points), lr)
        # 计算当前的均方误差，用于监控训练进度
        loss = mse(b, w, points)
        MSE.append(loss)
        Epoch.append(step)
        if step % 50 == 0 :
            print(f"iteration:{step}, loss:{loss}, w:{w}, b:{b}")
    plt.plot(Epoch, MSE, color='C1', label='均方差')
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.title('MSE function')
    plt.legend(loc = 1)
    plt.show()
    return [b, w]

# 主函数
def solve(data) :
    # 学习率
    lr = 0.01
    initial_b = 0
    initial_w = 0
    num_iterations = 1000
    [b, w] = gradient_descent(data, initial_b, initial_w, lr, num_iterations)
    loss = mse(b, w, data)
    print(f'Final loss:{loss}, w{w}, b{b}')


if __name__ == "__main__":
    data = get_data()
    solve(data)