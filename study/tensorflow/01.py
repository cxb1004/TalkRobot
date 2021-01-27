# 导入tensorflow
import os
# 降低TF的日志输出级别，只输出error/fatal
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


# 查看tensorflow的版本
print(tf.__version__)

"""
TensorFlow 使用张量（Tensor）作为数据的基本单位
张量在概念上等同于多维数组，我们可以使用它来描述数学中的标量（0 维数组）、向量（1 维数组）、矩阵（2 维数组）等各种量
"""
# 标量：单个数字
random_float = tf.random.uniform(shape=())
print(random_float)

# 向量：一维数字组成的对象。 如果所有元素都是0，称之为0向量
zero_vector = tf.zeros(shape=(2))
print(zero_vector)

# 定义两个2×2的常量矩阵
A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])
print(A)
print(B)

# 向量的属性：shape（图形）dtype（数据类型）numpy(数值)
zero_vector2 = tf.zeros(shape=(3, 2, 1))
print(zero_vector2.shape)
print(zero_vector2.dtype)
print(zero_vector2.numpy())
import tensorflow as tf

"""
指数计算
"""
# 使用GradientTape进行求导，例子使用的幂函数：y=x(2) 幂函数导数：y'=2x(2-1)  当x = 3的时候，y'=6
# 以下的代码可以这么理解，with创建了一个记录器tape，tape是一个幂函数的图形（一维）；
# 然后tape.gradient(y,x)就对这个图形进行求导（变化率）的计算
x = tf.Variable(initial_value=3.)
# 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
with tf.GradientTape() as tape:
    # 定义一个幂函数
    y = tf.square(x)
# 对幂函数求导
y_grad = tape.gradient(y, x)        # 计算y关于x的导数
print(y, y_grad)

"""
线性回归
"""