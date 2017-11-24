# 基本数据类型
import math


def numbers():
    a = 100  # 整数
    b = 1.0  # 浮点型
    c = "string"  # string
    a1 = a2 = a3 = a  # 多变量相同赋值
    aa, bb, cc = a, b, c  # 多变量对应赋值
    aa = 0.1 + 2J
    print(aa)
    del a2, a3, bb, cc
    print(type(a1))
    print(type(aa) == int)
    print(True + 1)
    # print(a2)

    print(5 + 4, 4.3 - 2, 3 * 7, 2 / 4, 2 // 4, 10 % 3, 2 ** 5)  # 进行加减乘除，整除，取余，幂运算


def mathFunc():
    a = 0.05
    print(abs(a))
    print(math.floor(a))
    print(math.cos(a * math.pi * 10))
    print(math.cos(0))


def main_numbers():
    print('>>>>>> data >>>>>>')
    numbers()
    mathFunc()
    print('<<<<<< data <<<<<<\n')
