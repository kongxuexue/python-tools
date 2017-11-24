'''
用os进行文件夹和文件的操作
'''

import os
import codecs


def checkFiles():
    if os.path.exists('./MNIST_data/'):
        print('data already exits!')
    else:
        print('not exits...')
    if os.path.isfile('./MNIST_data/'):
        print('目标是文件!也许不存在这样的路径')
    else:
        print('目标不是文件')
    os.makedirs('./MNIST_data/')
    if os.path.exists('./MNIST_data/'):
        print('data already exits!')
    list = os.listdir('./MNIST_data/')  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        path = os.path.join('./MNIST_data/', list[i])
    if os.path.isfile(path):
        pass
        print(path + ' is a file')
        print(str(os.path.getsize(path)/1024) + 'KB')  # 读取文件大小 KB


if __name__ == '__main__':
    checkFiles()
