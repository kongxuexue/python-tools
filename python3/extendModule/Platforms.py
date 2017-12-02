"""
操作系统平台、版本和架构等
"""
import platform


def getOs():
    print('操作系统 ' + platform.system())
    print('操作系统版本号 ' + platform.version())
    print('系统架构 ' + str(platform.architecture()))
    print('cpu: ' + platform.machine())


if __name__ == '__main__':
    getOs()
    print('over!')
