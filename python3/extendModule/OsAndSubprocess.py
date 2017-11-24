import os
import subprocess,signal

def useCMD():
    '''
    当前进程调用cmd命令，属于阻塞式
    :return:
    '''
    print('一般os.system运行后不返回值')
    print(os.system("ipconfig"))

    print('使用os.open可以获取返回值')
    b = os.popen('ipconfig')
    print(b.read())

def my_subprocess():
    print('产生子进程运行命令，属于阻塞式')
    cmd = 'ping www.baidu.com'
    a = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    print(a.args)
    a.send_signal(signal.CTRL_C_EVENT)
    print(a.returncode)

if __name__ == '__main__':
    print('>>>>>>  os ')
    useCMD()
    my_subprocess()
    print('<<<<<<  os ')