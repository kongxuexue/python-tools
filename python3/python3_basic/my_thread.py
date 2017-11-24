import _thread
import threading
import time


# 为线程定义一个函数
def print_time(threadName, delay, counter):
    count = 0
    while count < counter:
        time.sleep(delay)
        count += 1
        print("%s: %s : %d" % (threadName, time.ctime(time.time()), count))


def thread_basic():
    # 创建两个线程
    try:
        _thread.start_new_thread(print_time, ("Thread-1", 1, 5))
        _thread.start_new_thread(print_time, ("Thread-2", 2, 5))
    except:
        print("Error: 无法启动线程")
    # 如果不使用线程或者循环，而直接运行到程序退出的话会导致新线程退出
    time.sleep(20)  # 30秒，给子线程一些运行时间
    # while 1:
    #    pass


def main_thread():
    print('\n\n>>>>>> _thread >>>>>>')
    a = input('是否运行多线程测试，输入1表示运行，否则不运行，请输入：')
    if a==1:
        thread_basic()
        print('开始Threading模块 >>>>>>')
        threading_basic()
    print('<<<<<< _thread <<<<<<\n')


exitFlag = True


def threading_basic():
    pass
    global exitFlag  # 默认使用全局变量，但是内部修改不会影响全局变量，真想改变全局变量就用global
    print(exitFlag)
    time.sleep(3)
    thread3 = myThread('mythread-3', 3, 10)
    thread3.start()  # start方法运行线程的run()
    exitFlag = True
    thread4 = myThread('mythread-4', 4, 7)
    thread4.start()
    if exitFlag:
        print('全局变量已经修改，在外围将强制退出线程：{}但是不知道线程如何退出'.format(thread4.threadName))
        thread3.join()
        thread4.join()


class myThread(threading.Thread):
    def __init__(self, threadName, delay, counter):
        threading.Thread.__init__(self)
        self.threadName = threadName
        self.delay = delay
        self.counter = counter

    def run(self):
        print("开始线程：%s" % self.threadName)
        print_time(self.threadName, self.delay, self.counter)
        print("线程结束：{}".format(self.threadName))
