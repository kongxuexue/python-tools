# 函数

def fib(n=3, *tup):
    print("n=%d" % n)
    for var in tup:
        print(var, end=" ")
    print()


def noNameFunc():
    sumF = lambda a, b, c: a + b + c
    print(sumF(1, 2, 3))


def function_basic():
    fib(4, 5, 6)
    noNameFunc()


def main_function():
    print('>>>>>> function >>>>>>')
    function_basic()
    print('<<<<<< function <<<<<<\n')

print("这是默认主程序。本模块被调用的时候，会默认运行这些内容。自身调用本模块时，和'__main__'里面的按先后顺序运行")
print(dir())

if __name__ == '__main__':
    print("这是自身调用本模块时才会运行的主程序，被调用的时候不会运行该内容")
