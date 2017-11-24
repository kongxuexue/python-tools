# iterator and generator 迭代器与生成器
def iterator_generator_basic():
    a = [1, 'a', 1.j]
    it = iter(a)
    while True:
        try:
            print(next(it))
        except StopIteration:
            break


def fibonacci(n=3):  # 生成器函数 - 斐波那契，指定参数默认值
    a, b, counter = 0, 1, 0
    while True:
        if (counter > n):
            return
        yield a
        a, b = b, a + b
        counter += 1


def genetator():
    f = fibonacci(10)  # f 是一个迭代器，由生成器返回生成

    while True:
        try:
            print(next(f), end=" ")
        except StopIteration:
            print()
            break


def main_iter_gene():
    print('>>>>>> iterator and generator >>>>>>')
    iterator_generator_basic()
    genetator()
    print('<<<<<< iterator and generator <<<<<<\n')
