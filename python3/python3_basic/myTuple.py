# tuple 元组

def tuple_basic():
    '''
    元组基本操作：定义、截取等
    :return:
    '''
    ta = ('a', 1, "STR", 0.1J)
    tb = ('b', 10)
    print(ta)
    print(ta[0:-2])
    print(tb + ta)
    print(tb * 2)
    print(ta[1:])
    list = [1, 2]
    tc = (list, 'c')  # tuple元素不可变，但是可包含可变的对象
    print(tc)

    empt = ()
    onet = ('a',)
    print(empt + onet)
    print(onet * 2)


def main_tuple():
    print('>>>>>> tuple >>>>>>')
    tuple_basic()
    print('<<<<<< tuple <<<<<<\n')
