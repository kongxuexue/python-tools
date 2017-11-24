# control block

def control_basic():
    a, b = 1, 1
    print('斐波那契： %d' % (a), end=",")
    for i in range(10):
        print(b, end=',')
        a, b = b, a + b
    else:
        print(b)

    if 1 or False:
        print("支持01表示True False")
    else:
        print("不支持01表示True False")

    for i in range(-5, 3, 2):
        print("%d" % (i), end=", ")
    else:
        pass
        print(i)


def main_control():
    print('>>>>>> control >>>>>>')
    control_basic()
    print('<<<<<< control <<<<<<\n')
