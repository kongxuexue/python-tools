# list 列表类型

def list_def():
    a = [1, 2.0, 0.3j, "test"]
    b = ['b', 10]
    print(a)
    print(a[:-1])
    print(b + a)
    print(b * 2)
    a[0] = "a"
    print(a)

def listFunc():
    a = [[1,2], 'c','c']
    print(a)
    print(a.count('c'))
    a.reverse()
    print(a) # reverse 不能返回原先的引用，没法直接输出a.reverse()

def main_list():
    print('>>>>>> list >>>>>>')
    list_def()
    listFunc()
    print('<<<<<< list <<<<<<\n')
