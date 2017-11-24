# dictionary 字典
def dictionary_basic():
    empd = {}
    print(empd)
    da = {'name':1, 'code':'yes', 3.0:'test'}
    print(da)
    for key in da:
        print(key,end=': ')
        print(da[key])
    da[3.0] = 3+2j
    print(da)
    print(da.keys()) #输出所有key
    print(da.values()) # 输出所有value

    db = dict([('name',1),(3,2)])
    print(db)


def dictFunc():
    a = {1:'a',"b":1,(1,'1'):0.1}
    print(a)
    print(a.pop('b')) # 类似出栈
    print(a.get(1))
    print(a.items())


def main_dictionary():
    print('>>>>>> dictionary >>>>>>')
    dictionary_basic()
    dictFunc()
    print('<<<<<< dictionary <<<<<<\n')

if __name__ == '__main__':
    main_dictionary()