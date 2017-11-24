# set 集合

def set_basic():
    sa = ({1, 'a', 3, 3, 3, 30})
    print(sa)  # 自动删除重复
    emps = set()  # 空集合
    print(emps)
    sb = set('a')# set只能创建set()或者set(一个元素)

    print(sa | sb)
    print(sa - sb)
    print(sa & sb)
    print(sa ^ sb)

    if 'a' in sa:
        print('a在sa中：True')
    else:
        print('False')


def main_set():
    print('>>>>>> set >>>>>>')
    set_basic()
    print('<<<<<< set <<<<<<\n')
