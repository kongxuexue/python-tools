# python3的字符串
import sys
import math


def comment():
    mycomment = '''测试换行
    注释'''
    print(mycomment)

    a = 'this'
    b = 'is\n'
    c = 'a string'
    d = a + \
        b + c
    print(a, b, c)
    print(d)
    print(r'自然字符串is\n', b)


def input_output():
    n = input("请输入一个数字，程序才能继续！n=")
    print('输入的内容为：', n)

    sys.stdout.write("使用stdout的write进行不换行输出ab" + 'c！！ ')
    print('紧接着上一行输出')

    # 让print不默认换行
    print('设置end参数以不换行！！ ', end='。。。')
    print('测试print的换行')


def string_substring():
    str = 'TEST:测试字符串的截取'
    print(str)
    print(str[0: 1])  # 包括左边但是不包括右边
    print(str[0: -1])
    print(str[-2: -1])
    print(str[-2:])
    print(str * 2)  # 输出两次


def stringFunc():
    a = "this is a String"
    if ("h" in a):
        print("h in %s" % a)
    print("%%我叫%s今年%+d岁!" % ('小明', 10))
    print('%-8s %-8s %-8s' % ('name', 'id', 'salary'))
    print('%-8s %-+8d %5.2f' % ('aa', 10, 12.123))
    print('%-8s %-+8d %5.2f' % ('xxxx', -10, 58.91))
    print('%-8s %-+8d %5.1f' % ('y', 0, 23.7895))

    print(str.capitalize(a))
    print(a.find('a'))
    print('%s %s %s' % (str.islower(a), str.lower(a), a.split(' ', 2)))

    # 填充常跟对齐一起使用
    # ^ 、 < 、 > 分别是居中、左对齐、右对齐，后面带宽度
    # :号后面带填充的字符，只能是一个字符，不指定的话默认是用空格填充
    print('step: {:<3}  loss: {:<6.6} acc: {:6.6}'.format(1, 2.12345678, 3.0))


def string_formate():
    print('{}网址： "{}!"'.format('菜鸟教程', 'www.runoob.com'))
    print('站点列表 {0}, {1}, 和 {other}。'.format('Google', 'Runoob', other='Taobao'))
    # '!a' (使用 ascii()), '!s' (使用 str()) 和 '!r' (使用 repr()) 可以用于在格式化某个值之前对其进行转化:
    print('常量 PI 的值近似为： {!r}。'.format(math.pi))
    # 可选项 ':' 和格式标识符可以跟着字段名
    print('常量 PI 的值近似为 {0:.3f}。'.format(math.pi))
    # 在 ':' 后传入一个整数, 可以保证该域至少有这么多的宽度。 用于美化表格时很有用
    table = {'Google': 1, 'Runoob': 2, 'Taobao': 3}
    for name, number in table.items():
        print('{0:10} ==> {1:10d}'.format(name, number))
    # 如果你有一个很长的格式化字符串, 而你不想将它们分开, 那么在格式化时通过变量名而非位置会是很好的事情。
    # 最简单的就是传入一个字典, 然后使用方括号 '[]' 来访问键值,也可以通过在 table 变量前使用 '**' 来实现相同的功能
    print('Runoob: {0[Runoob]:d}; Google: {0[Google]:d}; ' 'Taobao: {0[Taobao]:d}'.format(table))
    print('Runoob: {Runoob:d}; Google: {Google:d}; Taobao: {Taobao:d}'.format(**table))
    print('% 操作符也可以实现字符串格式化,但是属于旧式方法，最终会被 str.format() 替换')


def strToCharacter():
    astr = "不到长城非好汉@123.com"
    for word in astr:
        print(word, end=' ')
    print('可以对string进行字符级别的拆分')


def main_string():
    print('>>>>>> String >>>>>>')
    comment()
    input_output()
    string_substring()
    stringFunc()
    string_formate()
    strToCharacter()
    print('<<<<<< String <<<<<<\n')


if __name__ == '__main__':
    main_string()
