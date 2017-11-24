class MyClass():
    '''
    定义自己的类。类的方法需要第一个位置作为额外参数，惯例是叫self，指代类的实例
    双下划线开始的属性不能在类外部直接访问
    单下划线开始的属性也不能在类外部直接访问
    '''
    country = 'china' # 这里定义属性的话，每个类都会有该属性，并且可以设置默认值。如果不需要可以直接在init中设置，不需在此定义
    __weight = '60kg'  # 双下划线开始的属性不能在类外部直接访问

    def __init__(self, name, age):
        # init是初始化方法（构造方法）
        self.name = name
        self.age = age

    def printSelf(self):
        print(self)
        print(self.__class__)

    def setNo(self, no):
        self.no = no

    def show(self):
        print("country:{} name:{} age:{} no:{} weight:{}".format(self.country, self.name, self.age, self.no,
                                                                 self.__weight))


class human():
    __age = 23
    sex = 'unknown'

    def __init__(self, sex):
        self.sex = sex

    def showHuman(self):
        print("age:{} sex:{}".format(self.__age, self.sex))


class student(MyClass, human):
    '''
    支持多继承，如果方法在子类没找到，而多个父类有同名方法，会依据括号中顺序从左向右搜索
    '''
    stuID = -1
    __score = -1000
    _star = 1

    def __init__(self, stuId, name, age, sex, no):
        # self.stuID = stuId
        MyClass.__init__(self, name, age)
        MyClass.setNo(self, no)
        human.__init__(self, sex)

    def showStudent(self):
        print('stuID:{} name:{} age:{} sex:{}'.format(self.stuID, self.name, self.age, self.sex))
        self.__getscore()
        return self.name, self.age

    def __getscore(self):
        print('得分：%d' % self.__score)

    @property # 将方法设置为属性，常用于一些需要预处理的属性
    def full_name(self):
        return '{}_{}'.format(self.name,self.age)

    _star = property(fget=full_name,fset=full_name) # 属性还有一些更加高级的用法

def class_basic():
    tom = MyClass('tom', 10)
    tom.setNo(0)
    tom.show()
    tom.printSelf()
    # print(tom.__weight)
    jack = human('nan')
    jack.showHuman()

    will = student(100, 'will', 20, 'nv', 22)
    will.show()
    will.showHuman()
    will.showStudent()
    name, age = will.showStudent()
    print('return 返回的值为：{}, {}'.format(name, age))
    # will.__getscore()
    print(will.full_name)

def main_class():
    print('\n\n>>>>>> class >>>>>>')
    class_basic()
    print('<<<<<< class <<<<<<\n')

if __name__ == '__main__':
    main_class()
