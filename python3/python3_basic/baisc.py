# -*- coding:utf-8 -*-
# py3默认使用utf-8编码文件，字符串都是Unicode编码
"""
python的基本语法
    print(__doc__)的内容最好用三个双引号。
    文件名/目录小写加下划线，类名建议大写，方法名小写，使用下划线
"""
from __future__ import division
import keyword
import myString
import myNumber
import myList
import myTuple
import mySet
import myDictionary

import myControl
import myIteratorAndGenerator
import myFunction

import my_thread
import myLock
import myClass
import platform

print(__doc__)
print('注意import初始化、本包默认主函数、main主函数的运行先后顺序', '注意main函数只在当前文件为主运行文件时才运行')
print('python 版本为：%s ' % platform.python_version())
print('__future__是为了版本兼容和移植，比如想将py2工程陆续改成py3'
      , '可以先改所有的整除模块，py2中10/3是整除=3，但是py3的整除是10//3，那就从__future__导入division，10/3都改成10//3。'
      , '当工程都改用了对应的future模块就成为py3工程了')

if __name__ == '__main__':
    print('>>>>>> main >>>>>>')
    print('python3的保留字有：', keyword.kwlist)
    myString.main_string()
    myNumber.main_numbers()
    myList.main_list()
    myTuple.main_tuple()
    mySet.main_set()
    myDictionary.main_dictionary()

    myControl.main_control()
    myIteratorAndGenerator.main_iter_gene()
    myFunction.main_function()

    my_thread.main_thread()
    myLock.main_lock()
    myClass.main_class()
