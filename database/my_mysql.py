'''py 3.5使用pymysql连接mysql'''
# encoding:utf-8
import pymysql

def insertData(cursor, db):
    f = open('D:/qa/qa20161216tagCategory.txt', 'r+', encoding='utf-8')
    # con  = f.read()
    # print(con)
    while 1:
        line1 = f.readline()
        line2 = f.readline()
        if not line1:
            break
        category = line1.split('	')[1]
        tag = line2.split('	')[1].replace('\n', '')
        print('tag:cate: ', tag, category)

        try:
            sql = '''INSERT INTO qa_tag_category (`tag_name`, `tag_category` )
                          VALUES ('%s', '%d');
                  '''
            cursor.execute(sql % (tag, int(category)))
            db.commit()
        except:
            db.rollback()

def crud(cursor):
    '''
    DML：增查改删
    :param cursor: 游标
    :return:
    '''
    # SQL 查询语句
    sql = "SELECT * FROM EMPLOYEE \
           WHERE INCOME > '%d'" % (1000)
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        results = cursor.fetchall()
        for row in results:
            fname = row[0]
            lname = row[1]
            age = row[2]
            sex = row[3]
            income = row[4]
            # 打印结果
            print("fname=%s,lname=%s,age=%d,sex=%s,income=%d" % \
                  (fname, lname, age, sex, income))
    except:
        print("Error: unable to fecth data")

    # SQL 更新语句
    sql = "UPDATE EMPLOYEE SET AGE = AGE + 1 WHERE SEX = '%c'" % ('M')
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 提交到数据库执行
        db.commit()
    except:
        # 发生错误时回滚
        db.rollback()

    # SQL 删除语句
    sql = "DELETE FROM EMPLOYEE WHERE AGE > '%d'" % (20)
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 提交修改
        db.commit()
    except:
        # 发生错误时回滚
        db.rollback()

def tableAlter(cursor):
    # 使用 execute() 方法执行 SQL，如果表存在则删除
    cursor.execute("DROP TABLE IF EXISTS EMPLOYEE")
    # 使用预处理语句创建表
    sql = """CREATE TABLE EMPLOYEE (
             FIRST_NAME  CHAR(20) NOT NULL,
             LAST_NAME  CHAR(20),
             AGE INT,
             SEX CHAR(1),
             INCOME FLOAT )"""
    cursor.execute(sql)


if __name__ == '__main__':
    print('\033[1;31;40m')  # \033[默认等级;前景色;背景色;m
    print('start...')
    print('\033[0m')  # 使用默认输出样式
    # 打开数据库连接
    db = pymysql.connect('127.0.0.1', 'root', '123', db='tcm', charset="utf8")
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    ''''''
    crud(cursor)
    # insertData(cursor)
    ''''''
    # 关闭数据库连接
    db.close()
    print('\033[1;31;40m')
    print('over ' * 3)
    print('\033[0m')
