"""
处理 MongoDB
"""

import pymongo, os


def update(collection):
    pass
    # # $set用来指定一个键并更新键值，若键不存在并创建。
    # collection.update({'temple_name': '潭柘寺'}, {"$set": {"TEL": "666666", "Password": "123"}})
    # print(collection.find_one({'temple_name': '潭柘寺'}))
    # # 使用修改器$unset时，不论对目标键使用1、0、-1或者具体的字符串等都是可以删除该目标键。
    # collection.update({'temple_name': '潭柘寺'}, {"$unset": {"TEL": "", "Password": ""}})


if __name__ == '__main__':

    pass
    client = pymongo.MongoClient('10.15.82.50', 27017)
    adr_db = client.adr
    # adr_db.drop_collection('temple_detail') # 删除collection
    adr_collection = adr_db.adr_total_copy
    print(adr_collection.count())  # 总数

    dbSet = set()
    # retrieve
    # for doc in adr_collection.find():  # 查找所有
    #     # print(doc['adrId'])
    #     dbSet.add(str(doc['adrId']))
    # print(adr_collection.find_one({'adrId': 'adr_0'}))  # 查找一个

    for doc in adr_collection.find():
        id = doc['adrId']
        #adr_collection.update({'adrId': id}, {"$set": {"path": "不良反应期刊全文/"+id+'.pdf'}})
        yearqi = doc['publishDate']
        qi = yearqi.split('年')[1].replace('期','')
        print(qi)
        if int(qi)>0:
            pass
    print("over!")
