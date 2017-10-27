"""
Elasticsearch for Python:
    * see for detail: https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/index.html
"""
from elasticsearch import Elasticsearch
from datetime import datetime

print(__doc__)
if __name__ == '__main__':
    # 获取链接，默认链接localhost:9200
    print('=' * 20 + ' 获取链接')
    hosts = [{'host': '10.15.82.65', 'port': 9200}, {'host': '10.15.82.66', 'port': 9200}]  # hosts使用list[dict]
    es = Elasticsearch(hosts)
    print(es.count)

    # 添加doc
    print('=' * 20 + ' 添加doc。这里添加两个doc用于测试查询')
    doc = {
        "title": "《悲惨世界》",
        "body": "这是一个喜剧，虽然主人公冉阿让最后死了，但是他完成了自我救赎，让孩子们获得了幸福。",
        "author": {
            "name": "雨果",
            "country": "French"
        },
        "time": datetime.now()
    }
    ret = es.index(index="test", doc_type="book", id=42, body=doc)
    print(ret)

    doc = {
        "title": "《悲惨世界 名著》",
        "body": "这是一个喜剧，虽然主人公冉阿让最后死了，但是他完成了自我救赎，让孩子们获得了幸福。",
        "author": {
            "name": "雨果",
            "country": "French"
        },
        "time": datetime.now()
    }
    ret = es.index(index="test", doc_type="book", body=doc)
    print(ret)

    # 修改doc，如果对应的id的doc已经存在，那么进行index()添加的时候，会自动覆盖，返回的状态是 'result': 'created'
    print('=' * 20 + ' 修改doc')
    doc = {
        "title": "《悲惨世界》",
        "body": "世界著名长篇小说。这是一个喜剧，虽然主人公冉阿让最后死了，但是他完成了自我救赎，让孩子们获得了幸福。",
        "author": {
            "name": "雨果",
            "country": "French"
        },
        "time": datetime.now()
    }
    ret = es.index(index="test", doc_type="book", id=42, body=doc)
    print(ret)
    print(ret['result'])

    # 删除doc
    print('=' * 20 + ' 删除doc。这里先新增一个doc并获取id，根据id删除。也可以delete_by_query()等')
    ret = es.index(index="test", doc_type="book", body=doc)
    print(ret['_id'])
    ret = es.delete(index="test", doc_type="book", id=ret['_id'])
    print(ret)

    # 查找doc
    print('=' * 20 + ' 查找doc，最简单的就是直接用id进行get，复杂些的见其他示例')
    ret = es.get(index="test", doc_type="book", id=42)
    print(ret)

