"""
Elasticsearch 高级查询示例
    * see for detail: https://elasticsearch-py.readthedocs.io/en/master/api.html
"""
from elasticsearch import Elasticsearch

if __name__ == '__main__':
    # 获取链接，默认链接localhost:9200
    print('=' * 20 + ' 获取链接')
    hosts = [{'host': '10.15.82.65', 'port': 9200}, {'host': '10.15.82.66', 'port': 9200}]  # hosts使用list[dict]
    es = Elasticsearch(hosts)
    print(es)

    print('=' * 20 + ' 查找doc')
    search = {
        "query": {
            "bool": {
                "must": {
                    "term": {
                        "author.name.keyword": "雨果"
                    }
                },
                "must_not": {
                    "term": {
                        "title.keyword": "《悲惨世界》"
                    }
                },
            }
        }
    }
    ret = es.search(index="test", doc_type="book", body=search)
    print(ret)
    for hit in ret['hits']['hits']:
        print(hit)

