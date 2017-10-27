"""
    * 使用neo4j-driver进行neo4j的接口驱动
    * 使用Neo4j的查询语言‘Cypher’进行数据库查询 [Cypher](http://neo4j.com/docs/developer-manual/current/get-started/cypher/)
"""
from neo4j.v1 import GraphDatabase, basic_auth

if __name__ == '__main__':
    # 页面访问端口7474，ssl访问端口7687
    driver = GraphDatabase.driver("bolt://10.15.82.64:7687", auth=basic_auth("neo4j", "cadalTCM220"))
    session = driver.session()

    # session.run("CREATE (a:Person {name: {name}, title: {title}})",{"name": "Arthur", "title": "King"})

    result = session.run("match (a) return a limit 20")  # match all nodes
    result = session.run("match (a:别名) return a limit 20")  # match all nodes with label 别名
    result = session.run("match (a:饮片 {中文名: '麻黄'}) return a ")  # 中文名是麻黄，且含有标签是饮片的数据
    result = session.run("match (a {中文名: '麻黄'})--(b:别名) return a,b limit 20")  # a中文名是麻黄，b有标签别名，关系无方向
    result = session.run("match (a {中文名: '麻黄'})-->(b:别名) return a,b limit 20")  # 关系是炮制，一层匿名关系，且关系有方向
    result = session.run("match (a {中文名: '麻黄'})<--(b:别名) return a,b limit 20")  # 关系是炮制，一层匿名关系，且关系有方向
    result = session.run("match (a {中文名: '麻黄'})-[relation]-(b:别名) return a,b,type(relation) limit 20")  # 关系无方向，且返回关系
    result = session.run("match (a {中文名: '麻黄'})-[r {可信度:1}]-(b:别名) return a,b,type(r) limit 20")  # 限制关系的属性

    for record in result:
        print(record)

    session.close()
