"""
参考rdflib手册：https://github.com/RDFLib/rdflib/blob/master/examples/
"""
from rdflib import Graph, Literal, BNode, Namespace, RDF, URIRef
from rdflib.namespace import DC, FOAF
import rdflib, time

print(__doc__)


def generate_rdf():
    g = Graph()
    # Create an identifier to use as the subject for Donna.
    donna = BNode()
    # Add triples using store's add method.
    g.add((donna, RDF.type, FOAF.Person))
    g.add((donna, FOAF.nick, Literal("donna", lang="foo")))
    g.add((donna, FOAF.name, Literal("Donna Fales")))
    g.add((donna, FOAF.mbox, URIRef("mailto:donna@example.org")))
    # Iterate over triples in store and print them out.
    print("--- printing raw triples ---")
    for s, p, o in g:
        print((s, p, o))

    # For each foaf:Person in the store print out its mbox property.
    print("--- printing mboxes ---")
    for person in g.subjects(RDF.type, FOAF.Person):
        for mbox in g.objects(person, FOAF.mbox):
            print(mbox)

    # Bind a few prefix, namespace pairs for more readable output
    g.bind("dc", DC)
    g.bind("foaf", FOAF)
    print(g.serialize(format='n3'))


def query_rdf():
    g = rdflib.Graph()
    g.load("./foaf.rdf")
    # the QueryProcessor knows the FOAF prefix from the graph
    # which in turn knows it from reading the RDF/XML file
    for row in g.query(
            'select ?s where { [] foaf:knows ?s .}'):
        print(row.s)
        # or row["s"]
        # or row[rdflib.Variable("s")]
    print('\n\n')

    for row in g:
        print(row)


def rdf_to_sql_relation():
    g = rdflib.Graph()
    g.load("./tcm_rdf_sample.rdf")

    # 将rdf转换为三元关系
    prefix = "http://zcy.ckcest.cn/tcm/"
    prefix_len = len(prefix)
    print('prefix_len: {}'.format(prefix_len))

    # 解析rdf文件，并将数据添加到字典{key，set(tuple)}中
    rdfdict = ({})
    resset = set()
    for row in g:
        # print(row)
        [s, p, o] = row
        s = s[prefix_len:]
        p = p[prefix_len:]
        if o.find(prefix) >= 0:
            o = o[prefix_len:]
        else:
            o = o[:]
        # print('{:20}, {:20}, {:20}'.format(s, p, o))
        if rdfdict.get(s):
            resset = rdfdict[s]
            resset = resset | ({(s, p, o)})
            rdfdict[s] = resset
        else:
            rdfdict[s] = ({(s, p, o)})
    # print('\n\nrdfdict')
    # print(rdfdict)

    # 导出关系数据
    for key in rdfdict.keys():
        # print(key)
        keyset = rdfdict[key]
        # print(keyset)
        for atuple in keyset:
            # print(atuple)
            print("'{}','{}','{}'".format(atuple[0], atuple[1], atuple[2]))


if __name__ == '__main__':
    start = time.time()

    # generate_rdf()
    # query_rdf()

    rdf_to_sql_relation()

    end = time.time()
    print('exit(0) total time: {}s'.format(end - start))
