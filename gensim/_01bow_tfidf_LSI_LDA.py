"""
README
* function
    * 使用 gensim 计算文档的bow向量和tf-idf向量
    * 使用 gensim 构建LSI主题向量和模型，计算文档相似度
* reference
    * [Gensim入门教程](http://www.cnblogs.com/iloveai/p/gensim_tutorial.html)
    * [gensim API](https://radimrehurek.com/gensim/apiref.html)
"""
import gensim

print(__doc__)

if __name__ == '__main__':
    print('=' * 20 + '\nbow向量和tfidf向量的第一个值是词语在词典中的id，第二个值是bow出现次数或者tfidf的值' + '=' * 20)
    # 每个文档含有多个词汇组成
    docs = [['human', 'human', 'computer', 'interface', 'computer'],
            ['survey', 'user', 'computer', 'system', 'response', 'time'],
            ['eps', 'user', 'interface', 'system'],
            ['system', 'human', 'system', 'eps'],
            ['我们', '人民', '人民'],
            ['trees', 'trees'],
            ['graph', 'trees'],
            ['graph', 'minors', 'trees'],
            ['graph', 'minors', 'survey']]
    test_doc = ['你是谁', '我们', 'human', '人民']

    # * 在全部文档上构建每个word的索引词典
    dic = gensim.corpora.Dictionary(docs)

    # * bow采用的是词语出现次数
    print('=' * 20 + '文档集转化为bow向量：' + '=' * 20)
    bows = [dic.doc2bow(doc) for doc in docs]
    corpus = bows
    print(corpus)

    # * 构建bow向量的tfidf模型
    print('=' * 20 + '训练文档的bow和tfidf：' + '=' * 20)
    tfidfModel = gensim.models.TfidfModel(corpus)
    print(corpus[0])
    print(tfidfModel[corpus[0]])

    # * 对新的文档，需要特征在文档集中，否则完全没有作用、没有影响
    print('=' * 20 + '测试文档的bow和tfidf：' + '=' * 20)
    new_bow = dic.doc2bow(test_doc)
    print(new_bow)
    print(tfidfModel[new_bow])

    # * 将tfidf模型保存和加载使用
    print('=' * 20 + 'tfidf模型的保存和加载：' + '=' * 20)
    modelName = './01tfidf.mod'
    tfidfModel.save(modelName)
    tfidfModel = gensim.models.TfidfModel.load(modelName)
    print('模型存储为：' + modelName)
    for num_topics in [2, 3, 4, 5, 6, 7, 8]:  # 不同的核数对主题向量有影响，比如7效果不错
        # * 构造LSI主题模型，计算出目标文档与文档集之间的相似度
        print('=' * 20 + '基于文档集构建LSI主题向量模型，计算测试文档的LSI向量，然后计算相似度：' + '=' * 20)
        lsi_model = gensim.models.LsiModel(corpus, num_topics=num_topics, id2word=dic)
        # 获取文档集的lsi主题向量和测试文档的主题向量
        lsi_docs_corpus = lsi_model[corpus]
        lsi_new_doc = lsi_model[new_bow]
        print(lsi_docs_corpus)
        print(lsi_new_doc)
        # 构建文档集的相似度计算向量，并且会根据文档集自动构建并保存index索引文件，该文件和save()保存不一样！
        fname = '01文档集LSI相似性.mod'
        similar = gensim.similarities.Similarity('01文档集LSI索引', corpus=lsi_docs_corpus, num_features=len(dic))
        # Similarity模型的保存和加载
        similar.save(fname)
        similar = gensim.similarities.Similarity.load(fname)
        sims = similar[lsi_new_doc]  # 保存的模型会有精度损失，导致测试集和文档集的向量都有变化
        print(sims)
        print(sims.argmax())  # 相似度最高的文档
        print(sims.max())  # 相似度最高值

        # 构造LDA主题模型
        print('=' * 20 + '基于文档集构建LDA主题向量模型，计算测试文档的LDA向量，然后计算相似度：' + '=' * 20)
        lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dic)
        lda_docs_corpus = lda_model[corpus]
        lda_new_doc = lda_model[new_bow]
        print(lda_docs_corpus)
        print(lda_new_doc)
        fname = '01文档集LDA相似性.mod'
        similar = gensim.similarities.Similarity('01文档集LSI索引', corpus=lda_docs_corpus, num_features=len(dic))
        # Similarity模型的保存和加载
        similar.save(fname)
        similar = gensim.similarities.Similarity.load(fname)
        sims = similar[lsi_new_doc]  # 保存的模型会有精度损失，导致测试集和文档集的向量都有变化
        print(sims)
        print(sims.argmax())  # 相似度最高的文档
        print(sims.max())  # 相似度最高值
