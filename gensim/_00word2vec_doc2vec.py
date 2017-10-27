"""
gensim 提供强大的自然语言处理，包括LSI、LDA、TFIDF、word2vec、doc2vec等等
    [Gensim进阶教程：训练word2vec与doc2vec模型](http://www.cnblogs.com/iloveai/p/gensim_tutorial2.html)
"""
from gensim.models import word2vec
from gensim.models.doc2vec import TaggedDocument
import gensim
import os

print(__doc__)


def word2vec(corpus_path=None, size=50):
    """
     使用分好词的评论语料训练词向量
     :param corpus_path 分词后的语料路径
     :param size 词向量维度
    :return:
    """
    corpus_path = 'D:/data/qa_841k/qa_segemented.txt'
    model_path = 'data/qa_841k.word2vec.ginsim.mod'

    # 没有保存之前的模型就重新训练model
    if not os.path.exists(model_path):
        sentences = word2vec.Text8Corpus(corpus_path)
        model = word2vec.Word2Vec(sentences, size)
        dis = model.similarity('感冒', '发烧')
        print(dis)
        sims = model.most_similar(positive='感冒', topn=20)
        for item in sims:
            print(item)
        model.save(model_path)
    # 加载模型并查看效果
    savedModel = gensim.models.Word2Vec.load(model_path)
    print('感冒的词向量：', savedModel['感冒'])
    dis = savedModel.similarity('感冒', '发烧')
    print('词语相似度：', dis)


def doc2vec():
    texts = ['你好 ， 你 是 谁', '我 不知道 你 叫 什么']
    docs = [TaggedDocument(words=texts[i].strip().split(' '), tags=str(i)) for i in range(len(texts))]
    print(docs)

    # * 初始化训练一次
    model = gensim.models.Doc2Vec(docs)

    # * 不初始化，训练一次
    model = gensim.models.Doc2Vec()
    print('ok')
    model.build_vocab(sentences=docs)
    # doc2vec_model.train(sentences=docs)
    print(model)

    # * 迭代训练
    for epoch in range(10):
        model.train(docs)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no deca
        model.train(docs)
    print(model.docvecs[0])

    # 模型保存和加载
    fname = "data/00doc2vec.mod"
    # model.save(fname)
    model = gensim.models.Doc2Vec.load(fname)
    print(model.most_similar())


if __name__ == '__main__':
    print('='*40 + 'word2vec')
    word2vec()
    print('='*40 + 'doc2vec')
    doc2vec()
