"""
gensim的BM25摘要算法
    [Text Summarization with Gensim](https://rare-technologies.com/text-summarization-with-gensim/)
"""

import gensim
from gensim.summarization.bm25 import BM25
if __name__ == '__main__':

    text1 = "Thomas A. Anderson is a man living two lives. By day he is an " + \
            "average computer programmer and by night a hacker known as " + \
            "Neo. Neo has always questioned his reality, but the truth is " + \
            "far beyond his imagination. Neo finds himself targeted by the " + \
            "police when he is contacted by Morpheus, a legendary computer " + \
            "hacker branded a terrorist by the government. Morpheus awakens " + \
            "Neo to the real world, a ravaged wasteland where most of " + \
            "humanity have been captured . by a race of machines that live " + \
            "off of the humans' body heat . and electrochemical energy and " + \
            "who imprison their minds within an artificial reality known as " + \
            "the Matrix. As a rebel against the machines. Neo must return to " + \
            "the Matrix . and confront the agents: super-powerful computer . " + \
            "programs devoted to snuffing out Neo and the entire human " + \
            "rebellion. "
    text2 = "Google 公司（ 中文译名 ：谷歌 ） ， . 是 一家 美国 的 跨国 科技企业 ， . " \
            "致力于 互联网 搜索 、 云计算 、 广告 技术 等 领域 .  谷歌 的 使命 是 整合 全球 信息 . " \
            "谷歌 是 第一个 被 公认为 全球 最大的 搜索引擎. 谷歌 联合创始人 拉里·佩奇 和 谢尔盖·布林 " \
            "在 学生 宿舍 里 共同 开发了 全新的 在线 搜索引擎. 成立 数天后， 公司 注册了 Google.com 域名。 " \
            " 1999年6月7日， 谷歌 获得了 两家 风险 投资公司 的 投资"
    texts = [text1, text2]

    for text in texts:
        print(text + '\n' + '=' * 40 + ' 摘要生成：')

        # 摘要生成，ratio限制与原文的内容，word_count限制结果的最多字数，split输出list
        # 摘要生成就是选取最重要的几句话，而且是以英文的.作为一句话分割。关键词也是空格切分后的词语，与出现次数有关。
        summary = gensim.summarization.summarize(text, ratio=0.1, word_count=50, split=0)
        print(summary)

        print('=' * 40 + ' 关键词：')
        # 关键词生成，也有很多参数
        keywords = gensim.summarization.keywords(text)
        print(keywords)
        print()

    # bm25 算法
    '''
    BM25算法，通常用来作搜索相关性评分。其主要思想：
    首先，对Query进行语素解析，生成语素qi(相当于中文语句对应到字)。
    然后，对于每个文档doc，计算每个语素qi与doc的相关性得分（基于出现频次、文档长度等关系）。
    最后，将qi相对于D的相关性得分进行加权求和（常见idf），从而得到Query与D的相关性得分。
    '''
    for text in texts:
        bm = gensim.summarization.bm25.BM25(corpus=text)
        print(bm.corpus)
        print(bm.df)  # 都是每个字的得分
        print(bm.idf)
        print(bm.f)
        print(bm.avgdl) # 平均文档长度

