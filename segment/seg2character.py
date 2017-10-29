"""
中文分词成一个一个汉字

"""
print(__doc__)
if __name__ == '__main__':
    line = "中国传统哲学是中国传统文化的脊梁，是中华民族博大精神的集中体现。结合中医学内容的实际需要，特选择学术界公认共识的基本哲学思想作如下阐释"
    for chara in line:
        print(chara, end=' ')
    print()
    print("/".join(line))
