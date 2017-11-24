'''
* 久久问医网进行爬取数据：
 http://ask.9939.com/id/1 的1一直到8000000，选择title，body和reply，只选第一个且有答案的
 使用多线程进行获取数据
'''
print(__doc__)
import threading, os, requests, time

from bs4 import BeautifulSoup


def get_html(url):
    '''
    根据url获取html源码
    :param url:
    :return:
    '''
    # print(url)
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, sdch',
        'Accept-Language': 'zh-CN,zh;q=0.8,en;q=0.6',
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.87 Safari/537.36'
    }
    html = requests.get(url, headers=headers, timeout=15000)
    # html.encoding = 'utf8'
    # html.encoding = 'gb2312'
    return html.text


def vpn_disconnect(timeForOtherThread=60):
    print('当前线程sleep {}秒，以等待其他线程都sleep'.format(timeForOtherThread))
    print('使用os.open调用cmd模块')
    b = os.popen('rasdial VPNckcest6 /disconnect')
    print(b.read())
    time.sleep(3)
    c = os.popen('rasdial VPN ckc med ')
    print(c.read())


def parse_html(start, end, offset):
    file = open('../../files/qa_crawler20170328/9939/data' + str(start) + '-' + str(end) + '.txt', 'a+',
                encoding="utf-8")
    for i in range(offset, end, 1):
        url = 'http://ask.9939.com/id/' + str(i)
        html = get_html(url)
        if html.find('1.您要访问的网页不存在或已被删除。') > 0:
            print('{} : {}'.format(i, 'no data'))
            i = i - 1
            continue
        bs = BeautifulSoup(html, "html.parser")  # 创建BeautifulSoup对象
        try:
            # 只爬取有完整的问题、问题描述、回答的页面
            title = bs.body.find('div', {'class': 'inqur'}).findChild().get_text()
            body = bs.body.find('div', {'class': 'inqur'}).find('div', {'class': 'descip'}).get_text()
            reply = bs.body.find('div', {'class': 'adoctor'}).find('div', {'class': 'descip'}).get_text()
            title = str(title).replace('\n', '')
            body = str(body).replace('\n', '  ').replace('\r', '  ')
            reply = str(reply).replace('\n', '  ')
            # print('{} : {}: {}: {}'.format(i, title,body,reply))
            file.write('###' + str(i) + '\r###' + title + '\r###' + body + '\r###' + reply + "\r")
        except (RuntimeError, TypeError, NameError, AttributeError) as e:
            print('出错页面为{}，原因为：{}'.format(i, e))
            i = i - 1
    file.close()


class Job(threading.Thread):
    def __init__(self, mystart, end, offset):
        threading.Thread.__init__(self)
        self.mystart = mystart
        self.end = end
        self.offset = offset

    def run(self):
        print('开始新的线程，目标从{}到{}'.format(self.mystart, self.end))
        parse_html(self.mystart, self.end, self.offset)
        print('线程处理完毕，目标从{}到{}'.format(self.mystart, self.end))


if __name__ == '__main__':
    print("starting...")
    threadSize = 5  # 基本也就支撑6条
    start = 1
    total = 8000000
    step = total // threadSize
    list = [36207, 1716207, 3254821, 4833684, 6410459]  # 由于被拉黑，需要调整开始值
    for i in range(threadSize):
        item = list[i]
        job = Job(start, start + step, item + 1)
        job.start()
        start += step
    while 1:
        pass
    print('process over!!!')
