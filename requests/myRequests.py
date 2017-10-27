'''
use requests to get http requests
'''
print(__doc__)

import requests,time

def get_cotent(url, data = None):
    header = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Content-Type':'application/json;charset=UTF-8',
        'Accept-Encoding': 'gzip, deflate, sdch',
        'Accept-Language': 'zh-CN,zh;q=0.8',
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.235'
    }
    ret = requests.get(url, headers=header, timeout=5)
    ret.encoding = 'utf-8'
    print(ret.text)
    return ret.text

if __name__ == '__main__':
    url = 'http://localhost:8080/tcm/qaos//tag/newtagsapi'
    url = 'http://localhost:8080/tcm/qaos//tag/newtags'
    url = 'http://zcy.ckcest.cn/tcm/qaos/tag/newtags'

    get_cotent(url)
