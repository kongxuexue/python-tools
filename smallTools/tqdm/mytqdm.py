"""
使用tqdm进行展示进度条
    tqdm的github：<https://github.com/tqdm/tqdm>
"""
from tqdm import tqdm
import time


def easy_tqdm():
    mylist = [x for x in range(50)]
    for _ in tqdm(mylist,desc="多次循环，会显示迭代次数如1/2、已经总耗时/剩余总耗时、当前轮耗时"):
        pass
        time.sleep(0.1)

    for _ in tqdm(range(20),desc="到20"):
        time.sleep(0.5)


if __name__ == '__main__':
    easy_tqdm()
