'''
文件读写操作
'''
import time,os
from os.path import join, getsize

if __name__ == '__main__':
    rootdir = 'F:\\qa\\20170411\\丁香园\\raw'
    for root, dirs, files in os.walk(rootdir):
        print(root, "consumes", end="")
        print(sum([getsize(join(root, name)) for name in files]), end="")
        print("bytes in", len(files), "non-directory files")
        if 'CVS' in dirs:
            dirs.remove('CVS')  # don't visit CVS directories