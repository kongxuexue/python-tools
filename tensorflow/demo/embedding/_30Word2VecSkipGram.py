# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import math
import os
import random
import zipfile

import numpy as np
# from six.moves import urllib # python2
import requests
# from six.moves import xrange  # pylint: disable=redefined-builtin # python2
import tensorflow as tf

'''
Word2Vec：CBOW模型/Skip-Gram模型
* CBOW continuous bag of words：连续词袋模型
    根据源词汇的上下文词汇my name is 预测目标词如Tom。CBOW有平滑处理，如将一整段上下文视为一个观察量，对于大数据集很有效。
    训练时候，对p(target_i|context)进行预测，对p使用softmax，再进行log对数似然训练，思路可以，但是每次训练i个词太浪费，使用二分类器（逻辑回归），
    从指定的k个虚构的噪声词中选取出是不是目标词。通过对噪声词loss的对数最大似然，将目标词概率最大，噪声词概率最小，属于一个log回归（逻辑回归）
* Skip-Gram
    使用目标词预测源词汇。数据集本来窗口如3，目标词name的上下文是my is，skip-gram交换目标词和上下文，变成name预测my、is。
    SG将（上下文词汇，目标词汇）视为观察量，对于小型数据集更为有效，效果更好。tf只有sg模型的实现。
'''

'''
向量空间模型VSMs多数使用分布式假设：拥有相似上下文情景的词汇具有相类似的语义。常用两种方法：基于统计方法、基于预测方法。
* 基于统计方法：统计某词汇与其邻近词汇在一个大型语料库中共同出现的频率及其他统计量，再将这些统计量映射到一个小型且稠密的向量中。
* 基于预测方法：直接从某词汇的邻近词汇对其进行预测，在此过程中学习到小型且稠密的嵌套向量。
'''

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    '''
    自动下载训练数据，并检测文件大小
    :param filename: 文件名
    :param expected_bytes: 文件大小
    :return:
    '''
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        # filename, _ = urllib.request.urlretrieve(url + filename, filename)
        filename, _ = requests.get(url + filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

'''
    text8.zip数据只有一个文件，且只有一行英文语料，内容为：
    anarchism originated as a term of abuse first used against early working ...
'''
filename = maybe_download('text8.zip', 31344016)

# Read the data into a list of strings.
def read_data(filename):
    '''
    读取数据集，并按照空白字符split
    :param filename:
    :return:
    '''
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


vocabulary = read_data(filename)
print('Data size', len(vocabulary)) # 词汇一共17005207个

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000


def build_dataset(words, n_words):
    '''
    根据数据集创建训练数据，去掉一些unk/unknown的词，如低频次。
    选取最常见的n_words个单词
    :param words:
    :param n_words:
    :return:
    '''
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1)) # count存放最常见的n个词和词频
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

# data是用词典的index表示词语，count是所有词频统计，dictionary是高频词词典：词/index，reverse_dict是逆词典：index/词
data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)

del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    '''
    生成训练batch
    :param batch_size:
    :param num_skips:
    :param skip_window:
    :return:
    '''
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ] # span大小是左右两个skip和中间一个target
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index]) # 读取5个词
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

# 构造数据，generate_batch返回batch_size个目标词和源词语，batch是一维目标词，label是二维源词，第一维是batch，第二维是源词
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

embedding_size = 128  # Dimension of the embedding vector. #设置embedding矩阵的大小
batch_size = 128 # 每次生成多少组（目标词与源词）
skip_window = 1  # How many words to consider left and right. # 设置左右的源词窗口大小（即限定源词范围）
num_skips = 2  # How many times to reuse an input to generate a label. # 设置每个目标词的源词个数（从源词窗口中选定）

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
# 随机选取一些小id的词语验证knn词语相似性（字典中id越小越高频）
valid_size = 16  # Random set of words to evaluate similarity on. # 验证valid_size个词语的相似结果
valid_window = 100  # Only pick dev samples in the head of the distribution. # 设置验证的词语的范围
valid_examples = np.random.choice(valid_window, valid_size, replace=False) # 从0-valid_window中随机选取valid_size个词语进行验证，词语是index表示
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph() # tf默认图

with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation # embedding_lookup缺乏GPU实现？
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)) # embedding矩阵变量
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        # embedding的查询就是选取对应的行，embedding本身就是很多行参数，每行参数可以作为词向量

        # Construct the variables for the NCE loss
        # noise-contrastive estimation (NCE) 噪声对比训练,噪声-比对的损失计算使用一个逻辑回归模型，使用一个w和b
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    '''
    计算 NCE 损失函数, 每次使用负标签的样本，计算整个batch的平均nce loss作为loss
        NCE 噪声对比训练避免使用全部的语料词汇，只需分类到指定的num_classes个类中，本身是一个y=wx+b的多分类，
    '''
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset) # 验证集对应的embedding构成的矩阵
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True) # 相乘得到验证集和所有词语的cos值

    # Add variable initializer.
    init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    for step in range(num_steps):
        # 不断生成目标词和源词数据进行训练，使用global data_index向后生成训练集
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]] # 得到选取的要验证的词语
                top_k = 8  # number of nearest neighbors # 每个验证词选取top8相似词
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]  # 由词向量cos值递减排序，并选取除本身外的topK
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()


# Step 6: Visualize the embeddings.
# 可视化展示，将图片进行保存

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)


try:
    # pylint: disable=g-import-not-at-top
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
