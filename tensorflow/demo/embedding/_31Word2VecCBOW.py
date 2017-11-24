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
    用多个输入预测目标词。
    相比skip-gram模型，基本类似，需要改动的地方为：
    1. cbow使用的是（多个源词预测目标词），所以生成训练batch不同
    2. 由于是多个源词预测目标词，所以对源词的词向量计算均值，使用均值和目标词计算loss
    3. loss函数和skip-gram一样都可以使用nce也可以使用sampled_softmax_loss等函数
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
print('Data size', len(vocabulary))  # 词汇一共17005207个

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
    count.extend(collections.Counter(words).most_common(n_words - 1))  # count存放最常见的n个词和词频
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
'''
CBOW 修改生成的batch数据，应该是（多个源词对应一个目标词）
'''


def generate_batch(batch_size, skip_window):
    # skip window is the amount of words we're looking at from each side of a given word
    # creates a single batch
    global data_index
    assert skip_window % 2 == 1

    span = 2 * skip_window + 1  # [ skip_window target skip_window ]

    batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # e.g if skip_window = 2 then span = 5
    # span is the length of the whole frame we are considering for a single word (left + word + right)
    # skip_window is the length of one side

    # queue which add and pop at the end
    buffer = collections.deque(maxlen=span)

    # get words starting from index 0 to span
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # num_skips => # of times we select a random word within the span?
    # batch_size (8) and num_skips (2) (4 times)
    # batch_size (8) and num_skips (1) (8 times)
    for i in range(batch_size):
        target = skip_window  # target label at the center of the buffer
        target_to_avoid = [skip_window]  # we only need to know the words around a given word, not the word itself

        # do this num_skips (2 times)
        # do this (1 time)

        # add selected target to avoid_list for next time
        col_idx = 0
        for j in range(span):
            if j == span // 2:
                continue
            # e.g. i=0, j=0 => 0; i=0,j=1 => 1; i=1,j=0 => 2
            batch[i, col_idx] = buffer[j]  # [skip_window] => middle element
            col_idx += 1
        labels[i, 0] = buffer[target]

        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    assert batch.shape[0] == batch_size and batch.shape[1] == span - 1
    return batch, labels


# 构造数据，CBOW使用span中所有的词语
batch, labels = generate_batch(batch_size=8, skip_window=1)
print(batch.shape, batch[0][0])
for i in range(8):
    print(batch[i][0], reverse_dictionary[batch[i][0]], batch[i][1], reverse_dictionary[batch[i][1]],
          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

embedding_size = 128  # Dimension of the embedding vector. #设置embedding矩阵的大小
batch_size = 128  # 每次生成多少组（目标词与源词）
skip_window = 1  # How many words to consider left and right. # 设置左右的源词窗口大小（即限定源词范围）
num_skips = 2  # How many times to reuse an input to generate a label. # 设置每个目标词的源词个数（从源词窗口中选定）

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
# 随机选取一些小id的词语验证knn词语相似性（字典中id越小越高频）
valid_size = 16  # Random set of words to evaluate similarity on. # 验证valid_size个词语的相似结果
valid_window = 100  # Only pick dev samples in the head of the distribution. # 设置验证的词语的范围
# pick 16 samples from 100
valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
valid_examples = np.append(valid_examples, random.sample(range(1000, 1000 + valid_window), valid_size // 2))
num_sampled = 64  # Number of negative examples to sample.

# Input data.
'''
    CBOW train_dataset 的 placeholder 改变为 （b x 2*skip_window）（记住，span-1 = 2*skip_window）。其它保持不变。
'''
train_dataset = tf.placeholder(tf.int32, shape=[batch_size, 2 * skip_window])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Ops and variables pinned to the CPU because of missing GPU implementation # embedding_lookup缺乏GPU实现？
# with tf.Graph().as_default(), tf.device('/cpu:0'):
# Look up embeddings for inputs.
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))  # embedding矩阵变量
# embed = tf.nn.embedding_lookup(embeddings, train_inputs)
embeds = None
'''
# CBOW的 embedding_lookup需要改写
查找 train_dataset (大小为 b x 2*skip_window) 的每一行，查找行中词ID对应的向量。
然后将这些向量保存在临时变量（ embedding_i ）中，在把这些向量连接起来称为复合向量（embeds）(大小为 2*skip_window x b x D)，
进而在 axis 0 上求得 reduce mean 。最终我们可以对 data 的每个 batch 生成 train_labels 中词相应上下文的平均向量。
'''
for i in range(2 * skip_window):
    embedding_i = tf.nn.embedding_lookup(embeddings, train_dataset[:, i])
    print('embedding %d shape: %s' % (i, embedding_i.get_shape().as_list()))
    emb_x, emb_y = embedding_i.get_shape().as_list()
    if embeds is None:
        embeds = tf.reshape(embedding_i, [emb_x, emb_y, 1])
    else:
        embeds = tf.concat([embeds, tf.reshape(embedding_i, [emb_x, emb_y, 1])], 2)  # 在第三维度上进行concat

assert embeds.get_shape().as_list()[2] == 2 * skip_window
print("Concat embedding size: %s" % embeds.get_shape().as_list())
avg_embed = tf.reduce_mean(embeds, 2, keep_dims=False)
print("Avg embedding size: %s" % avg_embed.get_shape().as_list())

# 替换NCE loss，使用sampled_softmax_loss
softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                  stddev=1.0 / math.sqrt(embedding_size)))
softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

'''
使用sampled_softmax_loss作为loss函数
'''
loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(weights=softmax_weights,
                               biases=softmax_biases,
                               labels=train_labels,
                               inputs=avg_embed,
                               num_sampled=num_sampled,
                               num_classes=vocabulary_size))

# Construct the SGD optimizer using a learning rate of 1.0.
optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

# Compute the cosine similarity between minibatch examples and all embeddings.
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# Add variable initializer.
init = tf.global_variables_initializer()

# Step 5: Begin training.
num_steps = 100001

with tf.Session() as session:
    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    for step in range(num_steps):
        # 不断生成目标词和源词数据进行训练，使用global data_index向后生成训练集
        batch_inputs, batch_labels = generate_batch(batch_size, skip_window)
        feed_dict = {train_dataset: batch_inputs, train_labels: batch_labels}

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
                valid_word = reverse_dictionary[valid_examples[i]]  # 得到选取的要验证的词语
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
