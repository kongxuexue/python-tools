'''
* 参考文章：[用scikit-learn进行LDA降维](http://www.toutiao.com/i6371673477581111809/)
* 使用python3.5、sklearn
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # %matplotlib inline
from sklearn.datasets.samples_generator import make_classification
from sklearn.decomposition import PCA
print(__doc__)

'''生成三类三维的数据'''
X, y = make_classification(n_samples=1000, n_features=3, n_redundant=0, n_classes=3,
                           n_informative=2, n_clusters_per_class=1, class_sep=0.5, random_state=10)
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
plt.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o', c=y)
print("生成三类数据成功")
plt.show()

'''
PCA降维
    PCA没有利用类别信息，我们可以看到降维后，样本特征和类别的信息关联几乎完全丢失
'''
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
X_new = pca.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
print("PCA降维成功")
plt.show()

'''
LDA降维
    有类别标签的数据，优先选择LDA去尝试降维；当然也可以使用PCA做很小幅度的降维去消去噪声，然后再使用LDA降维
'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X, y)
X_new = lda.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
print("LDA降维成功")
plt.show()
