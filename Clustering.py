from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings

from DataCleaning import df_new

data=df_new.values
X=data[:,:-1]
y=data[:,-1]-1

#归一化
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# 主成分分析
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_norm)
X=X_pca
warnings.filterwarnings('ignore')#这里交叉验证时，样本数量太少会出现警告，忽略这个警告
# 聚类算法
#K-mean
kmeans = KMeans(n_clusters=3)
y_pred1 = kmeans.fit_predict(X)

#GMM高斯混合模型
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
y_pred2= gmm.predict(X)


# 计算轮廓系数
print("k-mean：")
silhouette_avg = silhouette_score(X, y_pred1)
print("轮廓系数为：", silhouette_avg)#[-1,1]之间，值越大越好
# 计算Davies-Bouldin指数
db_index = davies_bouldin_score(X, y_pred1)
print("Davies-Bouldin指数为：", db_index)#[0,正无穷]之间，值越小越好

print("高斯混合模型：")
print("k-mean")
silhouette_avg = silhouette_score(X, y_pred2)
print("轮廓系数为：", silhouette_avg)
# 计算Davies-Bouldin指数
db_index = davies_bouldin_score(X, y_pred2)
print("Davies-Bouldin指数为：", db_index)

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=y_pred1, cmap=plt.cm.Set1)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y_pred2, cmap=plt.cm.Set1)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('GMM Clustering')
plt.show()


