from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, cross_val_predict
import warnings

from DataCleaning import df_new


# 处理数据集
data=df_new.values
X=data[:,:-1]
y=data[:,-1]-1
features = np.array(X)
min_val = np.min(features)
shifted_features = features - min_val


# 创建分类器
classifier1 = DecisionTreeClassifier()#决策树
classifier2 = MultinomialNB()#朴素贝叶斯
classifier3 = SVC()#支持向量机

# 进行5折交叉验证
warnings.filterwarnings('ignore')#这里交叉验证时，样本数量太少会出现警告，忽略这个警告
y_pred1= cross_val_predict(classifier1, shifted_features, y, cv=5)
y_pred2= cross_val_predict(classifier2, shifted_features, y, cv=5)
y_pred3= cross_val_predict(classifier3, shifted_features, y, cv=5)


#打印
report1 = classification_report(y, y_pred1,zero_division=0)
report2 = classification_report(y, y_pred2,zero_division=0)
report3 = classification_report(y, y_pred3,zero_division=0)

print("决策树:")
print(report1)
print("")
print("朴素贝叶斯:")
print(report2)
print("")
print("支持向量机:")
print(report3)
print("")
