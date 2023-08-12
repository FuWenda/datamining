import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 读取.data文件
data_file_path = 'data/arrhythmia.data'
data_df = pd.read_csv(data_file_path, header=None)

# 读取.names文件
names_file_path = 'data/arrhythmia.names'
with open(names_file_path, 'r') as names_file:
    names = names_file.read()


# 为数据框添加列名
column_names = [f'Col{i}' for i in range(1, 281)]
data_df.columns = column_names
data_df.rename(columns={data_df.columns[-1]: 'Class'}, inplace=True)
# data_df.to_csv('data.csv', index=False)  # 如果您不希望保存索引列，请将index参数设置为False
#数据清洗工作
#缺失值查询
data_df.replace('?', np.NaN, inplace=True)
missing_values = data_df.isnull().sum()# 利用isnull().sum()方法计算每一列的缺失值数量
columns_with_missing_values = missing_values[missing_values > 0]
# print(columns_with_missing_values)

#计算每一个类别的mean
list=['Col11', 'Col12', 'Col13', 'Col15', 'Class']
subset = data_df.loc[:, list]#用来计算不同类的mean
subset1= data_df.loc[:, list]#用来替换
subset.replace(np.NaN,0, inplace=True)
subset=subset.astype('int64')
mean_class=subset.groupby('Class')[list[:-1]].mean()
mean_class=mean_class.values.astype(int)#转化为np,取int32
new_rows = np.zeros((3, mean_class.shape[1]))#给11,12,13添加空值方便索引
mean_class= np.insert(mean_class, 10, new_rows, axis=0)
#用类别mean替换NaN
for row in range(452):#452
    for col in range(4):
        x=subset1.iloc[row,col]
        if(pd.isnull(x)):
            c = subset1.iloc[row, -1]
            subset1.iloc[row, col]=mean_class[c-1,col]

df_new = data_df.drop('Col14', axis=1)#删除
df_new.loc[:, list] = subset1
df_new=df_new.astype('float32')

print('数据清洗完毕')


# # 数据的可视化
# print("\nNames file contents:")
# print(names)
# def bar(data_df):
#
#     # 提取最后一个点的值
#     last_point = data_df.iloc[:, -1]
#     # 统计数据并绘制分布图
#     plt.hist(last_point, bins=20, edgecolor='black')
#     plt.xlabel("Value")
#     plt.ylabel("Frequency")
#     plt.title("Categorical bar chart of arrhythmias")
#     plt.show()
# bar(data_df)