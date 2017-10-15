# breast-cancer-wisconsin 乳腺癌
import datetime
import pandas as pd
import numpy as np
# 分割 训练集、测试集
from sklearn.model_selection import train_test_split
# 标准化
from sklearn.preprocessing import StandardScaler
# 逻辑斯蒂回归、随机梯度…
from sklearn.linear_model import SGDClassifier
# 分类报告 classification_report
from sklearn.metrics import classification_report

colume_names = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion'
    ,'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']

df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
                   ,names=colume_names)

# 数据预处理，清除含空数据
data = pd.DataFrame(df)
data = data.replace(to_replace='?',value=np.nan)
data = data.dropna(how='any')

# 随机采样 25% 的数据用于测试，剩下 75% 用于构建训练集合
X_train,X_test,y_train,y_test = train_test_split(data[colume_names[1:10]],data[colume_names[10]],test_size=0.25,random_state=33)

# 查验训练样本的数量和类别分布
print(pd.value_counts(y_train))
# 查验测试样本的数量和类别分布
print(pd.value_counts(y_test))

# 标准化数据，均值0，方差1
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 计时开始
starttime = datetime.datetime.now()

# 初始化 SGDClassifier
sgdc = SGDClassifier()

# 调用 SGDClassifier 中 fit 函数来训练模型参数，并用训练出的 sgdc 模型预测，存在sgdc_y_predict
sgdc.fit(X_train,y_train)
sgdc_y_predict = sgdc.predict(X_test)

# 计时结束
endtime = datetime.datetime.now()

# 模型训练用时
print((endtime - starttime).microseconds)

# 使用 SGDClassifier 模型自带的评分函数 score 获得模型在测试集上的准确性结果
print('Accuracy of SGD Classifier:',sgdc.score(X_test,y_test))
# 使用 classification_report 模块获得 SGDClassifier 其他三个指标的结果
print(classification_report(y_test,sgdc_y_predict,target_names=['Benign(阴性2)','Malignant(阳性4)']))