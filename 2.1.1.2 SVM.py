from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# 导入数据
digits = load_digits()
# print(digits.data.shape)

X_train,X_test,y_train,y_test = train_test_split(digits.data, digits.target, test_size=0.25,random_state=33)

# 标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# 线性支持向量机分类器
lsvc = LinearSVC()
lsvc.fit(X_train,y_train)

# 预测
y_predict =lsvc.predict(X_test)

# 使用模型自带的评分函数 score 获得模型在测试集上的准确性结果
print('The Accuracy of Linear SVC is:',lsvc.score(X_test,y_test))
# 使用 classification_report 模块获得其他三个指标的结果
print(classification_report(y_test,y_predict,target_names=digits.target_names.astype(str)))