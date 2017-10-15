# 泰坦尼克号乘客数据
# 导入pandas用户数据分析
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
# 导入决策树分类器
from sklearn.tree import DecisionTreeClassifier

# 互联网获取泰坦尼克号数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
# print(titanic.head())
# titanic.info()
X = titanic[['pclass','age','sex']]
y = titanic['survived']
# 对当前选择的特征进行探查
# print(X.info())
#############################################################################################
# 借助上面的输出，我们设计如下几个数据处理的任务【数据预处理】
# 1. age这个列只有633个，需要补完
# 2. sex与pclass两个数据列的值都是类别型，需要转化为数值特征，用0/1代替
# 首先我们补充age里面的数据，使用平均数或者中位数都是对模型偏离造成最小影响的策略
X['age'].fillna(X['age'].mean(),inplace=True)
# print(X.info())
#############################################################################################
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=33)
# 特征转换器
vec = DictVectorizer(separator=False)
X_train = vec.fit_transform(X_train.to_dict(orient='record'))

# 使用默认配置初始化决策树分类器
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_predict = dtc.predict(X_test)

# 性能
from sklearn.metrics import classification_report
print(dtc.score(X_test,y_test))
print(classification_report(y_predict,y_test,target_names=['died','survived']))