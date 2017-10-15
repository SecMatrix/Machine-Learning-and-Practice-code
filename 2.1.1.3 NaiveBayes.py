# 导入新闻数据抓取器fetch_20newsgroup
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
# 文本特征向量转化模块
from sklearn.feature_extraction.text import CountVectorizer
# 导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 需要即时下载数据
news = fetch_20newsgroups(subset='all')
# 查验数据规模和细节
print(len(news.data))
# print(news.data[1])
X_train,X_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25,random_state=33)

# 文本特征向量抽取
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

# 默认配置 朴素贝叶斯
mnb = MultinomialNB()
mnb.fit(X_train,y_train)

y_predict = mnb.predict(X_test)
print("The Accuracy of Naive Bayes Classifier is:",mnb.score(X_test,y_test))
print(classification_report(y_test,y_predict,target_names=news.target_names))