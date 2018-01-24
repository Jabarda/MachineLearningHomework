import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import csv

news_train = list(csv.reader(open('news_train.txt', 'rt', encoding="utf8"), delimiter='\t'))
#читаем данные
news_train_data = []
for x in news_train:
    news_train_data.append(x[2])


news_train_data_target = []
for x in news_train:
    news_train_data_target.append(x[0])

#создаем словарь характерных признаков и переводим документ в векторы признаков
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(news_train_data)
#считаем частоту и обратную частоту терминов
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)#прогоняем алгоритм оценки на данных
X_train_tf = tf_transformer.transform(X_train_counts)#преобразуем матрицу к виду tf-idf
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)#а можно сделать сразу и быстрее
#обучаем классификатор
clf = MultinomialNB().fit(X_train_tfidf, news_train_data_target)#полиномиальный Наивный Байесовский классификатор
docs_new = ['В Ираке новые танки']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
#######################
#Чтобы с цепочкой vectorizer => transformer => classifier, используем Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])
#обучаем модель
text_clf = text_clf.fit(news_train_data, news_train_data_target)

print("Fetching test data...")
news_test = list(csv.reader(open('news_train.txt', 'rt', encoding="utf8"), delimiter='\t'))

print("Fetching test dataset...")
news_test_data = []

for x in news_test:
    news_test_data.append(x[2])
print("Fetching test dataset target...")

news_test_data_target = []
for x in news_train:
    news_test_data_target.append(x[0])

print("Starting testing...")

docs_test = news_test_data
predicted = text_clf.predict(docs_test)



print (np.mean(predicted == news_test_data_target))


print("Fetching FINAL dataset...")
news_test_final = list(csv.reader(open('news_test.txt', 'rt', encoding="utf8"), delimiter='\t'))

news_test_data_final = []
for x in news_test_final:
    news_test_data_final.append(x[1])

docs_test = news_test_data_final

predicted = text_clf.predict(docs_test)
fh = open("final_output_Bayes.txt", 'w')
for item in predicted:
  fh.write("%s\n" % item)

#попробуем улучшить с помощью svm
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, max_iter=5, random_state=42)),
])
_ = text_clf.fit(news_train_data, news_train_data_target)
predicted = text_clf.predict(docs_test)
print("Writing to file...")
fh = open("final_output.txt", 'w')
for item in predicted:
  fh.write("%s\n" % item)
print (np.mean(predicted == news_test_data_target))
