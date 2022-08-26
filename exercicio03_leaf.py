import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time
import pandas as p

warnings.filterwarnings('ignore')

#LEAF

df = p.read_csv(u'./leaf/leaf.csv')
df_train, df_test, l_train, l_test = train_test_split(df.iloc[:, 1:16].values, df.iloc[:,0].values, test_size=0.2, random_state=int(time.time()))

print("#################### DECISION TREE #########################")
dt = DecisionTreeClassifier(class_weight='balanced', random_state=int(time.time()))
dt.fit(df_train, l_train)
dt_pred = dt.predict(df_test)
accuracy = dt.score(df_test, l_test)
print(f'Acurácia: {accuracy}')
print(classification_report(l_test, dt_pred))

#NAIVE BAYES LEAF
print("#################### MULTINOMIAL NAIVE BAYES #########################")
mnb = MultinomialNB()
mnb.fit(df_train, l_train)
mnb_pred = mnb.predict(df_test)
accuracy = mnb.score(df_test, l_test)
print(f'Acurácia: {accuracy}')
print(classification_report(l_test, mnb_pred))

print("#################### GAUSSIAN NAIVE BAYES #########################")
gnb = GaussianNB()
gnb.fit(df_train, l_train)
gnb_pred = gnb.predict(df_test)
accuracy = gnb.score(df_test, l_test)
print(f'Acurácia: {accuracy}')
print(classification_report(l_test, gnb_pred))