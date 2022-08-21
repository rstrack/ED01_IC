from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import time
import pandas as p

df = p.read_csv(u'./car/car.data', header=None)

le = LabelEncoder()

for i in df.columns:
    if i !=6:
        df[i] = le.fit_transform(df[i])

df_train, df_test, l_train, l_test = train_test_split(df.iloc[:, 0:6].values, df.iloc[:,6].values, test_size=0.2, random_state=int(time.time()))

print("#################### DECISION TREE #########################")

dt = DecisionTreeClassifier(class_weight='balanced', random_state=int(time.time()))

dt.fit(df_train, l_train)

dt_pred = dt.predict(df_test)

accuracy = dt.score(df_test, l_test)
print(f'Score: {accuracy}')

print(classification_report(l_test, dt_pred))

print("#################### NAIVE BAYES #########################")

#NAIVE BAYES

gnb = GaussianNB()
gnb.fit(df_train, l_train)
gnb_pred = gnb.predict(df_test)
accuracy = gnb.score(df_test, l_test)
print(f'Score: {accuracy}')
print(classification_report(l_test, gnb_pred))