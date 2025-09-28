import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

df = pd.read_csv('Titanic-Dataset.csv')

x = df[['Age','Pclass','Fare','SibSp','Parch']]
# 量的データのみ
y = df['Survived']

model = DecisionTreeClassifier()
model.fit(x,y)
score = model.score(x,y)
print(score)
# スコアの表示

