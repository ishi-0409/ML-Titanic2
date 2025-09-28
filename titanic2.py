import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('Titanic-Dataset.csv')

df_drop = df.dropna()
# 欠損値の削除

x = df_drop[['Age','Pclass','Fare','SibSp','Parch']]
y = df_drop['Survived']

survive_value = df['Survived'].value_counts()
# 生存者と死亡者の数

print(survive_value)

x_train,x_test,y_train,y_test = train_test_split(x,y)

model = DecisionTreeClassifier()
model.fit(x_train,y_train)
score = model.score(x_test,y_test)
print('正解率:',score)
