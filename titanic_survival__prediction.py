import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("titanic02.csv")
missing_data = df.isnull().sum()
print(missing_data)


# Heatmap of Missing values

sns.heatmap(df.isnull(),cmap="viridis", yticklabels=False)
plt.title("Missing Data Heatmap")
plt.show()


# Countplot between passengers survived vs died

data01 = df[['Survived', 'Sex']]
# filling missing values of sex with mode
data01.fillna(data01['Sex'].mode()[0], inplace = True)
sns.countplot(x = 'Survived',hue = 'Sex', data = data01, palette='rainbow', edgecolor = 'black')
plt.title("Passangers Survived vs Died")
plt.show()


# Histrograph showing the age distribution

data02 = df['Age']
# filling the missing age values with the median age of the data
data02.fillna(data02.median(), inplace = True)
sns.histplot(data02, kde = True)
plt.title("Age distrubution graph")
plt.show()


# Fare distribution histograph

data03 = df['Fare']
# filling missing values with median
data03.fillna(data03.median(), inplace = True)
sns.histplot(data03, kde = True)
plt.title("Fare distribution graph")
plt.show()


# Count plot of Survival by Passanger Class

data04 = df[["Pclass", "Survived"]]
# filling missing values with mode
data04.fillna(data04['Pclass'].mode()[0], inplace=True)
sns.countplot(x = 'Survived', hue = "Pclass", data = data04, palette='plasma')
plt.title("Passangers survived by class")
plt.show()


# A bar plot of survival by ages

data05 = df[["Age", "Survived"]]
data05.fillna(data05['Age'].median(), inplace=True)
sns.barplot(x = 'Survived', data= data05, y = "Age", palette='viridis', edgecolor = 'black')
plt.title("Passangers survived by age")
plt.show()


# Precicting whether a passenger survived on the basis of other features

data06 = df.drop(columns= {"Name", "PassengerId", "SibSp", "Parch","Ticket", "Cabin"})
data06['Pclass'].fillna(data06['Pclass'].mode()[0], inplace =  True)
data06['Sex'].fillna(data06['Sex'].mode()[0], inplace = True)
data06['Age'].fillna(data06['Age'].median(), inplace = True)
data06['Fare'].fillna(data06['Fare'].mean(), inplace = True)
data06['Embarked'].fillna(data06['Embarked'].mode()[0], inplace = True)

le = LabelEncoder()
data06["Sex_Encoded"] = le.fit_transform(data06['Sex'])
data06['Embarked_Encoded'] = le.fit_transform(data06['Embarked'])

data06.drop(columns = {'Sex', "Embarked"}, inplace = True)

x = data06.drop(columns={"Survived"})
y = data06['Survived']
print(x, y)

scalar = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)

model = LogisticRegressionCV(max_iter=2000, class_weight="balanced")
model.fit(x_train, y_train)
model.predict(x_test)
print(model.score(x_test, y_test))

new_data = pd.DataFrame([{
    'Pclass': 2,
    'Age': 51,
    'Fare': 12,
    'Sex_Encoded': 1,
    'Embarked_Encoded': 0
}])

new_data = scalar.transform(new_data)
model.predict(new_data)
new_data = scalar.transform(new_data)

predicted_value = model.predict(new_data)
print(predicted_value)