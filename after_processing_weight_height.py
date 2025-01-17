
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/drive/My Drive/weight-height.csv')
df

df.info()

df.shape

df.describe()

df.rename(columns = {
    "Height(Inches)":"Height",
    "Weight(Pounds)":"Weight"
}, inplace = True)

plt.figure(figsize = (7,6))
plt.hist(df["Height"], bins = 20)
plt.xlabel("Height")
plt.ylabel("Frequency")

plt.figure(figsize = (7,6))
plt.hist(df["Height"], bins = 20, rwidth = 0.8)
plt.xlabel("Height")
plt.ylabel("Frequency")

plt.scatter(x = df.Height, y = df.Weight)
plt.xlabel("Students Height")
plt.ylabel("Students Weight")
plt.show()

plt.figure(figsize = (7,6))
sns.scatterplot(x = df.Height, y = df.Weight)

plt.figure(figsize = (7,6))
plt.hist(df["Weight"], bins = 20, rwidth = 0.8)
plt.xlabel("Weight")
plt.ylabel("Frequency")

df.isnull().sum()

df2 = df.fillna(df.mean)

df2.isnull().sum()

df["Height"] = df["Height"] * 2.54
df["Weight"] = df["Weight"] / 2.205

df.head()

df = df.drop("Gender", axis = 1)

x = df.drop("Weight", axis = 1)
y = df.drop("Height", axis = 1)

x = x.fillna(x.mean())
y = y.fillna(y.mean())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)

print("shape_of_x_train", x_train.shape)
print("shape_of_y_train", y_train.shape)
print("shape_of_x_test", x_test.shape)
print("shape_of_y_test", y_test.shape)

df.info()

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(x_train, y_train)

x.isnull().sum()

m = lr.coef_
c = lr.intercept_

height = int(input("Enter the value of the height for the estimated weight: "))
y = m * height + c
y

plt.scatter(x_test,y_test)

plt.plot(x_train,y_train, color = "red")

import joblib

print(dir(joblib))

joblib.dump(lr,"student_weight_prediction.pkl")

chaukiet = joblib.load("student_weight_prediction.pkl")

height = int(input("Enter the value of height: "))
chaukiet.predict([[height]])[0][0]
