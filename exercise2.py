import pandas as pd
df = pd.read_csv("IRIS.csv")
setosa = df[df['species'] == 'setosa']
versicolor = df[df['species'] == 'versicolor']
virginica = df[df['species'] == 'virginica']

print("Setosa Species:")
print(setosa)
print("\nVersicolor Species:")
print(versicolor)
print("\nVirginica Species:")
print(virginica)

covariance = df.groupby("species")[["sepal_length","sepal_width","petal_length","petal_width"]].cov()
print("Covariance Matrix:")
print(covariance)

correlation = df.groupby("species")[["sepal_length","sepal_width","petal_length","petal_width"]].corr()
print("Correlation Matrix:")
print(correlation)