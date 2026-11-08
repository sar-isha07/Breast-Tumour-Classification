from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

print(df.head())
print("Shape:", df.shape)
print(df.isnull().sum().sum())
print(df["target"].value_counts())