import pandas as pd
from sklearn.preprocessing import LabelEncoder

path_train = "../../data/Give Me Some Credit/cs-training.csv"

data = pd.read_csv(path_train, index_col=0)

shape = data.shape
num_samples = shape[0]
num_features = shape[1]
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")

threshold = 0.5 * len(data)
data = data.loc[:, data.isnull().mean() < threshold]

for col in data.columns:
    if data[col].isnull().any():
        mode_value = data[col].mode()[0]
        data[col].fillna(mode_value, inplace=True)

le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])

shape = data.shape
num_samples = shape[0]
num_features = shape[1]
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")


data.to_csv("data.csv", index=False)

print("Data processing successful!")
