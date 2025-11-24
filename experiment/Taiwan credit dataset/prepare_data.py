import pandas as pd
from sklearn.preprocessing import LabelEncoder

path_train = "../../data/Taiwan credit dataset/default of credit card clients.xls"

data = pd.read_excel(path_train, header=1)

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

shape = data.shape
num_samples = shape[0]
num_features = shape[1]
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")

data.to_csv("data.csv", index=False)

print("Data processing successful!")
