import pandas as pd
from sklearn.preprocessing import LabelEncoder

path_train = "../../data/Taiwan credit dataset/default of credit card clients.xls"

# 读取数据
data = pd.read_excel(path_train, header=1)

# 获取数据集的形状
shape = data.shape
num_samples = shape[0]
num_features = shape[1]
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")

# 删除缺失值超过50%的列
threshold = 0.5 * len(data)
data = data.loc[:, data.isnull().mean() < threshold]

# 使用剩余特征中的众数填充缺失值
for col in data.columns:
    if data[col].isnull().any():
        mode_value = data[col].mode()[0]
        data[col].fillna(mode_value, inplace=True)

# 获取数据集的形状
shape = data.shape
num_samples = shape[0]
num_features = shape[1]
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")


# 导出处理后的特征和标签
data.to_csv("data.csv", index=False)

print("Data processing successful!")
