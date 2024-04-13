import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, roc_curve

# print("lightgbm version:", lgb.__version__)


def calculate_ks(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
    ks = np.max(tpr - fpr)
    return ks


def evaluate_model(classifier, X, y):
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # 初始化存储分数的列表
    metrics_summary = {
        'val_auc': [], 'val_accuracy': [],
        'val_recall': [], 'val_precision': [],
        'val_f1': [], 'val_ks': []
    }

    for train_index, val_index in kf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        classifier.fit(X_train, y_train)

        # 对训练集和验证集的预测标签
        y_val_pred = classifier.predict(X_val)
        y_val_pred_proba = classifier.predict_proba(X_val)

        # 计算各项指标
        metrics_summary['val_auc'].append(roc_auc_score(y_val, y_val_pred_proba[:, 1]))
        metrics_summary['val_accuracy'].append(accuracy_score(y_val, y_val_pred))
        metrics_summary['val_recall'].append(recall_score(y_val, y_val_pred))
        metrics_summary['val_precision'].append(precision_score(y_val, y_val_pred))
        metrics_summary['val_f1'].append(f1_score(y_val, y_val_pred))
        metrics_summary['val_ks'].append(calculate_ks(y_val, y_val_pred_proba))

    # 打印平均指标
    print('#' * 20)
    for metric in metrics_summary:
        mean_value = np.mean(metrics_summary[metric])
        std_value = np.std(metrics_summary[metric], ddof=1)  # 用于样本标准差的计算
        print(f'Mean {metric}: {mean_value:.4f}, Std {metric}: {std_value:.4f}')


# 加载处理后的特征数据和标签
data = pd.read_csv("data.csv")
y = data['is_default']
X = data.drop(['is_default','loan_id','user_id'], axis=1)

# 实例化lightgbm分类器
lgb_classifier = lgb.LGBMClassifier(
    n_estimators=50,
    max_depth=12,
    random_state=42
)


# 实例化随机森林分类器
rf_classifier = RandomForestClassifier(
    n_estimators=50,
    max_depth=12,
    random_state=42
)

# 实例化决策树分类器
dt_classifier = DecisionTreeClassifier(
    max_depth=12,
    random_state=42
)

# 实例化逻辑回归分类器
logistic_regression = LogisticRegression()

# 使用函数评估模型
evaluate_model(dt_classifier, X, y)
