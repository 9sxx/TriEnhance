import pandas as pd
import numpy as np
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, roc_curve
from TriEnhance import TriEnhanceClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def calculate_ks(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
    ks = np.max(tpr - fpr)
    return ks


def evaluate_model(classifier, X, y):
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    metrics_summary = {
        'val_auc': [], 'val_accuracy': [],
        'val_recall': [], 'val_precision': [],
        'val_f1': [], 'val_ks': []
    }

    for train_index, val_index in kf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        classifier.fit(X_train, y_train, X_val)

        y_val_pred = classifier.predict(X_val)
        y_val_pred_proba = classifier.predict_proba(X_val)

        metrics_summary['val_auc'].append(roc_auc_score(y_val, y_val_pred_proba[:, 1]))
        metrics_summary['val_accuracy'].append(accuracy_score(y_val, y_val_pred))
        metrics_summary['val_recall'].append(recall_score(y_val, y_val_pred))
        metrics_summary['val_precision'].append(precision_score(y_val, y_val_pred))
        metrics_summary['val_f1'].append(f1_score(y_val, y_val_pred))
        metrics_summary['val_ks'].append(calculate_ks(y_val, y_val_pred_proba))

    print('#' * 20)
    for metric in metrics_summary:
        mean_value = np.mean(metrics_summary[metric])
        std_value = np.std(metrics_summary[metric], ddof=1)
        print(f'Mean {metric}: {mean_value:.4f}, Std {metric}: {std_value:.4f}')


# 加载数据
data = pd.read_csv("data.csv")
y = data['Class']
X = data.drop('Class', axis=1)

# 指定离散变量
discrete_columns = []

# 实例化决策树分类器
dt_classifier = DecisionTreeClassifier(
    max_depth=12,
    random_state=42
)

# 实例化模型增强器
APLIDC_clf = TriEnhanceClassifier(base_classifier=dt_classifier, discrete_columns=discrete_columns)

# 评估增强后的模型
evaluate_model(APLIDC_clf, X, y)
