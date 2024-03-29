import copy
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, roc_curve
from ctgan import CTGAN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from sklearn.model_selection import KFold, train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin


def calculate_ks(y_true, y_prob):
    """
    计算并返回Kolmogorov-Smirnov统计量。

    参数:
    - y_true: array-like, 真实标签数组。
    - y_prob: array-like, 预测为正类的概率数组。

    返回:
    - ks: float, KS统计量的值。
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks = np.max(tpr - fpr)
    return ks


class TriEnhanceClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier: BaseEstimator,
                 discrete_columns=[],
                 pseudo_label=-1, n_splits=3,
                 random_state=42, target_metric='f1'):
        """
        初始化APLIDC分类器。

        参数:
        - base_classifier: BaseEstimator, 用作训练的基本分类器。
        - discrete_columns: list, 指定哪些列是离散的，用于CTGAN模型，默认为空列表。
        - pseudo_label: int, 用于主动学习中的伪标签值，默认为-1。
        - n_splits: int, 在主动学习中使用的K折交叉验证的折数，默认为5折。
        - random_state: int, 随机种子，用于确保可重复性，默认为42。
        - target_metric: str, 模型优化目标的性能指标（如'auc', 'accuracy', 'f1'等），默认为‘f1’。
        """
        self.base_classifier_template = base_classifier  # 保存原始分类器作为模板
        self.base_classifier = copy.deepcopy(base_classifier)  # 创建用于训练的分类器实例
        self.method = None  # 数据合成层参数
        self.discrete_columns = discrete_columns  # ctgan模型下需要指定离散变量列
        self.mode = None  # 主动学习模型选择
        self.pseudo_label = pseudo_label  # 主动学习中所用伪标签
        self.n_splits = n_splits  # K折伪标签折数
        self.random_state = random_state  # 随机种子
        self.target_metric = target_metric  # 希望提高的评估指标

    def fit(self, X_train, y_train, X_unlabeled):
        """
        训练APLIDC分类器。

        参数:
        - X_train: DataFrame, 训练数据的特征。
        - y_train: Series, 训练数据的标签。
        - X_unlabeled: DataFrame, 未标记数据的特征。

        返回:
        - self: 对象本身，允许链式调用。
        """
        # 记录原始数据的标签比例
        original_label_proportions = y_train.value_counts(normalize=True)
        print(original_label_proportions)

        # 使用50%的数据作为验证集
        X_train_copy, y_train_copy = X_train, y_train
        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train_copy, y_train_copy, test_size=0.5, random_state=self.random_state, stratify=y_train_copy)

        # 遍历不同的模式，评估性能，选择最佳数据合成层模式
        best_score = 0
        best_method = None # 记录最佳模式
        misclassified_samples_best = None  # 存储最佳模式下错误分类的样本
        misclassified_labels_best = None  # 存储错误分类样本的真实标签
        correctly_classified_samples_best = None  # 存储最佳模式下正确分类的样本
        correctly_classified_labels_best = None  # 存储正确分类样本的真实标签

        for method in ['regular', 'ctgan']:
            self.method = method
            X_train_split_balanced, y_train_split_balanced = self._balance_data(X_train_split, y_train_split)

            model = copy.deepcopy(self.base_classifier_template)
            model.fit(X_train_split_balanced, y_train_split_balanced)

            # 根据用户选择的指标计算得分
            predictions = model.predict(X_test_split)
            predictions_proba = model.predict_proba(X_test_split)[:, 1]
            score = self._calculate_metric(y_test_split, predictions, predictions_proba)

            if score > best_score:
                best_score = score
                best_method = method
                # 获取分类错误的样本索引
                misclassified_indices = np.where(predictions != y_test_split)[0]
                # 使用这些索引获取错误分类的样本及其真实标签
                misclassified_samples_best = X_test_split.iloc[misclassified_indices]
                misclassified_labels_best = y_test_split.iloc[misclassified_indices]
                # 获取分类正确的样本索引
                correctly_classified_indices = np.where(predictions == y_test_split)[0]
                # 使用这些索引获取正确分类的样本及其真实标签
                correctly_classified_samples_best = X_test_split.iloc[correctly_classified_indices]
                correctly_classified_labels_best = y_test_split.iloc[correctly_classified_indices]

        print(f"The best method is {best_method}")
        self.method = best_method
        X_train_split_balanced, y_train_split_balanced = self._balance_data(X_train_split, y_train_split)

        # 打印或返回错误分类的样本及其真实标签
        print("错误分类的样本数量：", len(misclassified_samples_best))
        print("错误分类的样本占验证集的比例：", len(misclassified_samples_best)/len(X_test_split))
        print("错误分类的样本的标签统计：", misclassified_labels_best.value_counts())

        # 合并正确分类的样本与最佳模式下平衡后的数据与错误分类的样本
        X_combined = pd.concat([X_train_split_balanced, correctly_classified_samples_best])
        y_combined = pd.concat([y_train_split_balanced, correctly_classified_labels_best])
        X_combined = pd.concat([X_combined, misclassified_samples_best])
        y_combined = pd.concat([y_combined, misclassified_labels_best])

        X_train_filtered, y_train_filtered = X_combined, y_combined

        # 使用过滤后的数据中30%的数据作为验证集
        X_train_filtered_split, X_test_filtered_split, y_train_filtered_split, y_test_filtered_split = train_test_split(X_train_filtered, y_train_filtered, test_size=0.3, random_state=self.random_state, stratify=y_train_filtered)

        # 使用数据增强前划分的训练集训练模型，计算得分作为baseline
        self.base_classifier = copy.deepcopy(self.base_classifier_template)
        self.base_classifier.fit(X_train_filtered_split, y_train_filtered_split)
        predictions = self.base_classifier.predict(X_test_filtered_split)
        predictions_proba = self.base_classifier.predict_proba(X_test_filtered_split)[:, 1]
        score = self._calculate_metric(y_test_filtered_split, predictions, predictions_proba)
        print(f"baseline score:{score}")

        # 遍历不同的模式，评估性能，选择最佳主动学习层模式
        best_score = score
        best_mode = None
        modes = ['generate_pseudo_labels', 'enhance_training_set']
        for mode in modes:
            self.mode = mode
            X_train_split_updated, y_train_split_updated, _, _ = self._enhance_training_set(X_train_filtered_split, y_train_filtered_split, X_unlabeled, X_test_filtered_split, y_test_filtered_split)
            print(f"{self.mode}:学习完成！")

            self.base_classifier = copy.deepcopy(self.base_classifier_template)
            self.base_classifier.fit(X_train_split_updated, y_train_split_updated)
            predictions = self.base_classifier.predict(X_test_filtered_split)
            predictions_proba = self.base_classifier.predict_proba(X_test_filtered_split)[:, 1]
            score = self._calculate_metric(y_test_filtered_split, predictions, predictions_proba)
            print(f"{self.mode} score:{score}")

            if score > best_score:
                best_score = score
                best_mode = mode

        # 使用最佳模式增强数据
        print(f"The best mode is {best_mode}")
        self.mode = best_mode
        if best_mode != None:
            X_train_updated, y_train_updated, _, _ = self._enhance_training_set(X_train_filtered, y_train_filtered, X_unlabeled)
            self.base_classifier = copy.deepcopy(self.base_classifier_template)
            self.base_classifier.fit(X_train_updated, y_train_updated)
        else:
            self.base_classifier = copy.deepcopy(self.base_classifier_template)
            self.base_classifier.fit(X_train_filtered, y_train_filtered)

        return self

    def _calculate_metric(self, y_true, y_pred, y_pred_proba):
        """
        根据指定的性能指标计算并返回模型性能评估结果。

        参数:
        - y_true: array-like, 真实标签数组。
        - y_pred: array-like, 模型预测的标签数组。
        - y_pred_proba: array-like, 预测为正类的概率数组。

        返回:
        - score: float, 指定性能指标的计算结果。
        """
        if self.target_metric == 'auc':
            return roc_auc_score(y_true, y_pred_proba)
        elif self.target_metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif self.target_metric == 'recall':
            return recall_score(y_true, y_pred)
        elif self.target_metric == 'precision':
            return precision_score(y_true, y_pred)
        elif self.target_metric == 'f1':
            return f1_score(y_true, y_pred)
        elif self.target_metric == 'ks':
            return calculate_ks(y_true, y_pred_proba)
        else:
            raise ValueError("Unsupported metric.")

    def predict(self, X):
        """
        使用训练好的模型进行预测。

        参数:
        - X: DataFrame, 待预测数据的特征。

        返回:
        - predictions: array, 模型的预测标签数组。
        """
        return self.base_classifier.predict(X)

    def predict_proba(self, X):
        """
        预测给定数据为正类的概率。

        参数:
        - X: DataFrame, 待预测数据的特征。

        返回:
        - probabilities: array, 预测为正类的概率数组。
        """
        return self.base_classifier.predict_proba(X)

    def _balance_data(self, X, y):
        """
        使用指定的方法平衡数据。

        参数:
        - X: DataFrame, 特征数据。
        - y: Series, 标签数据。

        返回:
        - X_resampled: DataFrame, 平衡后的特征数据。
        - y_resampled: Series, 平衡后的标签数据。
        """
        # 初始化记录合成样本的标签
        synthetic_labels = []

        if self.method in ['regular', 'borderline', 'svm', 'adasyn']:
            smote_methods = {
                'regular': SMOTE,
                'borderline': BorderlineSMOTE,
                'svm': SVMSMOTE,
                'adasyn': ADASYN
            }
            smote = smote_methods[self.method](random_state=self.random_state)
            X_resampled, y_resampled = smote.fit_resample(X, y)
        elif self.method == 'ctgan':
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)

            class_counts = y.value_counts()
            max_class_count = class_counts.max()

            X_resampled, y_resampled = X, y

            for minority_class in class_counts.index:
                num_minority = class_counts[minority_class]
                if num_minority < max_class_count:
                    num_samples_to_generate = max_class_count - num_minority

                    X_minority = X[y == minority_class]

                    ctgan = CTGAN(epochs=500, verbose=False)
                    ctgan.fit(X_minority, discrete_columns=self.discrete_columns)

                    synthetic_data = ctgan.sample(num_samples_to_generate)
                    X_resampled = pd.concat([X_resampled, synthetic_data], ignore_index=True)
                    synthetic_labels.extend([minority_class] * num_samples_to_generate)

            y_resampled = pd.concat([y_resampled, pd.Series(synthetic_labels)], ignore_index=True)
        else:
            raise ValueError("Unsupported method for data balancing.")

        return X_resampled, y_resampled

    def _filter_difficult_samples(self, X_train, y_train, original_label_proportions, difficulty_threshold=0.1):
        """
        过滤难以分类的样本。

        参数:
        - X_train: DataFrame, 训练数据的特征。
        - y_train: Series, 训练数据的标签。
        - original_label_proportions: dict, 原始标签比例。
        - difficulty_threshold: float, 难度阈值，用于过滤样本。

        返回:
        - X_train_filtered: DataFrame, 过滤后的训练数据特征。
        - y_train_filtered: Series, 过滤后的训练数据标签。
        """
        self.base_classifier = copy.deepcopy(self.base_classifier_template)
        self.base_classifier.fit(X_train, y_train)
        y_pred_proba = self.base_classifier.predict_proba(X_train)

        # 复制预测结果用于计算第二大概率
        y_pred_copy = y_pred_proba.copy()
        np.put_along_axis(y_pred_copy, np.argmax(y_pred_proba, axis=1)[:, None], 0, axis=1)

        # 计算每个样本最大概率与第二大概率的差异
        diff = np.max(y_pred_proba, axis=1) - np.max(y_pred_copy, axis=1)

        # 根据全局难度阈值标识需要过滤的样本
        filter_mask = diff < difficulty_threshold

        # 初始化保留样本掩码
        retain_samples_mask = np.zeros(len(y_train), dtype=bool)

        # 根据original_label_proportions直接确定各类别需要保留的样本数量
        for label, proportion in original_label_proportions.items():
            label_mask = (y_train == label) & filter_mask
            n_retain = int(np.sum(label_mask) * proportion)
            retain_indices = np.random.choice(np.where(label_mask)[0], n_retain, replace=False)
            retain_samples_mask[retain_indices] = True

        # 最终保留的样本为未标记过滤或被保留的样本
        final_retain_mask = ~filter_mask | retain_samples_mask
        X_train_filtered = X_train[final_retain_mask]
        y_train_filtered = y_train[final_retain_mask]

        return X_train_filtered, y_train_filtered

    def find_optimal_threshold_and_refilter(self, X_combined, y_combined, misclassified_samples_best, misclassified_labels_best, original_label_proportions):
        """
        寻找最优的难度阈值并重新过滤。

        参数:
        - X_train: DataFrame, 训练数据的特征。
        - y_train: Series, 训练数据的标签。
        - original_label_proportions: dict, 原始标签比例。

        返回:
        - X_train_filtered: DataFrame, 使用最优难度阈值过滤后的训练数据特征。
        - y_train_filtered: Series, 使用最优难度阈值过滤后的训练数据标签。
        """

        optimal_threshold = None
        best_performance = -np.inf
        for threshold in np.arange(0, 0.2, 0.04):
            # 使用当前阈值过滤样本
            X_filtered, y_filtered = self._filter_difficult_samples(X_combined, y_combined, original_label_proportions, difficulty_threshold=threshold)
            # 在过滤后的数据上训练模型
            self.base_classifier = copy.deepcopy(self.base_classifier_template)
            self.base_classifier.fit(X_filtered, y_filtered)
            # 在验证集上评估模型性能
            predictions = self.base_classifier.predict(misclassified_samples_best)
            predictions_proba = self.base_classifier.predict_proba(misclassified_samples_best)[:, 1]
            performance = self._calculate_metric(misclassified_labels_best, predictions, predictions_proba)

            if performance > best_performance:
                best_performance = performance
                optimal_threshold = threshold

        print(f"Optimal difficulty threshold found: {optimal_threshold} with performance: {best_performance}")

        # 使用找到的最优阈值重新过滤原始训练集
        X_combined_filtered, y_combined_filtered = self._filter_difficult_samples(X_combined, y_combined, original_label_proportions, difficulty_threshold=optimal_threshold)

        # 将过滤后的数据与监督数据混合
        X_filtered = pd.concat([X_combined_filtered, misclassified_samples_best])
        y_filtered = pd.concat([y_combined_filtered, misclassified_labels_best])

        return X_filtered, y_filtered

    def _enhance_training_set(self, X_train, y_train, X_unlabeled, X_test=None, y_test=None, target_sample_percent=0.3):
        """
        通过未标记数据增强训练集。

        参数:
        - X_train: DataFrame, 训练数据的特征。
        - y_train: Series, 训练数据的标签。
        - X_test: DataFrame, 验证数据的特征。
        - y_test: Series, 验证数据的标签。
        - X_unlabeled: DataFrame, 未标记数据的特征。
        - target_sample_percent: float, 选取未标记数据的百分比用于增强。

        返回:
        - X_train_updated: DataFrame, 更新后的训练数据特征。
        - y_train_updated: Series, 更新后的训练数据标签。
        - X_pseudo_labeled: DataFrame, 伪标签数据的特征（如果适用）。
        - X_pseudo_unlabeled: DataFrame, 仍然未标记的数据特征（如果适用）。
        """
        # 确保必要的参数已提供
        if self.mode is None:
            raise ValueError("Please Enter the 'mode' parameter.")

        if X_train is None or y_train is None or X_unlabeled is None:
            raise ValueError("Missing required parameters for generating pseudo labels.")

        if self.mode == 'generate_pseudo_labels':
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            pseudo_labeled_data = pd.DataFrame()

            for train_index, test_index in kf.split(X_unlabeled):
                X_train_fold, X_test_fold = X_unlabeled.iloc[train_index], X_unlabeled.iloc[test_index]
                if X_test is not None:
                    X_combined = pd.concat([X_train, X_test])
                    y_combined = pd.concat([y_train, y_test])
                    X_combined = pd.concat([X_combined, X_train_fold])
                    y_combined = pd.concat([y_combined, pd.Series([self.pseudo_label] * len(X_train_fold))])
                else:
                    X_combined = pd.concat([X_train, X_train_fold])
                    y_combined = pd.concat([y_train, pd.Series([self.pseudo_label] * len(X_train_fold))])
                self.base_classifier = copy.deepcopy(self.base_classifier_template)
                self.base_classifier.fit(X_combined, y_combined)
                pseudo_labels = self.base_classifier.predict(X_test_fold)

                temp_df = X_test_fold.copy()
                temp_df['pseudo_label'] = pseudo_labels
                pseudo_labeled_data = pd.concat([pseudo_labeled_data, temp_df])

            pseudo_labeled_data.reset_index(drop=True, inplace=True)

            X_pseudo_labeled = pseudo_labeled_data[pseudo_labeled_data['pseudo_label'] != self.pseudo_label]
            X_pseudo_unlabeled = pseudo_labeled_data[pseudo_labeled_data['pseudo_label'] == self.pseudo_label].drop(
                'pseudo_label', axis=1)

            X_train_updated = pd.concat([X_train, X_pseudo_labeled.drop('pseudo_label', axis=1)])
            y_train_updated = pd.concat([y_train, X_pseudo_labeled['pseudo_label']])

            return X_train_updated, y_train_updated, X_pseudo_labeled, X_pseudo_unlabeled

        elif self.mode == 'enhance_training_set':
            if X_test is not None:
                X_train_part, X_test_part, y_train_part, y_test_part = X_train, X_test, y_train, y_test
            else:
                # 拆分数据为训练集和验证集
                X_train_part, X_test_part, y_train_part, y_test_part = train_test_split(X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train)

            # 初始训练基分类器
            self.base_classifier = copy.deepcopy(self.base_classifier_template)
            self.base_classifier.fit(X_train_part, y_train_part)
            baseline_metric = self._calculate_metric(y_test_part, self.base_classifier.predict(X_test_part), self.base_classifier.predict_proba(X_test_part)[:, 1])

            improvement = True  # 用于追踪性能是否有提升
            # 从 X_train 和 y_train 中取第一条样本初始化 X_enhanced 和 y_enhanced
            X_enhanced = X_train.iloc[:1].copy()
            y_enhanced = y_train.iloc[:1].copy()

            while improvement and len(X_unlabeled) > 0:
                # 预测未标记数据的概率
                y_pred_proba = self.base_classifier.predict_proba(X_unlabeled)[:, 1]
                proba_indices = np.argsort(y_pred_proba)[::-1]  # 根据概率降序排序

                # 选取前30%的样本
                top_sample_indices = proba_indices[:int(len(proba_indices) * target_sample_percent)]
                X_selected = X_unlabeled.iloc[top_sample_indices]
                y_selected_pred = self.base_classifier.predict(X_selected)

                # 转换 X_selected 中的列数据类型以匹配 X_train_part
                for column in X_selected.columns:
                    if X_selected[column].dtype != X_train_part[column].dtype:
                        X_selected[column] = X_selected[column].astype(X_train_part[column].dtype)

                # 临时更新训练集并评估
                X_temp_combined = pd.concat([X_train_part, X_enhanced, X_selected]).reset_index(drop=True)
                y_temp_combined = pd.concat([y_train_part, y_enhanced, pd.Series(y_selected_pred, index=X_selected.index)]).reset_index(drop=True)

                self.base_classifier = copy.deepcopy(self.base_classifier_template)
                self.base_classifier.fit(X_temp_combined, y_temp_combined)

                # 根据用户选择的指标重新计算得分
                updated_metric = self._calculate_metric(y_test_part, self.base_classifier.predict(X_test_part), self.base_classifier.predict_proba(X_test_part)[:, 1])

                # 检查性能是否有改善
                if updated_metric > baseline_metric:
                    print("Performance improved with added samples.")
                    baseline_metric = updated_metric  # 更新基线性能指标
                    X_enhanced = pd.concat([X_enhanced, X_selected])  # 保存有效的样本
                    # 保存有效的增强样本标签
                    y_enhanced = pd.concat([y_enhanced, pd.Series(y_selected_pred, index=X_selected.index)])
                    X_unlabeled = X_unlabeled.drop(X_selected.index)  # 从未标记数据中移除选中的样本
                else:
                    print("No performance improvement with added samples.")
                    improvement = False  # 停止增强过程

            # 循环结束后，删除初始化时加入的样本
            if len(X_enhanced) > 1:  # 确保有添加的样本
                X_enhanced = X_enhanced.iloc[1:].reset_index(drop=True)
                y_enhanced = y_enhanced.iloc[1:].reset_index(drop=True)
                # 将验证有效的增强样本和标签合并到完整的训练集中
                X_train_final = pd.concat([X_train, X_enhanced]).reset_index(drop=True)
                y_train_final = pd.concat([y_train, y_enhanced]).reset_index(drop=True)
            else:
                X_train_final = X_train
                y_train_final = y_train

            return X_train_final, y_train_final, None, None

        else:
            raise ValueError("Invalid mode specified.")


