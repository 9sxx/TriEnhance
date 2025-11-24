import copy
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, roc_curve
from ctgan import CTGAN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from sklearn.model_selection import KFold, train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin


def calculate_ks(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks = np.max(tpr - fpr)
    return ks


class TriEnhanceClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_classifier: BaseEstimator,
                 discrete_columns=[],
                 pseudo_label=-1, n_splits=3,
                 random_state=42, target_metric='f1'):
        self.base_classifier_template = base_classifier
        self.base_classifier = copy.deepcopy(base_classifier)
        self.method = None
        self.discrete_columns = discrete_columns
        self.mode = None
        self.pseudo_label = pseudo_label
        self.n_splits = n_splits
        self.random_state = random_state
        self.target_metric = target_metric

    def fit(self, X_train, y_train, X_unlabeled):
        original_label_proportions = y_train.value_counts(normalize=True)
        print(original_label_proportions)

        X_train_copy, y_train_copy = X_train, y_train
        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train_copy, y_train_copy, test_size=0.5, random_state=self.random_state, stratify=y_train_copy)

        best_score = 0
        best_method = None
        misclassified_samples_best = None
        misclassified_labels_best = None
        correctly_classified_samples_best = None
        correctly_classified_labels_best = None

        for method in ['regular', 'ctgan']:
            self.method = method
            X_train_split_balanced, y_train_split_balanced = self._balance_data(X_train_split, y_train_split)

            model = copy.deepcopy(self.base_classifier_template)
            model.fit(X_train_split_balanced, y_train_split_balanced)

            predictions = model.predict(X_test_split)
            predictions_proba = model.predict_proba(X_test_split)[:, 1]
            score = self._calculate_metric(y_test_split, predictions, predictions_proba)

            if score > best_score:
                best_score = score
                best_method = method
                misclassified_indices = np.where(predictions != y_test_split)[0]
                misclassified_samples_best = X_test_split.iloc[misclassified_indices]
                misclassified_labels_best = y_test_split.iloc[misclassified_indices]
                correctly_classified_indices = np.where(predictions == y_test_split)[0]
                correctly_classified_samples_best = X_test_split.iloc[correctly_classified_indices]
                correctly_classified_labels_best = y_test_split.iloc[correctly_classified_indices]

        print(f"The best method is {best_method}")
        self.method = best_method
        X_train_split_balanced, y_train_split_balanced = self._balance_data(X_train_split, y_train_split)

        X_combined = pd.concat([X_train_split_balanced, correctly_classified_samples_best])
        y_combined = pd.concat([y_train_split_balanced, correctly_classified_labels_best])

        X_train_filtered, y_train_filtered = self.find_optimal_threshold_and_refilter(X_combined, y_combined, misclassified_samples_best, misclassified_labels_best, original_label_proportions)

        X_train_filtered_split, X_test_filtered_split, y_train_filtered_split, y_test_filtered_split = train_test_split(X_train_filtered, y_train_filtered, test_size=0.3, random_state=self.random_state, stratify=y_train_filtered)

        self.base_classifier = copy.deepcopy(self.base_classifier_template)
        self.base_classifier.fit(X_train_filtered_split, y_train_filtered_split)
        predictions = self.base_classifier.predict(X_test_filtered_split)
        predictions_proba = self.base_classifier.predict_proba(X_test_filtered_split)[:, 1]
        score = self._calculate_metric(y_test_filtered_split, predictions, predictions_proba)
        print(f"baseline score:{score}")

        best_score = score
        best_mode = None
        modes = ['generate_pseudo_labels', 'enhance_training_set']
        for mode in modes:
            self.mode = mode
            X_train_split_updated, y_train_split_updated, _, _ = self._enhance_training_set(X_train_filtered_split, y_train_filtered_split, X_unlabeled, X_test_filtered_split, y_test_filtered_split)

            self.base_classifier = copy.deepcopy(self.base_classifier_template)
            self.base_classifier.fit(X_train_split_updated, y_train_split_updated)
            predictions = self.base_classifier.predict(X_test_filtered_split)
            predictions_proba = self.base_classifier.predict_proba(X_test_filtered_split)[:, 1]
            score = self._calculate_metric(y_test_filtered_split, predictions, predictions_proba)
            print(f"{self.mode} score:{score}")

            if score > best_score:
                best_score = score
                best_mode = mode

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
        return self.base_classifier.predict(X)

    def predict_proba(self, X):
        return self.base_classifier.predict_proba(X)

    def _balance_data(self, X, y):
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
        self.base_classifier = copy.deepcopy(self.base_classifier_template)
        self.base_classifier.fit(X_train, y_train)
        y_pred_proba = self.base_classifier.predict_proba(X_train)

        y_pred_copy = y_pred_proba.copy()
        np.put_along_axis(y_pred_copy, np.argmax(y_pred_proba, axis=1)[:, None], 0, axis=1)

        diff = np.max(y_pred_proba, axis=1) - np.max(y_pred_copy, axis=1)

        filter_mask = diff < difficulty_threshold

        retain_samples_mask = np.zeros(len(y_train), dtype=bool)

        for label, proportion in original_label_proportions.items():
            label_mask = (y_train == label) & filter_mask
            n_retain = int(np.sum(label_mask) * proportion)
            retain_indices = np.random.choice(np.where(label_mask)[0], n_retain, replace=False)
            retain_samples_mask[retain_indices] = True

        final_retain_mask = ~filter_mask | retain_samples_mask
        X_train_filtered = X_train[final_retain_mask]
        y_train_filtered = y_train[final_retain_mask]

        return X_train_filtered, y_train_filtered

    def find_optimal_threshold_and_refilter(self, X_combined, y_combined, misclassified_samples_best, misclassified_labels_best, original_label_proportions):
        optimal_threshold = None
        best_performance = -np.inf
        for threshold in np.arange(0, 0.2, 0.04):
            X_filtered, y_filtered = self._filter_difficult_samples(X_combined, y_combined, original_label_proportions, difficulty_threshold=threshold)
            self.base_classifier = copy.deepcopy(self.base_classifier_template)
            self.base_classifier.fit(X_filtered, y_filtered)
            predictions = self.base_classifier.predict(misclassified_samples_best)
            predictions_proba = self.base_classifier.predict_proba(misclassified_samples_best)[:, 1]
            performance = self._calculate_metric(misclassified_labels_best, predictions, predictions_proba)

            if performance > best_performance:
                best_performance = performance
                optimal_threshold = threshold

        print(f"Optimal difficulty threshold found: {optimal_threshold} with performance: {best_performance}")

        X_combined_filtered, y_combined_filtered = self._filter_difficult_samples(X_combined, y_combined, original_label_proportions, difficulty_threshold=optimal_threshold)

        X_filtered = pd.concat([X_combined_filtered, misclassified_samples_best])
        y_filtered = pd.concat([y_combined_filtered, misclassified_labels_best])

        return X_filtered, y_filtered

    def _enhance_training_set(self, X_train, y_train, X_unlabeled, X_test=None, y_test=None, target_sample_percent=0.3):
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
                X_train_part, X_test_part, y_train_part, y_test_part = train_test_split(X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train)

            self.base_classifier = copy.deepcopy(self.base_classifier_template)
            self.base_classifier.fit(X_train_part, y_train_part)
            baseline_metric = self._calculate_metric(y_test_part, self.base_classifier.predict(X_test_part), self.base_classifier.predict_proba(X_test_part)[:, 1])

            improvement = True
            X_enhanced = X_train.iloc[:1].copy()
            y_enhanced = y_train.iloc[:1].copy()

            while improvement and len(X_unlabeled) > 0:
                y_pred_proba = self.base_classifier.predict_proba(X_unlabeled)[:, 1]
                proba_indices = np.argsort(y_pred_proba)[::-1]

                top_sample_indices = proba_indices[:int(len(proba_indices) * target_sample_percent)]
                X_selected = X_unlabeled.iloc[top_sample_indices]
                y_selected_pred = self.base_classifier.predict(X_selected)

                for column in X_selected.columns:
                    if X_selected[column].dtype != X_train_part[column].dtype:
                        X_selected[column] = X_selected[column].astype(X_train_part[column].dtype)

                X_temp_combined = pd.concat([X_train_part, X_enhanced, X_selected]).reset_index(drop=True)
                y_temp_combined = pd.concat([y_train_part, y_enhanced, pd.Series(y_selected_pred, index=X_selected.index)]).reset_index(drop=True)

                self.base_classifier = copy.deepcopy(self.base_classifier_template)
                self.base_classifier.fit(X_temp_combined, y_temp_combined)

                updated_metric = self._calculate_metric(y_test_part, self.base_classifier.predict(X_test_part), self.base_classifier.predict_proba(X_test_part)[:, 1])

                if updated_metric > baseline_metric:
                    print("Performance improved with added samples.")
                    baseline_metric = updated_metric
                    X_enhanced = pd.concat([X_enhanced, X_selected])
                    y_enhanced = pd.concat([y_enhanced, pd.Series(y_selected_pred, index=X_selected.index)])
                    X_unlabeled = X_unlabeled.drop(X_selected.index)
                else:
                    print("No performance improvement with added samples.")
                    improvement = False

            if len(X_enhanced) > 1:
                X_enhanced = X_enhanced.iloc[1:].reset_index(drop=True)
                y_enhanced = y_enhanced.iloc[1:].reset_index(drop=True)
                X_train_final = pd.concat([X_train, X_enhanced]).reset_index(drop=True)
                y_train_final = pd.concat([y_train, y_enhanced]).reset_index(drop=True)
            else:
                X_train_final = X_train
                y_train_final = y_train

            return X_train_final, y_train_final, None, None

        else:
            raise ValueError("Invalid mode specified.")
        