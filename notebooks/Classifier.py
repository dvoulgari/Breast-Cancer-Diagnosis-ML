import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (f1_score, balanced_accuracy_score, precision_score,
                            matthews_corrcoef, recall_score, roc_auc_score,
                            fbeta_score, confusion_matrix)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

warnings.filterwarnings('ignore', category=UserWarning)  # Suppress lightGBM warnings

class ClassifierCV(BaseEstimator, ClassifierMixin):
    """Robust classifier evaluation with nested CV and preprocessing safety."""
    
    def __init__(self, classifier_name, class_weight={1: 1, 0: 1}, tuning_beta=1,
                 seed=42, outer_folds=5, inner_folds=3, optimization_trials=50,
                 trial_num=10, shuffle=True, verbose=True):
        self.classifier_name = classifier_name
        self.class_weight = class_weight
        self.tuning_beta = tuning_beta
        self.seed = seed
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.optimization_trials = optimization_trials
        self.trial_num = trial_num
        self.shuffle = shuffle
        self.verbose = verbose
        self.scaler = StandardScaler()
        self.fold_metrics = []
        self.model = None
        self.best_params_ = None
        np.random.seed(seed)

    def class_specific_impute(self, X, y):
        """Impute missing values using class-specific medians."""
        X = pd.DataFrame(X).copy()
        y = np.array(y)

        for cls in np.unique(y):
            mask = (y == cls)
            medians = X.loc[mask].median()
            X.loc[mask] = X.loc[mask].fillna(medians)

        return X.values

    def _initialize_model(self):
        """Initialize the model with default parameters."""
        if self.classifier_name == "LogisticRegression-elastic":
            self.model = LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                class_weight=self.class_weight,
                random_state=self.seed
            )
        elif self.classifier_name == "RandomForest":
            self.model = RandomForestClassifier(
                class_weight=self.class_weight,
                random_state=self.seed
            )
        elif self.classifier_name == "SVM":
            self.model = SVC(
                class_weight=self.class_weight,
                probability=True,
                random_state=self.seed
            )
        elif self.classifier_name == "LightGBM":
            self.model = lgb.LGBMClassifier(
                class_weight=self.class_weight,
                random_state=self.seed,
                verbosity=-1,
                force_row_wise=True
            )
        elif self.classifier_name == "GaussianNB":
            self.model = GaussianNB()
        elif self.classifier_name == "LDA":
            self.model = LinearDiscriminantAnalysis()
        else:
            raise ValueError(f"Classifier {self.classifier_name} not implemented")

    def _objective(self, trial, X_train, y_train, inner_cv):
        """Optuna objective function with full hyperparameter spaces."""
        if self.classifier_name == "LogisticRegression-elastic":
            params = {
                'C': trial.suggest_float('C', 1e-4, 100.0, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0)
            }
            model = LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                **params,
                class_weight=self.class_weight,
                random_state=self.seed
            )
            
        elif self.classifier_name == "RandomForest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            }
            model = RandomForestClassifier(
                **params,
                class_weight=self.class_weight,
                random_state=self.seed
            )
            
        elif self.classifier_name == "SVM":
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
            params = {
                'C': trial.suggest_float('C', 1e-4, 100.0, log=True),
                'kernel': kernel,
                'gamma': trial.suggest_float('gamma', 1e-5, 10.0, log=True) if kernel in ['poly', 'rbf', 'sigmoid'] else 'scale',
                'degree': trial.suggest_int('degree', 2, 5) if kernel == 'poly' else 3
            }
            model = SVC(
                **params,
                class_weight=self.class_weight,
                probability=True,
                random_state=self.seed
            )
            
        elif self.classifier_name == "LightGBM":
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0)
            }
            model = lgb.LGBMClassifier(
                **params,
                class_weight=self.class_weight,
                random_state=self.seed,
                verbosity=-1,
                force_row_wise=True
            )
            
        elif self.classifier_name == "GaussianNB":
            model = GaussianNB()
            
        elif self.classifier_name == "LDA":
            solver = trial.suggest_categorical('solver', ['svd', 'lsqr', 'eigen'])
            params = {
                'solver': solver,
                'shrinkage': None if solver == 'svd' else trial.suggest_float('shrinkage', 0.0, 1.0)
            }
            model = LinearDiscriminantAnalysis(**params)
            
        else:
            raise ValueError(f"Classifier {self.classifier_name} not implemented")

        scores = []
        for inner_train_idx, inner_test_idx in inner_cv.split(X_train, y_train):
            model.fit(X_train[inner_train_idx], y_train[inner_train_idx])
            y_pred = model.predict(X_train[inner_test_idx])
            scores.append(fbeta_score(y_train[inner_test_idx], y_pred, beta=self.tuning_beta))
            
        return np.mean(scores)

    def fit_transform(self, X_raw, y_raw):
        """Perform nested CV with preprocessing and hyperparameter tuning."""
        self.trials_seeds = np.random.randint(0, 1000, size=self.trial_num)
        all_metrics = []
        all_params = []
        
        for seed in self.trials_seeds:
            outer_cv = StratifiedKFold(n_splits=self.outer_folds, shuffle=self.shuffle, random_state=seed)
            inner_cv = StratifiedKFold(n_splits=self.inner_folds, shuffle=self.shuffle, random_state=seed)

            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_raw, y_raw)):
                # Preprocessing
                X_train_raw, y_train = X_raw[train_idx], y_raw[train_idx]
                X_test_raw, y_test = X_raw[test_idx], y_raw[test_idx]
                
                X_train_imp = self.class_specific_impute(X_train_raw, y_train)
                self.scaler.fit(X_train_imp)
                X_train = self.scaler.transform(X_train_imp)
                X_test = self.scaler.transform(self.class_specific_impute(X_test_raw, y_test))

                # Optimization
                self._initialize_model()
                if self.classifier_name != "GaussianNB":
                    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
                    study.optimize(lambda trial: self._objective(trial, X_train, y_train, inner_cv),
                                 n_trials=self.optimization_trials)
                    self.best_params_ = study.best_params
                    self._initialize_model()  # Reinitialize with best params
                    if hasattr(self.model, 'set_params'):
                        self.model.set_params(**study.best_params)

                # Evaluation
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
                
                # Calculate all metrics
                cm = confusion_matrix(y_test, y_pred)
                metrics = {
                    'Fold': fold_idx,
                    'Classifier': self.classifier_name,
                    'F1': f1_score(y_test, y_pred),
                    'Balanced_Accuracy': balanced_accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred),
                    'MCC': matthews_corrcoef(y_test, y_pred),
                    'Recall': recall_score(y_test, y_pred),
                    'ROC_AUC': roc_auc_score(y_test, y_pred),
                    'F2': fbeta_score(y_test, y_pred, beta=self.tuning_beta),
                    'Specificity': cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0,
                    'Negative_PV': cm[0,0]/(cm[0,0]+cm[1,0]) if (cm[0,0]+cm[1,0]) > 0 else 0,
                    'Best_Params': str(self.best_params_) if self.best_params_ else 'None'
                }
                all_metrics.append(metrics)
                if self.best_params_:
                    all_params.append(self.best_params_)

                if self.verbose:
                    print(f"Fold {fold_idx} - F1: {metrics['F1']:.3f}, ROC AUC: {metrics['ROC_AUC']:.3f}, MCC: {metrics['MCC']:.3f}, F2: {metrics['F2']:.3f}, Best Params: {metrics['Best_Params']}")

        # Create comprehensive results
        results_df = pd.DataFrame(all_metrics)
        self.fold_metrics = results_df
        
        if self.verbose:
            print("\n=== Metrics ===")
            agg_df = results_df.groupby('Classifier').agg({
                'F1': ['mean', 'std'],
                'F2': ['mean', 'std'],
                'MCC': ['mean', 'std'],
                'ROC_AUC': ['mean', 'std'],
                'Balanced_Accuracy': ['mean', 'std'],
                'Precision': ['mean', 'std'],
                'Recall': ['mean', 'std'],
                'Specificity': ['mean', 'std'],
                'Negative_PV': ['mean', 'std']
            })
            print(agg_df)
            
        return results_df, all_params

    def fit(self, X_raw, y_raw):
        """Fit final model on full data with optimal parameters."""
        X_imp = self.class_specific_impute(X_raw, y_raw)
        X_scaled = self.scaler.fit_transform(X_imp)
        
        if self.classifier_name != "GaussianNB":
            inner_cv = StratifiedKFold(n_splits=self.inner_folds, shuffle=self.shuffle, random_state=self.seed)
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.seed))
            study.optimize(lambda trial: self._objective(trial, X_scaled, y_raw, inner_cv),
                         n_trials=self.optimization_trials)
            self.best_params_ = study.best_params
            self._initialize_model()
            if hasattr(self.model, 'set_params'):
                self.model.set_params(**study.best_params)
        
        self.model.fit(X_scaled, y_raw)
        return self

    def predict(self, X_raw):
        """Predict with preprocessing."""
        X_imp = self.class_specific_impute(X_raw, np.zeros(len(X_raw)))
        X_scaled = self.scaler.transform(X_imp)
        return self.model.predict(X_scaled)

    def score(self, X_raw, y_true):
        """Calculate F1 and ROC AUC scores."""
        y_pred = self.predict(X_raw)
        return {
            'F1': f1_score(y_true, y_pred),
            'F2': fbeta_score(y_true, y_pred, beta=self.tuning_beta),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'ROC_AUC': roc_auc_score(y_true, y_pred),
            'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'Specificity': confusion_matrix(y_true, y_pred)[0,0] / (confusion_matrix(y_true, y_pred)[0,0] + confusion_matrix(y_true, y_pred)[0,1]),
            'Negative_PV': confusion_matrix(y_true, y_pred)[0,0] / (confusion_matrix(y_true, y_pred)[0,0] + confusion_matrix(y_true, y_pred)[1,0])
        }

    def save_metrics(self, filename):
        """Save fold metrics to CSV."""
        self.fold_metrics.to_csv(filename, index=False)