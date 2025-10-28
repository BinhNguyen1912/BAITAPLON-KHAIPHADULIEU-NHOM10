import time
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def get_models() -> Dict[str, object]:
    return {
        'Random Forest': RandomForestRegressor(
            n_estimators=500,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=1000,
            learning_rate=0.02,
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=42
        ),
        'XGBoost': XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        'LightGBM': LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.025,
            max_depth=8,
            num_leaves=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    }


def evaluate_models(models: Dict[str, object], X_train: pd.DataFrame, X_val: pd.DataFrame,
                    y_train: pd.Series, y_val: pd.Series, cv_folds: int = 5) -> Dict[str, dict]:
    results = {}
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        y_pred = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae = float(mean_absolute_error(y_val, y_pred))
        r2 = float(r2_score(y_val, y_pred))
        cv_scores_rmse = cross_val_score(model, X_train, y_train, cv=kf,
                                         scoring='neg_mean_squared_error', n_jobs=-1)
        cv_rmse = float(np.sqrt(-cv_scores_rmse.mean()))
        cv_std = float(np.sqrt(-cv_scores_rmse).std())
        results[name] = {
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'cv_rmse': cv_rmse,
            'cv_std': cv_std,
            'training_time': train_time,
            'predictions': y_pred
        }
    return results


def pick_best(results: Dict[str, dict], metric: str = 'rmse') -> Tuple[str, object, dict]:
    if not results:
        raise ValueError('No results to choose from')
    best_name = sorted(results.items(), key=lambda kv: kv[1][metric])[0][0]
    best_model = results[best_name]['model']
    return best_name, best_model, results[best_name]


