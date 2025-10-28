import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def regression_report(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}


def plot_metrics_comparison(metrics_before: dict, metrics_after: dict, title: str = 'Before vs After') -> None:
    labels = ['RMSE', 'MAE', 'R2', 'MAPE']
    before_vals = [metrics_before.get('rmse', 0), metrics_before.get('mae', 0), metrics_before.get('r2', 0), metrics_before.get('mape', 0)]
    after_vals = [metrics_after.get('rmse', 0), metrics_after.get('mae', 0), metrics_after.get('r2', 0), metrics_after.get('mape', 0)]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, before_vals, width, label='Before')
    ax.bar(x + width/2, after_vals, width, label='After')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_actual_vs_pred(y_true, y_pred, title: str = 'Actual vs Predicted', log_scale: bool = False) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], 'k--', linewidth=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title + (' (log)' if log_scale else ''))
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true, y_pred, title: str = 'Residuals') -> None:
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 4))
    sns.kdeplot(residuals, shade=True)
    plt.axvline(0, color='black', linestyle='--', linewidth=2)
    plt.title(title)
    plt.tight_layout()
    plt.show()


