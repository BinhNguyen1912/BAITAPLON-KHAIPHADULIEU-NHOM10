import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def plot_saleprice_distributions(train_df: pd.DataFrame, target: str = 'SalePrice') -> None:
    plt.figure(figsize=(18, 8))

    plt.subplot(1, 3, 1)
    sns.histplot(train_df[target], kde=True, bins=50, color='red')
    plt.title('SalePrice (Original)')
    plt.xlabel('SalePrice')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    stats_box = f"""
Mean: ${train_df[target].mean():,.0f}
Median: ${train_df[target].median():,.0f}
Std: ${train_df[target].std():,.0f}
Skew: {train_df[target].skew():.2f}
Kurtosis: {train_df[target].kurtosis():.2f}
"""
    plt.text(0.05, 0.95, stats_box, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='linen', alpha=0.8))
    plt.axis('off')

    plt.subplot(1, 3, 3)
    log_vals = np.log1p(train_df[target])
    sns.histplot(log_vals, kde=True, bins=50, color='green')
    plt.title('SalePrice (Log1p)')
    plt.xlabel('log1p(SalePrice)')
    plt.ylabel('Frequency')

    plt.suptitle('SalePrice Distribution: Original vs Log1p', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_outliers_saleprice(train_df: pd.DataFrame, target: str = 'SalePrice') -> None:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=train_df[target])
    plt.title('SalePrice Boxplot')
    plt.subplot(1, 2, 2)
    stats.probplot(train_df[target], dist="norm", plot=plt)
    plt.title('SalePrice Q-Q Plot')
    plt.tight_layout()
    plt.show()


def analyze_missing(df: pd.DataFrame, dataset_name: str = 'Dataset') -> pd.DataFrame | None:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print(f'No missing values in {dataset_name}')
        return None
    print(f"Missing values in {dataset_name}: {len(missing)} columns, total {missing.sum()}")
    miss_df = missing.to_frame('Missing Count')
    miss_df['Missing Percentage'] = (miss_df['Missing Count'] / len(df)) * 100

    plt.figure(figsize=(12, 8))
    head = miss_df.head(20)
    sns.barplot(x=head['Missing Percentage'], y=head.index)
    plt.title(f'Top Missing Features - {dataset_name}')
    plt.xlabel('Percentage Missing (%)')
    plt.tight_layout()
    plt.show()

    return miss_df


def plot_correlations(train_df: pd.DataFrame, target: str = 'SalePrice', top_k: int = 15) -> pd.Series:
    corr = train_df.select_dtypes(include=[np.number]).corr()
    top_corr = corr[target].abs().sort_values(ascending=False).head(top_k)
    plt.figure(figsize=(16, 14))
    top_features = top_corr.index
    sns.heatmap(corr.loc[top_features, top_features], annot=True, cmap='RdBu_r', center=0,
                fmt='.2f', linewidths=0.5)
    plt.title(f'Correlation Matrix - Top {top_k} with {target}')
    plt.tight_layout()
    plt.show()
    return top_corr


def plot_feature_relationships(train_df: pd.DataFrame, features: list[str], target: str = 'SalePrice') -> None:
    n = min(6, len(features))
    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
    axes = axes.ravel()
    for i, feature in enumerate(features[:n]):
        axes[i].scatter(train_df[feature], train_df[target], alpha=0.6)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel(target)
        axes[i].set_title(f'{feature} vs {target}')
    for j in range(n, rows * cols):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()


