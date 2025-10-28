import os
import numpy as np
import pandas as pd


def load_raw(train_path: str = 'train.csv', test_path: str = 'test.csv') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw Kaggle House Prices datasets and set index to 'Id'.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if 'Id' in train_df.columns:
        train_df.set_index('Id', inplace=True)
    if 'Id' in test_df.columns:
        test_df.set_index('Id', inplace=True)
    return train_df, test_df


def log_transform_target(train_df: pd.DataFrame, target: str = 'SalePrice') -> pd.Series:
    """
    Apply log1p to SalePrice and return original copy for comparison.
    Mutates train_df[target].
    """
    original = train_df[target].copy()
    train_df[target] = np.log1p(train_df[target])
    return original


def merge_for_processing(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate train and test on columns up to and including 'SaleCondition' as in original flow.
    """
    # If 'SaleCondition' is missing (defensive), concat full frames (excluding target in test).
    if 'SaleCondition' in train_df.columns and 'SaleCondition' in test_df.columns:
        all_data = pd.concat((train_df.loc[:, :'SaleCondition'], test_df.loc[:, :'SaleCondition']))
    else:
        drop_cols = []
        if 'SalePrice' in test_df.columns:
            drop_cols = ['SalePrice']
        all_data = pd.concat((train_df.drop(columns=[], errors='ignore'), test_df.drop(columns=drop_cols, errors='ignore')))
    return all_data


def impute_missing_values(all_data: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values per the grouped strategy used in the notebook.
    Returns a new DataFrame.
    """
    df = all_data.copy()

    # Group 1: numerical zero-fill
    for col in ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                'GarageCars', 'GarageArea', 'BsmtFullBath', 'BsmtHalfBath']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # LotFrontage by Neighborhood median
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    # Group 3: categorical 'None'
    for col in ['Alley', 'Fence', 'MiscFeature', 'PoolQC', 'FireplaceQu', 'GarageType',
                'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
                'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    # Group 4: mode-fill common categoricals
    for col in ['Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType',
                'Utilities', 'Functional', 'MSZoning']:
        if col in df.columns:
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else None
            if mode_val is not None:
                df[col] = df[col].fillna(mode_val)

    # Group 5: remaining numeric like GarageYrBlt
    if 'GarageYrBlt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)

    # Fallback: any remaining missing
    missing_cols = df.columns[df.isnull().any()].tolist()
    for col in missing_cols:
        if df[col].dtype == 'O':
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)
        else:
            df[col] = df[col].fillna(df[col].median())

    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode object dtype columns. Drop first to prevent multicollinearity.
    """
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) == 0:
        return df
    return pd.get_dummies(df, columns=list(categorical_cols), drop_first=True)


def split_train_test(all_encoded: pd.DataFrame, train_rows: int,
                     train_df_with_target: pd.DataFrame, target: str = 'SalePrice') -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Split back into X (train), y (log target from train_df), and X_test.
    """
    X = all_encoded.iloc[:train_rows, :].copy()
    X_test = all_encoded.iloc[train_rows:, :].copy()
    y = train_df_with_target[target].copy()
    return X, y, X_test


