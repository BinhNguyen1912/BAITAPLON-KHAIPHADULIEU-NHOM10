import numpy as np
import pandas as pd


def add_basic_features(all_data: pd.DataFrame) -> pd.DataFrame:
    df = all_data.copy()
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath'] = (df['FullBath'] + (0.5 * df['HalfBath']) +
                       df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))
    df['Age'] = df['YrSold'] - df['YearBuilt'] if 'YrSold' in df.columns else (2025 - df['YearBuilt'])
    return df


def add_quality_and_condition(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['OverallGrade'] = out['OverallQual'] * out['OverallCond']
    out['QualCondRatio'] = (out['OverallQual'] + 1) / (out['OverallCond'] + 1)
    exterior_qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    if 'ExterQual' in out.columns:
        out['ExterQualScore'] = out['ExterQual'].map(exterior_qual_map)
    if 'ExterCond' in out.columns:
        out['ExterCondScore'] = out['ExterCond'].map(exterior_qual_map)
    return out


def add_space_and_ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['TotalPorchSF'] = (out['OpenPorchSF'] + out['EnclosedPorch'] +
                           out['3SsnPorch'] + out['ScreenPorch'])
    out['LivingAreaRatio'] = out['GrLivArea'] / (out['LotArea'] + 1)
    out['RoomArea'] = out['GrLivArea'] / (out['TotRmsAbvGrd'] + 1)
    out['BedroomRatio'] = out['BedroomAbvGr'] / (out['TotRmsAbvGrd'] + 1)
    out['BathroomRatio'] = out['TotalBath'] / (out['TotRmsAbvGrd'] + 1)
    return out


def add_flags_and_time(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['HasBasement'] = (out['TotalBsmtSF'] > 0).astype(int)
    out['HasGarage'] = (out['GarageArea'] > 0).astype(int)
    out['HasPool'] = (out['PoolArea'] > 0).astype(int) if 'PoolArea' in out.columns else 0
    out['HasFireplace'] = (out['Fireplaces'] > 0).astype(int)
    out['HasSecondFloor'] = (out['2ndFlrSF'] > 0).astype(int)
    out['IsRemodeled'] = (out['YearRemodAdd'] != out['YearBuilt']).astype(int)
    out['YearsSinceRemodel'] = out['YrSold'] - out['YearRemodAdd'] if 'YrSold' in out.columns else (2025 - out['YearRemodAdd'])
    out['IsNew'] = (out.get('YrSold', out['YearBuilt']) == out['YearBuilt']).astype(int)

    # Age groups numeric
    out['AgeGroup_Num'] = pd.cut(out['Age'],
                                 bins=[-1, 5, 15, 30, 50, 100, 200],
                                 labels=[1, 2, 3, 4, 5, 6])

    # Season features
    def get_season_num(month):
        if month in [12, 1, 2]:
            return 1
        if month in [3, 4, 5]:
            return 2
        if month in [6, 7, 8]:
            return 3
        return 4

    if 'MoSold' in out.columns:
        out['SeasonSold_Num'] = out['MoSold'].apply(get_season_num)
        out['IsSummerSale'] = out['MoSold'].isin([6, 7, 8]).astype(int)
    return out


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['QualPerSF'] = out['OverallQual'] / (out['TotalSF'] + 1)
    out['BathPerBedroom'] = out['TotalBath'] / (out['BedroomAbvGr'] + 1)
    out['GarageCarsPerArea'] = out['GarageCars'] / (out['GarageArea'] + 1)
    out['LivingAreaPerBedroom'] = out['GrLivArea'] / (out['BedroomAbvGr'] + 1)
    if 'LotConfig' in out.columns:
        out['IsCornerLot'] = (out['LotConfig'] == 'Corner').astype(int)
        out['IsCulDSac'] = (out['LotConfig'] == 'CulDSac').astype(int)
    return out


def engineer_features(all_data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline.
    """
    df = add_basic_features(all_data)
    df = add_quality_and_condition(df)
    df = add_space_and_ratios(df)
    df = add_flags_and_time(df)
    df = add_interactions(df)
    return df


