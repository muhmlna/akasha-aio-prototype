"""
Feature engineering for Indonesian demand forecasting.
Includes Ramadan alignment, payday cycles, and lag features.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from hijri_converter import Hijri, Gregorian


def add_calendar_features(df):
    """Add basic calendar features."""
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df


def add_ramadan_features(df):
    """
    Add Ramadan-related features using Hijri calendar.
    Captures the 10-11 day annual shift.
    """
    df = df.copy()
    
    # Skip if is_ramadan already exists
    if 'is_ramadan' in df.columns:
        ramadan_dates = []
        for date in df['date']:
            greg = Gregorian(date.year, date.month, date.day)
            hijri = greg.to_hijri()
            
            if hijri.month < 9:
                days_to_ramadan = (9 - hijri.month) * 29
            elif hijri.month > 9:
                days_to_ramadan = (12 - hijri.month + 9) * 29
            else:
                days_to_ramadan = 0
            
            ramadan_dates.append({
                'hijri_month': hijri.month,
                'days_to_ramadan': days_to_ramadan,
            })
        
        ramadan_df = pd.DataFrame(ramadan_dates)
        df = pd.concat([df, ramadan_df], axis=1)
    else:
        ramadan_dates = []
        for date in df['date']:
            greg = Gregorian(date.year, date.month, date.day)
            hijri = greg.to_hijri()
            
            is_ramadan = hijri.month == 9
            
            if hijri.month < 9:
                days_to_ramadan = (9 - hijri.month) * 29
            elif hijri.month > 9:
                days_to_ramadan = (12 - hijri.month + 9) * 29
            else:
                days_to_ramadan = 0
            
            ramadan_dates.append({
                'is_ramadan': is_ramadan,
                'hijri_month': hijri.month,
                'days_to_ramadan': days_to_ramadan,
            })
        
        ramadan_df = pd.DataFrame(ramadan_dates)
        df = pd.concat([df, ramadan_df], axis=1)
    
    return df


def add_payday_features(df):
    """Add payday window features (25th-5th)."""
    df = df.copy()
    df['is_payday_window'] = df['date'].dt.day.apply(
        lambda d: 1 if d >= 25 or d <= 5 else 0
    )
    return df


def add_lag_features(df, target_col='imputed_demand', lags=[7, 14, 30]):
    """
    Add lag features per SKU-branch.
    
    Args:
        df: DataFrame sorted by date
        target_col: Column to create lags from
        lags: List of lag days
    """
    df = df.copy()
    df = df.sort_values(['sku_id', 'branch_id', 'date'])
    
    for lag in lags:
        df[f'lag_{lag}d'] = df.groupby(['sku_id', 'branch_id'])[target_col].shift(lag)
    
    # Rolling averages
    df['rolling_mean_7d'] = df.groupby(['sku_id', 'branch_id'])[target_col].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df['rolling_mean_30d'] = df.groupby(['sku_id', 'branch_id'])[target_col].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )
    
    # Year-over-year (365 days ago)
    df['lag_365d'] = df.groupby(['sku_id', 'branch_id'])[target_col].shift(365)
    
    return df


def engineer_features(df):
    """
    Apply all feature engineering steps.
    
    Args:
        df: DataFrame with imputed_demand column
    
    Returns:
        Feature-engineered DataFrame
    """
    print("Adding calendar features...")
    df = add_calendar_features(df)
    
    print("Adding Ramadan features (Hijri calendar)...")
    df = add_ramadan_features(df)
    
    print("Adding payday features...")
    df = add_payday_features(df)
    
    print("Adding lag features...")
    df = add_lag_features(df, target_col='imputed_demand')
    
    # Drop rows with NaN lags (first 365 days cannot have full features)
    print(f"Rows before dropping NaN: {len(df)}")
    df = df.dropna()
    print(f"Rows after dropping NaN: {len(df)}")
    
    return df


if __name__ == '__main__':
    # Load imputed data
    df = pd.read_csv('data/sales_imputed.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print("Original data shape:", df.shape)
    
    # Engineer features
    df_featured = engineer_features(df)
    
    print("\nFeatured data shape:", df_featured.shape)
    print("\nFeature columns:")
    feature_cols = [col for col in df_featured.columns if col not in [
        'date', 'sku_id', 'sku_name', 'category', 'branch_id', 
        'location', 'region', 'true_demand', 'observed_demand', 
        'stockout_flag', 'imputed_demand'
    ]]
    print(feature_cols)
    
    # Save
    df_featured.to_csv('data/sales_featured.csv', index=False)
    print("\nSaved to data/sales_featured.csv")
