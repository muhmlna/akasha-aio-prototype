"""
Model training and evaluation: Baseline vs. LightGBM.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta


def calculate_wape(y_true, y_pred):
    """Weighted Absolute Percentage Error."""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100


def train_test_split_temporal(df, test_months=2):
    """
    Split data temporally (no shuffling for time-series).
    
    Args:
        df: Feature-engineered DataFrame
        test_months: Number of months for test set
    
    Returns:
        train_df, test_df
    """
    df = df.sort_values('date')
    split_date = df['date'].max() - timedelta(days=test_months*30)
    
    train = df[df['date'] < split_date]
    test = df[df['date'] >= split_date]
    
    print(f"Train: {train['date'].min()} to {train['date'].max()} ({len(train)} records)")
    print(f"Test: {test['date'].min()} to {test['date'].max()} ({len(test)} records)")
    
    return train, test


def baseline_ramadan_aligned_naive(train_df, test_df):
    """
    Ramadan-aligned Seasonal Naive baseline.
    Uses demand from 365 days ago, adjusted for Hijri calendar shift.
    """
    predictions = []
    
    for idx, row in test_df.iterrows():
        # Use lag_365d feature (already aligned in feature engineering)
        pred = row['lag_365d']
        
        # Fallback to rolling mean if lag_365d is NaN
        if pd.isna(pred):
            pred = row['rolling_mean_30d']
        
        predictions.append(pred)
    
    return np.array(predictions)


def train_lightgbm(train_df, test_df):
    """
    Train LightGBM model with Indonesian features.
    
    Returns:
        model, predictions, metrics
    """
    feature_cols = [
        'month', 'day_of_week', 'week_of_year', 'is_weekend',
        'is_ramadan', 'hijri_month', 'days_to_ramadan',
        'is_payday_window',
        'lag_7d', 'lag_14d', 'lag_30d', 'lag_365d',
        'rolling_mean_7d', 'rolling_mean_30d',
    ]
    
    # Add categorical encoding for SKU and branch
    train_df['sku_encoded'] = train_df['sku_id'].astype('category').cat.codes
    train_df['branch_encoded'] = train_df['branch_id'].astype('category').cat.codes
    test_df['sku_encoded'] = test_df['sku_id'].astype('category').cat.codes
    test_df['branch_encoded'] = test_df['branch_id'].astype('category').cat.codes
    
    feature_cols += ['sku_encoded', 'branch_encoded']
    
    X_train = train_df[feature_cols]
    y_train = train_df['imputed_demand']
    X_test = test_df[feature_cols]
    y_test = test_df['imputed_demand']
    
    print(f"\nTraining LightGBM on {len(X_train)} samples...")
    print(f"Features: {len(feature_cols)}")
    
    # LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Train
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )
    
    # Predict
    predictions = model.predict(X_test, num_iteration=model.best_iteration)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    return model, predictions, feature_importance


def evaluate_models(test_df, baseline_preds, lgb_preds):
    """
    Compare baseline vs. LightGBM performance.
    
    Returns:
        Dictionary of metrics
    """
    y_true = test_df['imputed_demand'].values
    
    # Baseline metrics
    baseline_mae = mean_absolute_error(y_true, baseline_preds)
    baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_preds))
    baseline_wape = calculate_wape(y_true, baseline_preds)
    
    # LightGBM metrics
    lgb_mae = mean_absolute_error(y_true, lgb_preds)
    lgb_rmse = np.sqrt(mean_squared_error(y_true, lgb_preds))
    lgb_wape = calculate_wape(y_true, lgb_preds)
    
    # Improvement
    wape_improvement = ((baseline_wape - lgb_wape) / baseline_wape) * 100
    
    results = {
        'baseline': {
            'MAE': baseline_mae,
            'RMSE': baseline_rmse,
            'WAPE': baseline_wape,
        },
        'lightgbm': {
            'MAE': lgb_mae,
            'RMSE': lgb_rmse,
            'WAPE': lgb_wape,
        },
        'improvement': {
            'WAPE_reduction_%': wape_improvement,
        }
    }
    
    return results


if __name__ == '__main__':
    # Load featured data
    df = pd.read_csv('data/sales_featured.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print("Data shape:", df.shape)
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Train-test split (last 2 months as test)
    train_df, test_df = train_test_split_temporal(df, test_months=2)
    
    # Baseline: Ramadan-aligned Seasonal Naive
    print("\n=== Baseline: Ramadan-Aligned Seasonal Naive ===")
    baseline_preds = baseline_ramadan_aligned_naive(train_df, test_df)
    
    # LightGBM
    print("\n=== LightGBM Model ===")
    model, lgb_preds, feature_importance = train_lightgbm(train_df, test_df)
    
    # Evaluation
    print("\n=== Model Comparison ===")
    results = evaluate_models(test_df, baseline_preds, lgb_preds)
    
    print("\nBaseline (Ramadan-Aligned Seasonal Naive):")
    for metric, value in results['baseline'].items():
        print(f"  {metric}: {value:.2f}")
    
    print("\nLightGBM:")
    for metric, value in results['lightgbm'].items():
        print(f"  {metric}: {value:.2f}")
    
    print("\nImprovement:")
    print(f"  WAPE Reduction: {results['improvement']['WAPE_reduction_%']:.1f}%")
    
    print("\n=== Top 10 Feature Importance ===")
    print(feature_importance.head(10))
    
    # Save results
    results_df = pd.DataFrame({
        'model': ['Baseline', 'LightGBM'],
        'MAE': [results['baseline']['MAE'], results['lightgbm']['MAE']],
        'RMSE': [results['baseline']['RMSE'], results['lightgbm']['RMSE']],
        'WAPE': [results['baseline']['WAPE'], results['lightgbm']['WAPE']],
    })
    results_df.to_csv('data/model_comparison.csv', index=False)
    
    feature_importance.to_csv('data/feature_importance.csv', index=False)
    
    print("\nSaved results to data/model_comparison.csv and data/feature_importance.csv")
