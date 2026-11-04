"""
Kaplan-Meier imputation for censored demand (stockout periods).
"""

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter


def impute_censored_demand(df, sku_id, branch_id):
    """
    Impute demand during stockout periods using Kaplan-Meier survival analysis.
    
    Args:
        df: DataFrame with columns [observed_demand, stockout_flag]
        sku_id: SKU identifier
        branch_id: Branch identifier
    
    Returns:
        DataFrame with imputed_demand column
    """
    # Filter to specific SKU-branch
    mask = (df['sku_id'] == sku_id) & (df['branch_id'] == branch_id)
    subset = df[mask].copy()
    
    # Kaplan-Meier estimator
    kmf = KaplanMeierFitter()
    
    # Fit on observed demand (event_observed = NOT stockout)
    # In survival analysis: duration = demand, event = sale completed (not censored)
    kmf.fit(
        durations=subset['observed_demand'], 
        event_observed=~subset['stockout_flag']
    )
    
    # Impute stockout periods with median survival estimate
    # Median demand is the 50th percentile of the survival curve
    median_demand = kmf.percentile(0.5)
    
    # Apply imputation
    subset['imputed_demand'] = subset.apply(
        lambda row: median_demand if row['stockout_flag'] else row['observed_demand'],
        axis=1
    )
    
    return subset


def impute_all_series(df):
    """
    Apply Kaplan-Meier imputation to all SKU-branch combinations.
    
    Args:
        df: Full sales DataFrame
    
    Returns:
        DataFrame with imputed_demand for all records
    """
    imputed_dfs = []
    
    # Group by SKU-branch
    for (sku_id, branch_id), group in df.groupby(['sku_id', 'branch_id']):
        print(f"Imputing {sku_id} at {branch_id}...")
        imputed_group = impute_censored_demand(df, sku_id, branch_id)
        imputed_dfs.append(imputed_group)
    
    # Combine all imputed series
    result = pd.concat(imputed_dfs, ignore_index=True)
    return result


def evaluate_imputation_accuracy(df):
    """
    Evaluate imputation quality by comparing imputed vs. true demand.
    (Only possible with synthetic data where ground truth is known)
    
    Args:
        df: DataFrame with true_demand and imputed_demand columns
    
    Returns:
        Dictionary of accuracy metrics
    """
    # Filter to stockout periods only (where imputation was applied)
    stockout_records = df[df['stockout_flag'] == True].copy()
    
    if len(stockout_records) == 0:
        return {'error': 'No stockout records to evaluate'}
    
    # Calculate errors
    stockout_records['absolute_error'] = abs(
        stockout_records['true_demand'] - stockout_records['imputed_demand']
    )
    stockout_records['percentage_error'] = (
        stockout_records['absolute_error'] / stockout_records['true_demand'] * 100
    )
    
    metrics = {
        'num_stockouts': len(stockout_records),
        'mae': stockout_records['absolute_error'].mean(),
        'mape': stockout_records['percentage_error'].mean(),
        'median_error': stockout_records['absolute_error'].median(),
        'max_error': stockout_records['absolute_error'].max(),
    }
    
    return metrics


if __name__ == '__main__':
    # Load synthetic data
    df = pd.read_csv('data/synthetic_sales.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print("Original data:")
    print(f"Total records: {len(df)}")
    print(f"Stockout records: {df['stockout_flag'].sum()}")
    print(f"Mean observed demand: {df['observed_demand'].mean():.1f}")
    
    # Apply imputation
    df_imputed = impute_all_series(df)
    
    print("\nAfter imputation:")
    print(f"Mean imputed demand: {df_imputed['imputed_demand'].mean():.1f}")
    print(f"Mean true demand: {df_imputed['true_demand'].mean():.1f}")
    
    # Evaluate accuracy
    metrics = evaluate_imputation_accuracy(df_imputed)
    print("\nImputation accuracy on stockout periods:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Save imputed data
    df_imputed.to_csv('data/sales_imputed.csv', index=False)
    print("\nSaved imputed data to data/sales_imputed.csv")
