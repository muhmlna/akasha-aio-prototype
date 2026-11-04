"""
Generate synthetic Akasha sales data for prototype demonstration.
Simulates 18 months of daily sales for top 5 SKUs across Jakarta/Surabaya.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from hijri_converter import Hijri, Gregorian

np.random.seed(42)


def generate_ramadan_dates(start_date, end_date):
    """Generate Ramadan periods within date range (shifts ~11 days earlier each year)."""
    ramadan_periods = []
    
    # 2025 Ramadan: March 1 - March 29
    ramadan_2025_start = datetime(2025, 3, 1)
    ramadan_2025_end = datetime(2025, 3, 29)
    
    # 2026 Ramadan: Feb 18 - March 19 (11 days earlier)
    ramadan_2026_start = datetime(2026, 2, 18)
    ramadan_2026_end = datetime(2026, 3, 19)
    
    if ramadan_2025_start >= start_date and ramadan_2025_end <= end_date:
        ramadan_periods.append((ramadan_2025_start, ramadan_2025_end))
    if ramadan_2026_start >= start_date and ramadan_2026_end <= end_date:
        ramadan_periods.append((ramadan_2026_start, ramadan_2026_end))
    
    return ramadan_periods


def is_ramadan(date, ramadan_periods):
    """Check if date falls within Ramadan."""
    for start, end in ramadan_periods:
        if start <= date <= end:
            return True
    return False


def is_payday_window(date):
    """Check if date falls in payday window (25th-5th)."""
    day = date.day
    return day >= 25 or day <= 5


def generate_sales_data():
    """Generate 18 months of synthetic sales data."""
    
    # Date range: Jan 2025 - Jun 2026 (18 months)
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2026, 6, 30)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Top 5 SKUs
    skus = [
        {'sku_id': 'SKU001', 'name': 'Nestle Pure Life 600ml', 'category': 'Water', 'base_demand': 450},
        {'sku_id': 'SKU002', 'name': 'Nestle Pure Life 1.5L', 'category': 'Water', 'base_demand': 320},
        {'sku_id': 'SKU003', 'name': 'Fruity Juice Orange 1L', 'category': 'Juice', 'base_demand': 180},
        {'sku_id': 'SKU004', 'name': 'Makarizo Hair Serum 100ml', 'category': 'Beauty', 'base_demand': 85},
        {'sku_id': 'SKU005', 'name': 'Makarizo Shampoo 250ml', 'category': 'Beauty', 'base_demand': 110},
    ]
    
    # Branches
    branches = [
        {'branch_id': 'JKT01', 'location': 'Jakarta', 'region': 'Jakarta'},
        {'branch_id': 'SBY01', 'location': 'Surabaya', 'region': 'Surabaya'},
    ]
    
    ramadan_periods = generate_ramadan_dates(start_date, end_date)
    
    records = []
    
    for sku in skus:
        for branch in branches:
            for date in dates:
                # Base demand
                base = sku['base_demand']
                
                # Regional modifier (Jakarta 20% higher than Surabaya)
                if branch['location'] == 'Jakarta':
                    base *= 1.2
                
                # Ramadan boost (+40% for Water/Juice, +15% for Beauty)
                ramadan_multiplier = 1.0
                if is_ramadan(date, ramadan_periods):
                    if sku['category'] in ['Water', 'Juice']:
                        ramadan_multiplier = 1.4
                    else:
                        ramadan_multiplier = 1.15
                
                # Payday boost (+12%)
                payday_multiplier = 1.12 if is_payday_window(date) else 1.0
                
                # Weekend effect (-8% for weekdays)
                weekend_multiplier = 1.0 if date.weekday() >= 5 else 0.92
                
                # Weekly seasonality
                weekly_pattern = 1.0 + 0.05 * np.sin(2 * np.pi * date.weekday() / 7)
                
                # Calculate true demand
                true_demand = base * ramadan_multiplier * payday_multiplier * weekend_multiplier * weekly_pattern
                
                # Add noise
                noise = np.random.normal(1.0, 0.15)
                true_demand = max(0, int(true_demand * noise))
                
                # Simulate stockouts (5% chance, demand becomes censored)
                stockout = np.random.random() < 0.05
                observed_demand = 0 if stockout else true_demand
                
                records.append({
                    'date': date,
                    'sku_id': sku['sku_id'],
                    'sku_name': sku['name'],
                    'category': sku['category'],
                    'branch_id': branch['branch_id'],
                    'location': branch['location'],
                    'region': branch['region'],
                    'true_demand': true_demand,  # Ground truth (unknown in real scenario)
                    'observed_demand': observed_demand,  # What EBS records
                    'stockout_flag': stockout,
                    'is_ramadan': is_ramadan(date, ramadan_periods),
                    'is_payday_window': is_payday_window(date),
                    'is_weekend': date.weekday() >= 5,
                    'day_of_week': date.weekday(),
                })
    
    df = pd.DataFrame(records)
    return df


if __name__ == '__main__':
    df = generate_sales_data()
    
    # Save to CSV
    df.to_csv('data/synthetic_sales.csv', index=False)
    
    print(f"Generated {len(df)} records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"SKUs: {df['sku_id'].nunique()}")
    print(f"Branches: {df['branch_id'].nunique()}")
    print(f"Stockout rate: {df['stockout_flag'].mean():.1%}")
    print(f"\nSample records:")
    print(df.head(10))
