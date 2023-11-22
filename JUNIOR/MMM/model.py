from sklearn.linear_model import LinearRegression
from typing import Tuple, Dict

import pandas as pd


def linreg_total_sales(
    sales: pd.DataFrame, ad_costs: pd.DataFrame
) -> Tuple[float, Dict[str, float]]:
    """
    Fit linear regression model for total sales and ad costs
    Return R2 score and dict with coefficients and intercept
    Example:
    >>> linreg_total_sales(df_sales, df_ad)
    (0.948, {
        'TV': 0.3, 
        'SMM': 0.6, 
        'Website banners': 1.0, 
        'Google Ads': 0.5, 
        'intercept': 452.0
        }
    )
    """
    df_sales = sales.copy()
    df_costs = ad_costs.copy()
    df_y = df_sales.groupby('day').sum('sales')
    data = df_costs.merge(df_y, on='day')
    columns = ['TV', 'SMM', 'Website banners', 'Google Ads']
    X = data[columns].to_numpy()
    y = data[['sales']].to_numpy()
    reg = LinearRegression().fit(X, y)
    values = reg.coef_.flatten().tolist()
    coef = dict(zip(columns, values))
    coef['intercept'] = reg.intercept_[0]
    r2 = reg.score(X, y)
    return r2, coef


def linreg_category_sales(
    sales: pd.DataFrame, ad_costs: pd.DataFrame
) -> Dict[str, Tuple[float, Dict[str, float]]]:
    """
    Fit linear regression model for sales by category and ad costs
    Return R2 score and dict with coefficients and intercept for each category
    Example:
    >>> linreg_category_sales(df_sales, df_ad)
    {
        'Electronic': (0.948, {
            'TV': 0.3, 'SMM': 0.6, 
            'Website banners': 1.0, 
            'Google Ads': 0.5, 'intercept': 452.0
            }),
        'Fashion': (0.567, {
            'TV': 0.2, 'SMM': 0.3, 
            'Website banners': 7.0, 
            'Google Ads': 0.1, 
            'intercept': 527.0
            }),
    """
    result = {}
    grouped = sales.groupby('category')
    for category, df_cat in grouped:
        result[category] = linreg_total_sales(df_cat, ad_costs)
    return result
