import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from typing import Union, Dict

def rolling_correlation(
    series_x: pd.Series, 
    series_y: pd.Series, 
    window: int = 30
) -> pd.Series:
    """
    Calculates the rolling Pearson correlation between two pandas Series.

    Args:
        series_x: First time-series.
        series_y: Second time-series.
        window: Size of the moving window.

    Returns:
        pd.Series: A rolling correlation series.
    """
    return series_x.rolling(window=window).corr(series_y)

def cross_correlation(
    series_x: pd.Series, 
    series_y: pd.Series, 
    lag: int = 0
) -> float:
    """
    Calculates the correlation between series_x and series_y shifted by a lag.
    A positive lag means series_y is shifted forward, looking at the relationship
    of X today vs Y in the future (X leads Y).
    A negative lag looks at X today vs Y in the past (Y leads X).

    Args:
        series_x: Target time-series.
        series_y: Shifted time-series.
        lag: The number of periods to shift series_y.

    Returns:
        float: Correlation coefficient.
    """
    return float(series_x.corr(series_y.shift(lag)))

def partial_correlation(
    df: pd.DataFrame, 
    x: str, 
    y: str, 
    covar: str
) -> float:
    """
    Calculates the partial correlation of variables x and y in a DataFrame,
    controlling for the confounding variable covar.

    Args:
        df: Input pandas DataFrame.
        x: Name of the first column.
        y: Name of the second column.
        covar: Name of the confounding variable column.

    Returns:
        float: Partial correlation coefficient.
    """
    # Ensure we drop any NaNs
    clean_df = df[[x, y, covar]].dropna()

    # Regress x on covar and get residuals
    slope_x, intercept_x, _, _, _ = stats.linregress(clean_df[covar], clean_df[x])
    x_res = clean_df[x] - (slope_x * clean_df[covar] + intercept_x)

    # Regress y on covar and get residuals
    slope_y, intercept_y, _, _, _ = stats.linregress(clean_df[covar], clean_df[y])
    y_res = clean_df[y] - (slope_y * clean_df[covar] + intercept_y)

    # Correlate the residuals
    r_val, _ = stats.pearsonr(x_res, y_res)
    return float(r_val)

def autocorrelation(
    series: pd.Series, 
    max_lag: int = 20
) -> Dict[int, float]:
    """
    Calculates the autocorrelation coefficients for a given time-series up to a max lag.

    Args:
        series: Target time-series.
        max_lag: Maximum lag to calculate autocorrelation for.

    Returns:
        Dict[int, float]: Dictionary mapping lag to autocorrelation coefficient.
    """
    return {lag: float(series.autocorr(lag=lag)) for lag in range(max_lag + 1)}

def mutual_information(
    series_x: Union[pd.Series, np.ndarray], 
    series_y: Union[pd.Series, np.ndarray]
) -> float:
    """
    Calculates the Mutual Information regression score between two continuous variables.
    Captures both linear and non-linear dependencies.

    Args:
        series_x: First variable.
        series_y: Second variable.

    Returns:
        float: Mutual Information regression score.
    """
    x_arr = np.asarray(series_x).reshape(-1, 1)
    y_arr = np.asarray(series_y)
    return float(mutual_info_regression(x_arr, y_arr, random_state=42)[0])

def tail_dependency_correlation(
    series_x: pd.Series, 
    series_y: pd.Series, 
    quantile: float = 0.15, 
    lower: bool = True
) -> float:
    """
    Calculates exceedance correlation in the tails of the joint distribution.
    Useful for measuring correlation shifts during extreme market conditions.

    Args:
        series_x: Returns of first asset.
        series_y: Returns of second asset.
        quantile: Threshold quantile (e.g. 0.15 for lower/upper 15%).
        lower: If True, calculates for the lower tail (crash).
               If False, calculates for the upper tail (boom).

    Returns:
        float: Tail exceedance correlation coefficient.
    """
    df = pd.DataFrame({"x": series_x, "y": series_y}).dropna()
    
    if lower:
        thresh_x = df["x"].quantile(quantile)
        thresh_y = df["y"].quantile(quantile)
        tail_df = df[(df["x"] < thresh_x) & (df["y"] < thresh_y)]
    else:
        thresh_x = df["x"].quantile(1 - quantile)
        thresh_y = df["y"].quantile(1 - quantile)
        tail_df = df[(df["x"] > thresh_x) & (df["y"] > thresh_y)]

    if len(tail_df) < 3:
        return np.nan
        
    return float(tail_df["x"].corr(tail_df["y"]))